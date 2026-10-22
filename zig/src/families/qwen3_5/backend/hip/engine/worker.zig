//! A tensor-parallel rank above 0: runs the steps rank 0 sends in the same order, so the collectives pair up.

const std = @import("std");
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const Pick = @import("engine.zig").Pick;
const state = @import("../forward/state.zig");
const draw = @import("hip_serve").draw;

/// A message's first word: rank 0 decides every match, eviction, page and slot, and a rank only applies it.
pub const Op = enum(u32) { stop, prefill, verify, keep, release, fill, pages, snap, copy };

/// The prefill message's snapshot slot when the pass starts from nothing.
pub const no_snapshot: u32 = std.math.maxInt(u32);

/// Windows a round holds at most.
const max_windows = 128;

const Fill = struct { prompt: []u32, at: usize };

/// Words [at, at + n) of a message; one shorter than its own counts say is refused, not read past.
fn words(m: []const u32, at: usize, n: usize) ![]const u32 {
    if (at > m.len or n > m.len - at) return error.BadMessage;
    return m[at..][0..n];
}

const Lane = struct {
    caches: state.Caches,
    /// Slots written and kept: the next window starts here.
    len: usize,
    /// The prompt pass in progress: the prompt and the row it has reached.
    fill: ?Fill = null,
    /// The last verify's slot and rows, until rank 0's keep.
    pending: ?struct { window: usize, rows: usize } = null,
};

pub const Worker = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(u32, *Lane) = .empty,
    reqs: []draw.Request,
    out: []u32,
    /// The linear snapshots of rank 0's prefix tree, in the slots it names.
    snaps: state.Snapshots,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !Worker {
        const rows = e.o.batch_rows;
        const reqs = try gpa.alloc(draw.Request, rows);
        errdefer gpa.free(reqs);
        // a follower's draws are greedy and unread: the forward and its collectives are what it shares
        @memset(reqs, .{ .sampling = null, .position = 0 });
        return .{ .gpa = gpa, .e = e, .reqs = reqs, .out = try gpa.alloc(u32, rows), .snaps = state.Snapshots.init(gpa, &e.driver) };
    }

    pub fn deinit(w: *Worker) void {
        w.e.stream.synchronize() catch {};
        w.snaps.deinit();
        var it = w.lanes.valueIterator();
        while (it.next()) |l| w.destroy(l.*);
        w.lanes.deinit(w.gpa);
        w.gpa.free(w.out);
        w.gpa.free(w.reqs);
    }

    fn destroy(w: *Worker, l: *Lane) void {
        if (l.fill) |f| w.gpa.free(f.prompt);
        w.e.drain();
        l.caches.deinit(w.gpa);
        w.gpa.destroy(l);
    }

    fn prefill(w: *Worker, id: u32, total: usize, prompt: []const u32, at: usize, snap: ?u32, held: []const u32) !void {
        for (held) |page| if (page >= w.e.pool.count) return error.BadMessage;
        const gop = try w.lanes.getOrPut(w.gpa, id);
        if (gop.found_existing) w.destroy(gop.value_ptr.*);
        errdefer w.lanes.removeByPtr(gop.key_ptr);
        const lane = try w.gpa.create(Lane);
        errdefer w.gpa.destroy(lane);
        lane.* = .{ .caches = try w.e.emptyCaches(total), .len = prompt.len };
        errdefer lane.caches.deinit(w.gpa);
        try lane.caches.set(w.gpa, 0, held);
        if (snap) |slot| try w.snaps.put(slot, &lane.caches, w.e.stream.handle);
        lane.fill = .{ .prompt = try w.gpa.dupe(u32, prompt), .at = at };
        gop.value_ptr.* = lane;
    }

    /// The next chunk of a prompt pass, to row `to`: rank 0's cuts and chunk ends, run the same way.
    fn fill(w: *Worker, id: u32, to: usize) !void {
        const lane = w.lanes.get(id) orelse return error.UnknownStream;
        const f = &(lane.fill orelse return error.NoPromptPass);
        if (to < f.at or to > f.prompt.len) return error.BadMessage;
        if (to == f.prompt.len) {
            _ = try w.e.prefill(&lane.caches, f.prompt, f.at, null, w.reqs[0], null);
            w.gpa.free(f.prompt);
            lane.fill = null;
            return;
        }
        try w.e.advance(&lane.caches, f.prompt, f.at, to, null);
        f.at = to;
    }

    fn laneOf(w: *Worker, id: u32) !*Lane {
        return w.lanes.get(id) orelse error.UnknownStream;
    }

    fn verify(w: *Worker, ids: []const u32, rows: []const Engine.Rows, pick: Pick) !void {
        var total: usize = 0;
        for (rows) |r| total += r.tokens.len;
        if (total > w.reqs.len) return error.BadMessage;
        _ = try w.e.choose(rows, pick);
        _ = try w.e.verify(rows, w.reqs[0..total], w.out[0..total]);
        for (ids, 0..) |id, i| w.lanes.get(id).?.pending = .{ .window = i, .rows = rows[i].tokens.len };
    }

    fn keep(w: *Worker, id: u32, rows: usize) !void {
        const lane = w.lanes.get(id) orelse return error.UnknownStream;
        const p = lane.pending orelse return error.NothingToKeep;
        if (rows > p.rows) return error.BadMessage;
        w.e.keep(p.window, rows);
        lane.len += rows;
        lane.pending = null;
    }

    /// The stream's page table from index `first` on: within the table, and every page in the pool.
    fn pages(w: *Worker, id: u32, first: usize, ids: []const u32) !void {
        const caches = &(try w.laneOf(id)).caches;
        if (first > caches.table.items.len) return error.BadMessage;
        for (ids) |page| if (page >= w.e.pool.count) return error.BadMessage;
        try caches.set(w.gpa, first, ids);
    }

    fn release(w: *Worker, id: u32) void {
        const kv = w.lanes.fetchRemove(id) orelse return;
        w.e.stream.synchronize() catch {};
        w.destroy(kv.value);
    }

    /// Runs rank 0's steps until it says stop.
    pub fn follow(w: *Worker, link: hip.collective.Link) !void {
        var msg: std.ArrayList(u32) = .empty;
        defer msg.deinit(w.gpa);
        var ids: [max_windows]u32 = undefined;
        var rows: [max_windows]Engine.Rows = undefined;
        while (true) {
            try link.recv(w.gpa, &msg);
            const m = msg.items;
            if (m.len == 0) return error.BadMessage;
            switch (std.enums.fromInt(Op, m[0]) orelse return error.BadMessage) {
                .stop => return,
                // prefill: id, positions, prompt len, resumed at, cut count, snapshot slot, page count, then the pages
                .prefill => {
                    const head = try words(m, 0, 8);
                    const held = try words(m, 8, head[7]);
                    // the cuts sit between the pages and the prompt
                    const prompt = try words(m, 8 + @as(usize, head[7]) + head[5], head[3]);
                    try w.prefill(head[1], head[2], prompt, head[4], if (head[6] == no_snapshot) null else head[6], held);
                },
                // verify: graph pick, count, then id, rows, tokens each; every rank derives the plan shape from them
                .verify => {
                    const head = try words(m, 0, 3);
                    const pick = std.enums.fromInt(Pick, head[1]) orelse return error.BadMessage;
                    const n = head[2];
                    if (n > max_windows) return error.WindowTooWide;
                    var at: usize = 3;
                    for (0..n) |i| {
                        const window = try words(m, at, 2);
                        const lane = w.lanes.get(window[0]) orelse return error.UnknownStream;
                        ids[i] = window[0];
                        rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = try words(m, at + 2, window[1]) };
                        at += 2 + window[1];
                    }
                    try w.verify(ids[0..n], rows[0..n], pick);
                },
                // keep: count, then id and rows each
                .keep => {
                    const n = (try words(m, 0, 2))[1];
                    const kept = try words(m, 2, 2 * @as(usize, n));
                    for (0..n) |i| try w.keep(kept[2 * i], kept[2 * i + 1]);
                    try w.e.flush();
                },
                .release => w.release((try words(m, 0, 2))[1]),
                // fill: id, end of the prompt pass's next chunk
                .fill => {
                    const head = try words(m, 0, 3);
                    try w.fill(head[1], head[2]);
                },
                // pages: id, first page index, count, then the stream's table from that index on
                .pages => {
                    const head = try words(m, 0, 4);
                    try w.pages(head[1], head[2], try words(m, 4, head[3]));
                },
                // snap: id, slot that keeps the stream's linear state
                .snap => {
                    const head = try words(m, 0, 3);
                    try w.snaps.take(head[2], &(try w.laneOf(head[1])).caches, w.e.stream.handle);
                },
                // copy: from, to; page `to` becomes a copy of page `from`
                .copy => {
                    const head = try words(m, 0, 3);
                    try w.e.pool.copyPage(head[1], head[2], w.e.stream.handle);
                },
            }
        }
    }
};

test "a message shorter than its counts is refused" {
    const m = [_]u32{ 1, 2, 3 };
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, try words(&m, 1, 2));
    try std.testing.expectEqualSlices(u32, &.{}, try words(&m, 3, 0));
    try std.testing.expectError(error.BadMessage, words(&m, 2, 2));
    try std.testing.expectError(error.BadMessage, words(&m, 4, 0));
    try std.testing.expectError(error.BadMessage, words(&m, 1, std.math.maxInt(usize)));
}
