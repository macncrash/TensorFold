//! Kimi K3's text model shape from config.json's text_config, and which layers are KDA or MLA.
const std = @import("std");

pub const Kind = enum { kda, mla };

pub const Config = struct {
    hidden: u32 = 7168,
    layers: u32 = 93,
    vocab: u32 = 163840,
    eps: f32 = 1e-5,
    dense_layers: u32 = 1,
    dense_inter: u32 = 33792,
    experts: u32 = 896,
    topk: u32 = 16,
    shared_inter: u32 = 6144,
    moe_inter: u32 = 3072,
    latent: u32 = 3584,
    kda_heads: u32 = 96,
    kda_dim: u32 = 128,
    conv: u32 = 4,
    lower_bound: f32 = -5.0,
    mla_heads: u32 = 96,
    q_lora: u32 = 1536,
    kv_lora: u32 = 512,
    nope: u32 = 128,
    rope: u32 = 64,
    v_dim: u32 = 128,
    block: u32 = 12,
    situ_beta: f32 = 4.0,
    situ_linear: f32 = 25.0,
    routed_scale: f32 = 1.0,
    /// 0-based layer i is MLA when i + 1 is in config's 1-based full_attn_layers.
    full_attn: std.StaticBitSet(128) = defaultFull(),

    pub fn kind(c: *const Config, i: u32) Kind {
        return if (c.full_attn.isSet(i)) .mla else .kda;
    }

    pub fn isMoe(c: *const Config, i: u32) bool {
        return i >= c.dense_layers;
    }

    pub fn kdaWidth(c: *const Config) u32 {
        return c.kda_heads * c.kda_dim;
    }

    pub fn qHead(c: *const Config) u32 {
        return c.nope + c.rope;
    }

    /// Attention-residual block entries a row carries into layer i's first residual.
    pub fn blocksBefore(c: *const Config, i: u32) u32 {
        return (i + c.block - 1) / c.block;
    }

    pub fn countKind(c: *const Config, k: Kind) u32 {
        var n: u32 = 0;
        for (0..c.layers) |i| n += @intFromBool(c.kind(@intCast(i)) == k);
        return n;
    }

    /// Dense index of layer i among the layers of its kind (state arenas are per kind).
    pub fn kindIndex(c: *const Config, i: u32) u32 {
        var n: u32 = 0;
        for (0..i) |j| n += @intFromBool(c.kind(@intCast(j)) == c.kind(i));
        return n;
    }
};

fn defaultFull() std.StaticBitSet(128) {
    var s = std.StaticBitSet(128).empty;
    var l: u32 = 4;
    while (l <= 92) : (l += 4) s.set(l - 1);
    s.set(92);
    return s;
}

/// The text_config of a Kimi K3 config.json; unknown or vision fields are ignored, and a field of the wrong type or range is refused.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    const t = try object(root.get("text_config") orelse parsed.value);
    var c = Config{};
    const ints = .{
        .{ "hidden_size", "hidden" },               .{ "num_hidden_layers", "layers" },
        .{ "vocab_size", "vocab" },                 .{ "first_k_dense_replace", "dense_layers" },
        .{ "intermediate_size", "dense_inter" },    .{ "num_experts", "experts" },
        .{ "num_experts_per_token", "topk" },       .{ "moe_intermediate_size", "moe_inter" },
        .{ "routed_expert_hidden_size", "latent" }, .{ "num_attention_heads", "mla_heads" },
        .{ "q_lora_rank", "q_lora" },               .{ "kv_lora_rank", "kv_lora" },
        .{ "qk_nope_head_dim", "nope" },            .{ "qk_rope_head_dim", "rope" },
        .{ "v_head_dim", "v_dim" },                 .{ "attn_res_block_size", "block" },
    };
    inline for (ints) |p| if (t.get(p[0])) |v| {
        @field(c, p[1]) = try int(v);
    };
    const floats = .{
        .{ "rms_norm_eps", "eps" },                        .{ "activation_situ_beta", "situ_beta" },
        .{ "activation_situ_linear_beta", "situ_linear" }, .{ "routed_scaling_factor", "routed_scale" },
    };
    inline for (floats) |p| if (t.get(p[0])) |v| {
        @field(c, p[1]) = try float(v);
    };
    const shared = if (t.get("num_shared_experts")) |v| try int(v) else 0;
    c.shared_inter = std.math.mul(u32, c.moe_inter, shared) catch return error.BadConfig;
    if (t.get("linear_attn_config")) |la| {
        const o = try object(la);
        c.kda_heads = try int(o.get("num_heads") orelse return error.BadConfig);
        c.kda_dim = try int(o.get("head_dim") orelse return error.BadConfig);
        c.conv = try int(o.get("short_conv_kernel_size") orelse return error.BadConfig);
        if (o.get("gate_lower_bound")) |v| c.lower_bound = try float(v);
        const full = o.get("full_attn_layers") orelse return error.BadConfig;
        if (full != .array) return error.BadConfig;
        c.full_attn = std.StaticBitSet(128).empty;
        // 1-based layer numbers, each a layer the bit set holds
        for (full.array.items) |v| {
            const layer = try int(v);
            if (layer == 0 or layer > c.full_attn.capacity()) return error.BadConfig;
            c.full_attn.set(layer - 1);
        }
    }
    if (c.layers > 128) return error.TooManyLayers;
    return c;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadConfig;
}

fn int(v: std.json.Value) !u32 {
    if (v != .integer) return error.BadConfig;
    return std.math.cast(u32, v.integer) orelse error.BadConfig;
}

fn float(v: std.json.Value) !f32 {
    return @floatCast(switch (v) {
        .float => |f| f,
        .integer => |i| @as(f64, @floatFromInt(i)),
        else => return error.BadConfig,
    });
}

test "K3's defaults: 69 KDA and 24 MLA layers, layer 0 KDA, layers 3 and 92 MLA" {
    const c = Config{};
    try std.testing.expectEqual(@as(u32, 69), c.countKind(.kda));
    try std.testing.expectEqual(@as(u32, 24), c.countKind(.mla));
    try std.testing.expectEqual(Kind.kda, c.kind(0));
    try std.testing.expectEqual(Kind.mla, c.kind(3));
    try std.testing.expectEqual(Kind.mla, c.kind(92));
    try std.testing.expectEqual(Kind.mla, c.kind(91));
    try std.testing.expectEqual(@as(u32, 23), c.kindIndex(92));
    try std.testing.expectEqual(@as(u32, 8), c.blocksBefore(85));
}

test "a config field of the wrong type or range is refused" {
    const a = std.testing.allocator;
    const c = try parse(a,
        \\{"text_config": {"num_hidden_layers": 4, "moe_intermediate_size": 8, "num_shared_experts": 2,
        \\ "linear_attn_config": {"num_heads": 2, "head_dim": 4, "short_conv_kernel_size": 4, "gate_lower_bound": -5, "full_attn_layers": [2, 128]}}}
    );
    try std.testing.expectEqual(@as(u32, 16), c.shared_inter);
    try std.testing.expectEqual(Kind.mla, c.kind(1));
    try std.testing.expectEqual(Kind.mla, c.kind(127));
    try std.testing.expectEqual(Kind.kda, c.kind(0));
    inline for (.{
        "[]",
        "{\"text_config\": 1}",
        "{\"num_hidden_layers\": -1}",
        "{\"num_hidden_layers\": 4294967296}",
        "{\"hidden_size\": \"7168\"}",
        "{\"rms_norm_eps\": \"x\"}",
        "{\"moe_intermediate_size\": 100000, \"num_shared_experts\": 100000}",
        "{\"linear_attn_config\": []}",
        "{\"linear_attn_config\": {\"head_dim\": 4, \"short_conv_kernel_size\": 4, \"full_attn_layers\": []}}",
        "{\"linear_attn_config\": {\"num_heads\": 2, \"head_dim\": 4, \"short_conv_kernel_size\": 4, \"gate_lower_bound\": true, \"full_attn_layers\": []}}",
        "{\"linear_attn_config\": {\"num_heads\": 2, \"head_dim\": 4, \"short_conv_kernel_size\": 4, \"full_attn_layers\": 3}}",
        "{\"linear_attn_config\": {\"num_heads\": 2, \"head_dim\": 4, \"short_conv_kernel_size\": 4, \"full_attn_layers\": [0]}}",
        "{\"linear_attn_config\": {\"num_heads\": 2, \"head_dim\": 4, \"short_conv_kernel_size\": 4, \"full_attn_layers\": [129]}}",
    }) |json| try std.testing.expectError(error.BadConfig, parse(a, json));
}
