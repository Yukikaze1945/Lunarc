#ifndef GPU_MULTISCALE_FUSION_SHADER_H
#define GPU_MULTISCALE_FUSION_SHADER_H
static constexpr char fusion_shader[] = R"hlsl(
#define TAPS 24u
struct TapRow { float W[TAPS]; int I[TAPS]; };
struct Edge { uint Dest; float Weight; };
Texture2D<float4> InputA : register(t0);
Texture2D<float4> InputB : register(t1);
StructuredBuffer<TapRow> Plan : register(t2);
StructuredBuffer<uint> Offsets : register(t3);
StructuredBuffer<Edge> Edges : register(t4);
ByteAddressBuffer FullColor : register(t5);
StructuredBuffer<TapRow> VerticalPlan : register(t6);
RWTexture2D<float4> Output : register(u0);
RWStructuredBuffer<float> Partial : register(u1);
RWTexture2D<float4> Final : register(u2);
cbuffer Frame : register(b0) {
    uint OutWidth, OutHeight, InWidth, InHeight;
    uint FullPitch, GroupWidth, Mode;
    float Scale;
};
[numthreads(8,8,1)]
void ForwardY(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    precise float4 total = float4(0,0,0,0);
    [loop] for (uint t=0; t<TAPS; ++t)
        total += Plan[id.y].W[t] * InputA.Load(int3(id.x,Plan[id.y].I[t],0));
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void ForwardX(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    precise float4 total = float4(0,0,0,0);
    [loop] for (uint t=0; t<TAPS; ++t)
        total += Plan[id.x].W[t] * InputA.Load(int3(Plan[id.x].I[t],id.y,0));
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void AdjointX(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    precise float4 total = float4(0,0,0,0);
    for (uint k=Offsets[id.x]; k<Offsets[id.x+1]; ++k)
        total += Edges[k].Weight * InputA.Load(int3(Edges[k].Dest,id.y,0));
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void AdjointY(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    precise float4 total = float4(0,0,0,0);
    for (uint k=Offsets[id.y]; k<Offsets[id.y+1]; ++k)
        total += Edges[k].Weight * InputA.Load(int3(id.x,Edges[k].Dest,0));
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void CatmullBaseline(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    precise float4 total = float4(0,0,0,0);
    [unroll] for (uint j=0; j<4; ++j) {
        precise float4 row = float4(0,0,0,0);
        [unroll] for (uint i=0; i<4; ++i)
            row += Plan[id.x].W[i] *
                   InputA.Load(int3(Plan[id.x].I[i],VerticalPlan[id.y].I[j],0));
        total += VerticalPlan[id.y].W[j] * row;
    }
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void Vector(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    float4 a = InputA.Load(int3(id.xy,0));
    Output[id.xy] = Mode == 1u ? Scale * a :
        a + Scale * InputB.Load(int3(id.xy,0));
}
[numthreads(8,8,1)]
void Composite(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    uint packed = FullColor.Load(id.y*FullPitch+id.x*4u);
    float3 original = float3(packed&255u,(packed>>8)&255u,(packed>>16)&255u);
    float3 residual = InputA.Load(int3(id.xy,0)).rgb;
    Final[id.xy] = float4(round(clamp(original+Scale*residual,0,255))/255.0,1);
}
/* --- Typed HDR (scRGB) path ---------------------------------------------
 *
 * The two entries below exist for the HDR solver only; every kernel above is
 * the code the SDR path has always run and none of them changed. The payload
 * they move is linear scRGB half at 8 bytes a sample, which is why they read
 * the transport plane by byte offset instead of the RGBA8 byte triplets above.
 * The half widening and the unrepresentable test are the same formulations the
 * HDR bridge's own shader uses, so one sample means one number in both. */
float2 unpack_half2(uint word) {
    return float2(f16tof32(word & 0xFFFFu), f16tof32(word >> 16));
}
/* A component the half grid cannot hold, or a NaN, fails both comparisons;
 * finite values inside +-65504 pass them. */
bool unrepresentable(float value) {
    return !(value <= 65504.0f && value >= -65504.0f);
}
float4 load_half4(ByteAddressBuffer plane, uint byte_offset) {
    uint2 word = plane.Load2(byte_offset);
    float2 low = unpack_half2(word.x);
    float2 high = unpack_half2(word.y);
    return float4(low.x, low.y, high.x, high.y);
}
/* Widens one borrowed RGBA16F plane into the solver's own FP32 texture. A half
 * sample is exactly representable in float, so this is a change of storage and
 * not of value: the objective then runs on the CG precision the solver already
 * had. */
[numthreads(8,8,1)]
void UnpackHalf(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight) return;
    Output[id.xy] = load_half4(FullColor, id.y*FullPitch + id.x*8u);
}
groupshared uint SharedUnrepresentable[64];
groupshared uint SharedUnrecoverable[64];
/* final = source + strength * residual, in signed linear scRGB.
 *
 * Nothing is clamped, nothing is renormalised and nothing is rounded to a byte:
 * a negative component is the out-of-gamut signal the encode is written to
 * receive and count, and a value above 1.0 is headroom above paper white. The
 * one thing a half destination cannot do is hold a value outside +-65504 or a
 * NaN, so such a pixel falls back to the pristine master sample and is counted
 * - the same rule, at the same per-pixel granularity and with the same
 * three-components-per-pixel basis, as the single-layer HDR composite. It is
 * never stored as an infinity and reported as a success.
 *
 * A group whose fallback sample was *itself* outside the grid - a master that
 * already carries a non-finite or unrepresentable component - has no finite
 * value it could legitimately store, so it writes a poison total the host
 * recognises and the call fails. 1e30 is far above the largest honest group
 * total (64 pixels * 3 components) and is exact in float32. */
[numthreads(8,8,1)]
void CompositeHdr(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID,
                  uint local : SV_GroupIndex) {
    uint bad = 0u;
    uint unrecoverable = 0u;
    if (id.x < OutWidth && id.y < OutHeight) {
        float4 source = load_half4(FullColor, id.y*FullPitch + id.x*8u);
        precise float3 residual = InputA.Load(int3(id.xy,0)).rgb;
        precise float3 result = source.rgb + Scale * residual;
        if (unrepresentable(result.x) || unrepresentable(result.y) ||
            unrepresentable(result.z)) {
            if (unrepresentable(source.x) || unrepresentable(source.y) ||
                unrepresentable(source.z)) unrecoverable = 1u;
            result = source.rgb;
            bad = 3u;
        }
        Final[id.xy] = float4(result, 1.0);
    }
    SharedUnrepresentable[local] = bad;
    SharedUnrecoverable[local] = unrecoverable;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=32; stride>0; stride>>=1) {
        if (local<stride) {
            SharedUnrepresentable[local] += SharedUnrepresentable[local+stride];
            SharedUnrecoverable[local] += SharedUnrecoverable[local+stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    /* The per-group total goes through the buffer the CG already reduces with,
     * so the count reaches the host as a scalar and no picture ever does. */
    if (local==0) Partial[group.y*GroupWidth+group.x] =
        SharedUnrecoverable[0] > 0u ? 1.0e30f
                                    : float(SharedUnrepresentable[0]);
}
groupshared float SharedSum[64];
[numthreads(8,8,1)]
void Dot(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID,
         uint local : SV_GroupIndex) {
    float value = 0;
    if (id.x < OutWidth && id.y < OutHeight) {
        float3 a = InputA.Load(int3(id.xy,0)).rgb;
        float3 b = InputB.Load(int3(id.xy,0)).rgb;
        value = dot(a,b);
    }
    SharedSum[local] = value;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=32; stride>0; stride>>=1) {
        if (local<stride) SharedSum[local]+=SharedSum[local+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (local==0) Partial[group.y*GroupWidth+group.x]=SharedSum[0];
}
groupshared float ReduceSum[256];
[numthreads(256,1,1)]
void Reduce(uint3 group : SV_GroupID, uint local : SV_GroupIndex) {
    uint index = group.x * 256u + local;
    ReduceSum[local] = index < OutWidth ? asfloat(FullColor.Load(index * 4u)) : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride=128u; stride>0u; stride>>=1u) {
        if (local<stride) ReduceSum[local] += ReduceSum[local+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (local==0u) Partial[group.x] = ReduceSum[0];
}
)hlsl";
#endif
