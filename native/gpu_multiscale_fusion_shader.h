#ifndef GPU_MULTISCALE_FUSION_SHADER_H
#define GPU_MULTISCALE_FUSION_SHADER_H
static constexpr char fusion_shader[] = R"hlsl(
#define TAPS 24u
struct TapRow { float W[TAPS]; int I[TAPS]; float NormSquare; };
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
ByteAddressBuffer Scalar : register(t7);
struct CgPacket {
    float Square, Initial, Alpha, Gamma;
    uint Status, Iterations;
    float Tolerance, Reserved;
};
RWStructuredBuffer<CgPacket> Cg : register(u3);
cbuffer Frame : register(b0) {
    uint OutWidth, OutHeight, TapCount, BetaBits;
    uint FullPitch, GroupWidth, Mode;
    float Scale;
};
bool cg_inactive() { if ((Mode & 256u) == 0u) return false; return Scalar.Load(16) != 0u; }
[numthreads(8,8,1)]
void ForwardY(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight || cg_inactive()) return;
    precise float4 total = float4(0,0,0,0);
    #ifdef FUSION_TAP_COUNT
    [unroll] for (uint t=0; t<FUSION_TAP_COUNT; ++t)
    #else
    [loop] for (uint t=0; t<TapCount; ++t)
    #endif
        total += Plan[id.y].W[t] * InputA.Load(int3(id.x,Plan[id.y].I[t],0));
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void ForwardX(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight || cg_inactive()) return;
    precise float4 total = float4(0,0,0,0);
    #ifdef FUSION_TAP_COUNT
    [unroll] for (uint t=0; t<FUSION_TAP_COUNT; ++t)
    #else
    [loop] for (uint t=0; t<TapCount; ++t)
    #endif
        total += Plan[id.x].W[t] * InputA.Load(int3(Plan[id.x].I[t],id.y,0));
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void AdjointX(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight || cg_inactive()) return;
    precise float4 total = float4(0,0,0,0);
    for (uint k=Offsets[id.x]; k<Offsets[id.x+1]; ++k)
        total += Edges[k].Weight * InputA.Load(int3(Edges[k].Dest,id.y,0));
    Output[id.xy] = total;
}
[numthreads(8,8,1)]
void AdjointY(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OutWidth || id.y >= OutHeight || cg_inactive()) return;
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
    if (cg_inactive()) return;
    float4 a = InputA.Load(int3(id.xy,0));
    uint operation = Mode & 255u;
    if (operation == 2u) Output[id.xy] += asfloat(Scalar.Load(8)) * a;
    else if (operation == 3u) Output[id.xy] -= asfloat(Scalar.Load(8)) * a;
    else if (operation == 4u) Output[id.xy] = a + asfloat(Scalar.Load(12)) * Output[id.xy];
    else if (operation == 5u) Output[id.xy] += Scale * a;
    else if (operation == 6u) Output[id.xy] = Scale * Output[id.xy] + asfloat(BetaBits) * a;
    else if (operation == 7u) Output[id.xy] = Scale * (a - InputB.Load(int3(id.xy,0))) - Output[id.xy];
    else if (operation == 8u) Output[id.xy] = a /
        (asfloat(BetaBits) + Scale*Plan[id.x].NormSquare*VerticalPlan[id.y].NormSquare);
    else Output[id.xy] = operation == 1u ? Scale * a :
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
 * The two entries below exist for the HDR colour contract; the shared kernels
 * above solve both colour paths with FP32 working vectors. The payload
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
    if (!cg_inactive() && id.x < OutWidth && id.y < OutHeight) {
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
/* Control and subsequent vector passes use the same GPU packet. Status:
 * 0 active, 1 converged, 2 invalid initial norm, 3 invalid CG arithmetic. */
[numthreads(1,1,1)]
void CgControl(uint3 id : SV_DispatchThreadID) {
    float value = (Mode & 512u) != 0u ? Cg[0].Reserved : asfloat(Scalar.Load(0));
    uint operation = Mode & 255u;
    CgPacket state;
    if (operation == 0u) {
        state.Square = value; state.Initial = value;
        state.Alpha = 0; state.Gamma = 0; state.Iterations = 0;
        state.Tolerance = Scale; state.Reserved = 0;
        state.Status = !isfinite(value) || value < 0 ? 2u : value == 0 ? 1u : 0u;
    } else {
        state = Cg[0];
        if (operation == 4u) state.Reserved = 0;
        else if (operation == 5u) state.Reserved += value;
        else if (operation == 6u) {
            state.Square = value; state.Alpha = 0; state.Gamma = 0;
            state.Status = !isfinite(value) || value < 0 ? 2u : state.Initial == 0 ? 1u : 0u;
        }
        else if (operation == 3u) { // Reliable residual replacement; retain target/count.
            state.Square = value; state.Alpha = 0; state.Gamma = 0;
            state.Status = !isfinite(value) || value < 0 ? 3u : 0u;
        } else if (state.Status == 0u) {
            if (operation == 1u) {
                if (!isfinite(value) || value <= 0) state.Status = 3u;
                else {
                    state.Alpha = state.Square / value;
                    if (!isfinite(state.Alpha)) state.Status = 3u;
                }
            } else if (operation == 2u) {
                if (!isfinite(value) || value < 0) state.Status = 3u;
                else {
                    state.Gamma = value / state.Square;
                    state.Square = value; ++state.Iterations;
                    if (!isfinite(state.Gamma)) state.Status = 3u;
                    else if (value <= state.Tolerance*state.Tolerance*state.Initial)
                        state.Status = 1u;
                }
            }
        }
    }
    Cg[0] = state;
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
