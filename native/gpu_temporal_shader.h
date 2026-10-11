#ifndef GPU_TEMPORAL_SHADER_H
#define GPU_TEMPORAL_SHADER_H

/* Conservative frame continuity scoring plus explicitly opt-in static-region
 * low-frequency residual stabilization. Inputs remain GPU-resident. */
static constexpr char gpu_temporal_shader[] = R"hlsl(
ByteAddressBuffer Source : register(t0);
ByteAddressBuffer Candidate : register(t1);
ByteAddressBuffer PreviousState : register(t2);
ByteAddressBuffer PreviousGuide : register(t3);
ByteAddressBuffer UnusedInput : register(t4);
ByteAddressBuffer CurrentGuide : register(t5);
RWTexture2D<float4> Output : register(u0);
RWByteAddressBuffer StateOut : register(u1);
RWByteAddressBuffer StatsOut : register(u2);
cbuffer Frame : register(b0) {
    uint Width, Height, FullPitch, PixelBytes;
    uint GuideWidth, GuideHeight, HistoryValid, Reset;
    float HistoryWeight;
};

static const float3 Luma = float3(0.2126,0.7152,0.0722);
float4 load_pixel(ByteAddressBuffer plane, uint2 p) {
    uint address=p.y*FullPitch+p.x*PixelBytes;
    if (PixelBytes==4u) {
        uint word=plane.Load(address);
        return float4(word&255u,(word>>8)&255u,(word>>16)&255u,
                      (word>>24)&255u)/255.0;
    }
    uint2 words=plane.Load2(address);
    return float4(f16tof32(words.x&0xffffu),f16tof32(words.x>>16),
                  f16tof32(words.y&0xffffu),f16tof32(words.y>>16));
}
float4 load_raw4(ByteAddressBuffer plane,uint offset) {
    return asfloat(plane.Load4(offset));
}
uint2 guide_pixel(uint index) {
    uint gx=index%GuideWidth, gy=index/GuideWidth;
    return min(uint2(Width-1u,Height-1u),
        uint2((gx*Width+Width/2u)/GuideWidth,
              (gy*Height+Height/2u)/GuideHeight));
}

/* One small GPU thread group samples the fixed guide and normalizes RGB by
 * that frame's sampled absolute peak, making the shape/chroma comparison
 * robust to a uniform exposure scale. This is a sampled detector, not a
 * semantic scene classifier. */
[numthreads(1,1,1)]
void BuildGuide(uint3 unused : SV_DispatchThreadID) {
    float peak=0.0;
    uint count=GuideWidth*GuideHeight;
    [loop] for (uint i=0;i<count;++i) {
        float3 c=load_pixel(Source,guide_pixel(i)).rgb;
        if (all(isfinite(c))) peak=max(peak,max(abs(c.r),max(abs(c.g),abs(c.b))));
    }
    float divisor=max(peak,1.0e-8);
    [loop] for (uint j=0;j<count;++j) {
        float3 c=load_pixel(Source,guide_pixel(j)).rgb;
        if (!all(isfinite(c))) c=0.0;
        StateOut.Store4(j*16u,asuint(float4(c/divisor,peak)));
    }
}

/* Search bounded +/-4 guide-cell translations. The best mean RGB distance,
 * chroma distance, and local-match support are reduced by one GPU thread; the
 * readback contains only eight floats of diagnostics. */
[numthreads(1,1,1)]
void CompareGuide(uint3 unused : SV_DispatchThreadID) {
    if (HistoryValid==0u || Reset!=0u) {
        [unroll] for (uint k=0;k<8u;++k) StatsOut.Store(k*4u,0u);
        return;
    }
    float best=3.402823466e+38, best_chroma=0.0, best_support=0.0;
    int best_x=0,best_y=0;
    [loop] for (int dy=-4;dy<=4;++dy) {
        [loop] for (int dx=-4;dx<=4;++dx) {
            float total=0.0,chroma=0.0;
            uint matched=0,overlap=0;
            [loop] for (uint y=0;y<GuideHeight;++y) {
                int py=int(y)+dy;
                if (py<0 || py>=int(GuideHeight)) continue;
                [loop] for (uint x=0;x<GuideWidth;++x) {
                    int px=int(x)+dx;
                    if (px<0 || px>=int(GuideWidth)) continue;
                    uint at=(y*GuideWidth+x)*16u;
                    uint prev=((uint(py)*GuideWidth+uint(px))*16u);
                    float4 a=asfloat(CurrentGuide.Load4(at));
                    float4 b=asfloat(PreviousGuide.Load4(prev));
                    float3 delta=abs(a.rgb-b.rgb);
                    float distance=(delta.r+delta.g+delta.b)*(1.0/3.0);
                    float3 ca=a.rgb-dot(a.rgb,Luma).xxx;
                    float3 cb=b.rgb-dot(b.rgb,Luma).xxx;
                    total+=distance;
                    chroma+=dot(abs(ca-cb),1.0.xxx)/3.0;
                    matched+=distance<0.08 ? 1u : 0u;
                    overlap++;
                }
            }
            float mean=overlap>0u ? total/overlap : 3.402823466e+38;
            if (mean<best || (mean==best && matched>best_support*overlap)) {
                best=mean;
                best_chroma=overlap>0u ? chroma/overlap : 0.0;
                best_support=overlap>0u ? float(matched)/overlap : 0.0;
                best_x=dx; best_y=dy;
            }
        }
    }
    float current_peak=asfloat(CurrentGuide.Load(12u));
    float prior_peak=asfloat(PreviousGuide.Load(12u));
    float ratio=prior_peak>1.0e-8 ? current_peak/prior_peak : 1.0;
    StatsOut.Store(0u,asuint(best));
    StatsOut.Store(4u,asuint(best_support));
    StatsOut.Store(8u,asuint(best_chroma));
    StatsOut.Store(12u,asuint(ratio));
    StatsOut.Store(16u,asuint(float(best_x)));
    StatsOut.Store(20u,asuint(float(best_y)));
    StatsOut.Store(24u,asuint(current_peak));
    StatsOut.Store(28u,asuint(prior_peak));
}

float3 residual_at(uint2 p) {
    return load_pixel(Candidate,p).rgb-load_pixel(Source,p).rgb;
}
float3 low_residual(uint2 p) {
    const float w[3]={1.0,2.0,1.0};
    float3 sum=0.0;
    [unroll] for (int oy=-1;oy<=1;++oy) {
        [unroll] for (int ox=-1;ox<=1;++ox) {
            uint2 q=uint2(clamp(int2(p)+int2(ox,oy),int2(0,0),
                int2(Width-1u,Height-1u)));
            sum+=residual_at(q)*(w[ox+1]*w[oy+1]*(1.0/16.0));
        }
    }
    return sum;
}
bool stable_3x3(uint2 p) {
    [unroll] for (int oy=-1;oy<=1;++oy) {
        [unroll] for (int ox=-1;ox<=1;++ox) {
            uint2 q=uint2(clamp(int2(p)+int2(ox,oy),int2(0,0),
                int2(Width-1u,Height-1u)));
            float3 now=load_pixel(Source,q).rgb;
            float3 old=load_raw4(PreviousState,(q.y*Width+q.x)*16u).rgb;
            float threshold=PixelBytes==8u ? 1.0e-4 : 1.0/1024.0;
            float3 tolerance=max(max(abs(now),abs(old))*0.01,threshold.xxx);
            if (any(abs(now-old)>tolerance)) return false;
        }
    }
    return true;
}
bool representable_half(float3 value) {
    return all(isfinite(value)) && all(abs(value)<=65504.0);
}
bool representable_half4(float4 value) {
    return all(isfinite(value)) && all(abs(value)<=65504.0);
}
bool finite_neighborhood(uint2 p) {
    [unroll] for (int oy=-1;oy<=1;++oy) {
        [unroll] for (int ox=-1;ox<=1;++ox) {
            uint2 q=uint2(clamp(int2(p)+int2(ox,oy),int2(0,0),
                int2(Width-1u,Height-1u)));
            float4 s=load_pixel(Source,q),c=load_pixel(Candidate,q);
            if (!all(isfinite(s)) || !all(isfinite(c))) return false;
            if (PixelBytes==8u &&
                (!representable_half4(s) || !representable_half4(c))) return false;
        }
    }
    return true;
}
[numthreads(8,8,1)]
void TemporalProcess(uint3 id : SV_DispatchThreadID) {
    if (id.x>=Width || id.y>=Height) return;
    uint index=id.y*Width+id.x;
    float4 s=load_pixel(Source,id.xy);
    float4 c=load_pixel(Candidate,id.xy);
    bool valid=all(isfinite(s)) && all(isfinite(c)) && finite_neighborhood(id.xy);
    if (PixelBytes==8u) valid=valid && representable_half4(s) &&
        representable_half4(c);
    if (!valid) {
        uint ignored;
        StatsOut.InterlockedAdd(0u,1u,ignored);
        Output[id.xy]=0.0;
        StateOut.Store4(index*16u,asuint(float4(0.0,0.0,0.0,0.0)));
        StateOut.Store4((Width*Height+index)*16u,
                        asuint(float4(0.0,0.0,0.0,0.0)));
        return;
    }
    float3 low=low_residual(id.xy);
    StateOut.Store4(index*16u,asuint(s));
    StateOut.Store4((Width*Height+index)*16u,asuint(float4(low,0.0)));
    if (HistoryValid==0u || Reset!=0u || HistoryWeight<=0.0) {
        Output[id.xy]=c;
        return;
    }
    float3 result=c.rgb;
    bool stable=stable_3x3(id.xy);
    float3 previous_low=load_raw4(PreviousState,
        (Width*Height+index)*16u).rgb;
    float3 current_low=low;
    float low_norm=dot(current_low,current_low);
    float same_sign=dot(current_low,previous_low);
    float previous_norm=dot(previous_low,previous_low);
    float min_low=PixelBytes==8u ? 1.0e-8 : 1.0e-7;
    float magnitude_tolerance=max(previous_norm*0.69,min_low);
    bool accepted=stable && low_norm>min_low && same_sign>0.0 &&
        all(isfinite(previous_low)) &&
        abs(low_norm-previous_norm)<=magnitude_tolerance;
    if (accepted) {
        result=c.rgb+HistoryWeight*(previous_low-current_low);
        uint ignored;
        StatsOut.InterlockedAdd(8u,1u,ignored);
    } else {
        uint ignored;
        StatsOut.InterlockedAdd(12u,1u,ignored);
    }
    if (PixelBytes==8u && !representable_half(result)) {
        result=s.rgb;
        uint ignored;
        StatsOut.InterlockedAdd(4u,1u,ignored);
    }
    Output[id.xy]=float4(result,c.a);
}
)hlsl";

#endif
