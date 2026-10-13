#ifndef GPU_POST_CONTROL_SHADER_H
#define GPU_POST_CONTROL_SHADER_H

/* Independently derived residual-control shader. Inputs are normalized SDR
 * RGBA8 code or signed linear scRGB RGBA16F in Rec.709 primaries.
 *
 * For each RGB pixel, raw = candidate - source. The low residual is a
 * normalized 3x3 [1,2,1]x[1,2,1]/16 binomial neighborhood, with each spatial
 * weight multiplied by exp(-dot(neighbor-source,neighbor-source) /
 * (0.0004 + 0.04*dot(source,source))). high = raw-low. Lightness is
 * controlled in Oklab, with source a/b retained when chroma gain is zero.
 * Hue protection uses the source's Oklab hue direction and fades near gray.
 * Total gain and frequency controls act on RGB residuals before the Oklab
 * lightness/chroma controls. ROI is a protected region: its interior attenuates the
 * residual to zero, the outside retains the full residual, and feathering
 * transitions from full residual at the boundary to zero inside. Dark and
 * highlight weights also attenuate the residual.
 * Dark protection uses abs(source luma)/(0.02*scenePeak) for SDR and
 * abs(source luma)/(0.005*scenePeak) for HDR (with unit floors 1e-4 and
 * 0.0125 scRGB respectively). Highlight classification fades in over the top
 * 2% of the source RGB peak. HDR additionally attenuates only positive
 * residuals that would cross 125 BT.2020 linear scRGB (10000 nit PQ); it does
 * not clamp source values or impose a general HDR gamut/range clamp. Soft
 * compression is identity through 25% of max(length(source)+scenePeak,
 * unitFloor), then asymptotically reduces excess residual magnitude. Invalid
 * candidate pixels fall back to source; invalid source pixels fail in the
 * analysis pass. Diagnostic RGB uses 0.5 + 0.5*x/(abs(x)+1), so it is signed
 * and bounded but is not a calibrated HDR visualization. Debug weight shows
 * the combined residual-retention weight (1 means no protection, 0 means
 * fully protected), including ROI, dark, and highlight factors. */
static constexpr char gpu_post_control_shader[] = R"hlsl(
ByteAddressBuffer Source : register(t0);
ByteAddressBuffer Candidate : register(t1);
ByteAddressBuffer Analysis : register(t2);
RWTexture2D<float4> Output : register(u0);
RWByteAddressBuffer Fallback : register(u1);
RWByteAddressBuffer AnalysisOut : register(u2);
cbuffer Options : register(b0) {
    uint StructSize, AbiVersion, Width, Height;
    uint PixelBytes;
    float TotalGain, LightnessGain, ChromaGain, LowFrequencyGain;
    float HighFrequencyGain, HueProtection, DarkProtection, HighlightProtection;
    float SoftCompression;
    uint RoiKind;
    float RoiX0, RoiY0, RoiX1, RoiY1;
    float RoiFeather;
    uint DebugMode;
    uint ColorMode;
};

static const float3 Y709 = float3(0.2126, 0.7152, 0.0722);
// Oklab transform from Bjorn Ottosson's published 2021 matrices (public domain).
// https://bottosson.github.io/posts/oklab/
// Extended signed transfer and cube roots keep wide masters unclipped.
float decode_srgb(float x) {
    float a=abs(x);return sign(x)*(a<=.04045 ? a/12.92 : pow((a+.055)/1.055,2.4));
}
float encode_srgb(float x) {
    float a=abs(x);return sign(x)*(a<=.0031308 ? a*12.92 : 1.055*pow(a,1./2.4)-.055);
}
float decode_bt709(float x) { return x<.081 ? x/4.5 : pow((x+.099)/1.099,1./.45); }
float encode_bt709(float x) { return x<.018 ? x*4.5 : 1.099*pow(x,.45)-.099; }
float3 to_oklab(float3 rgb) {
    // SDR perceptual controls operate on the displayed code colours, matching
    // the capture path's sRGB interpretation. The SDR10 master itself stays linear.
    float3 code=ColorMode==2u ? float3(encode_bt709(rgb.r),encode_bt709(rgb.g),encode_bt709(rgb.b)) : rgb;
    float3 linear_rgb=ColorMode==1u ? rgb : float3(decode_srgb(code.r),decode_srgb(code.g),decode_srgb(code.b));
    float3 lms=mul(float3x3(.4122214708,.5363325363,.0514459929,
        .2119034982,.6806995451,.1073969566,.0883024619,.2817188376,.6299787005),linear_rgb);
    lms=sign(lms)*pow(abs(lms),1./3.);
    return mul(float3x3(.2104542553,.7936177850,-.0040720468,
        1.9779984951,-2.4285922050,.4505937099,.0259040371,.7827717662,-.8086757660),lms);
}
float3 from_oklab(float3 lab) {
    float3 lms=mul(float3x3(1.,.3963377774,.2158037573,
        1.,-.1055613458,-.0638541728,1.,-.0894841775,-1.2914855480),lab);
    float3 linear_rgb=mul(float3x3(4.0767416621,-3.3077115913,.2309699292,
        -1.2684380046,2.6097574011,-.3413193965,-.0041960863,-.7034186147,1.7076147010),lms*lms*lms);
    if(ColorMode==1u)return linear_rgb;
    float3 code=float3(encode_srgb(linear_rgb.r),encode_srgb(linear_rgb.g),encode_srgb(linear_rgb.b));
    return ColorMode==2u ? float3(decode_bt709(code.r),decode_bt709(code.g),decode_bt709(code.b)) : code;
}
float3 mapped_oklab(float3 lab) {
    if(ColorMode==1u)return from_oklab(lab);
    lab.x=saturate(lab.x);
    float3 rgb=from_oklab(lab);
    if(all(rgb>=-1e-6)&&all(rgb<=1.000001))return rgb;
    float lo=0,hi=1;
    // Reduce chroma along a fixed hue ray rather than clipping RGB channels.
    [loop] for(uint i=0;i<16u;++i){
        float mid=(lo+hi)*.5;
        float3 trial=from_oklab(float3(lab.x,lab.yz*mid));
        if(all(trial>=0)&&all(trial<=1))lo=mid;else hi=mid;
    }
    return from_oklab(float3(lab.x,lab.yz*lo));
}
static const float3x3 To2020 = float3x3(
    0.6274040, 0.3292820, 0.0433136,
    0.0690970, 0.9195400, 0.0113612,
    0.0163916, 0.0880132, 0.8955950);

float4 read_pixel(ByteAddressBuffer plane, uint2 p) {
    uint address = p.y * ((Width * PixelBytes + 255u) & ~255u) + p.x * PixelBytes;
    if (PixelBytes == 4u) {
        uint word = plane.Load(address);
        return float4(word & 255u, (word >> 8) & 255u,
                      (word >> 16) & 255u, (word >> 24) & 255u) / 255.0;
    }
    uint2 words = plane.Load2(address);
    return float4(f16tof32(words.x & 0xffffu), f16tof32(words.x >> 16),
                  f16tof32(words.y & 0xffffu), f16tof32(words.y >> 16));
}
bool finite3(float3 v) { return all(isfinite(v)); }
bool finite4(float4 v) { return all(isfinite(v)); }
float3 safe_source(uint2 p) {
    float3 v = read_pixel(Source, p).rgb;
    return finite3(v) ? v : float3(0.0,0.0,0.0);
}
float3 safe_residual(uint2 p) {
    float4 s4 = read_pixel(Source, p);
    float4 c4 = read_pixel(Candidate, p);
    return finite4(s4) && finite4(c4) ? c4.rgb - s4.rgb : float3(0.0,0.0,0.0);
}
float3 raw_residual(uint2 p, out bool valid) {
    float4 s4 = read_pixel(Source, p);
    float4 c4 = read_pixel(Candidate, p);
    valid = finite4(s4) && finite4(c4);
    return valid ? c4.rgb - s4.rgb : float3(0.0,0.0,0.0);
}

/* Pass one reduces max(positive Rec.709 source component) and counts invalid
 * source pixels. The peak remains on GPU for pass two; only 16 diagnostic bytes
 * are copied back after both dispatches. */
[numthreads(8,8,1)]
void Analyze(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    float4 sample = read_pixel(Source, id.xy);
    float3 source = sample.rgb;
    if (!finite4(sample)) {
        uint ignored;
        AnalysisOut.InterlockedAdd(4, 1u, ignored);
        return;
    }
    float peak = max(max(source.r, source.g), max(source.b, 0.0));
    uint ignored;
    AnalysisOut.InterlockedMax(0, asuint(peak), ignored);
}

float3 low_residual(uint2 p, float3 center) {
    const float spatial[3] = {1.0, 2.0, 1.0};
    float3 sum = 0.0;
    float weight_sum = 0.0;
    [unroll] for (int oy=-1; oy<=1; ++oy) {
        [unroll] for (int ox=-1; ox<=1; ++ox) {
            int2 q = clamp(int2(p) + int2(ox,oy), int2(0,0),
                           int2(Width-1,Height-1));
            uint2 uq = uint2(q);
            float3 s = safe_source(uq);
            float3 r = safe_residual(uq);
            float3 diff = s - center;
            float w = spatial[ox+1] * spatial[oy+1] * (1.0/16.0) *
                exp(-dot(diff,diff) / (0.0004 + 0.04 * dot(center,center)));
            sum += w * r;
            weight_sum += w;
        }
    }
    return weight_sum > 1.0e-20 ? sum / weight_sum : 0.0;
}
float roi_weight(uint2 p) {
    if (RoiKind == 0u) return 1.0;
    float2 uv = (float2(p) + 0.5) / float2(Width,Height);
    float min_dim = float(min(Width,Height));
    float2 px = uv * float2(Width,Height);
    float2 lo = float2(RoiX0,RoiY0) * float2(Width,Height);
    float2 hi = float2(RoiX1,RoiY1) * float2(Width,Height);
    float dist_px = 0.0;
    if (RoiKind == 1u) {
        if (any(uv < float2(RoiX0,RoiY0)) || any(uv > float2(RoiX1,RoiY1)))
            return 1.0;
        dist_px = min(min(px.x-lo.x, hi.x-px.x), min(px.y-lo.y, hi.y-px.y));
    } else {
        float2 center = (lo+hi)*0.5;
        float2 radius = max((hi-lo)*0.5, 1.0e-6);
        float radial = length((px-center)/radius);
        if (radial >= 1.0) return 1.0;
        dist_px = min(radius.x,radius.y)*(1.0-radial);
    }
    float feather_px = RoiFeather * min_dim;
    float protect = feather_px <= 0.0 ? 1.0 :
                    smoothstep(0.0,feather_px,dist_px);
    return 1.0-protect;
}
float diagnostic(float v) { return 0.5 + 0.5 * v / (abs(v) + 1.0); }

[numthreads(8,8,1)]
void Apply(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    bool valid;
    float4 source4 = read_pixel(Source,id.xy);
    float4 candidate4 = read_pixel(Candidate,id.xy);
    float3 source = source4.rgb;
    float3 raw = raw_residual(id.xy,valid);
    bool source_valid = finite4(source4);
    if (!source_valid || !valid) {
        if (PixelBytes == 8u && source_valid) {
            Output[id.xy] = float4(source,source4.a);
            uint ignored;
            Fallback.InterlockedAdd(0,1u,ignored);
        } else {
            Output[id.xy] = float4(source,source4.a);
        }
        return;
    }
    if (TotalGain == 1.0 && LightnessGain == 1.0 && ChromaGain == 1.0 &&
        LowFrequencyGain == 1.0 && HighFrequencyGain == 1.0 &&
        HueProtection == 0.0 && DarkProtection == 0.0 &&
        HighlightProtection == 0.0 && SoftCompression == 0.0 &&
        RoiKind == 0u && DebugMode == 0u) {
        Output[id.xy] = candidate4;
        return;
    }
    float3 low = low_residual(id.xy,source);
    float3 controlled = LowFrequencyGain * low +
                        HighFrequencyGain * (raw-low);
    float3 source_lab = to_oklab(source);
    float3 lab_delta = to_oklab(source + TotalGain*controlled)-source_lab;
    lab_delta.x *= LightnessGain;
    lab_delta.yz *= ChromaGain;
    float source_chroma = length(source_lab.yz);
    if (HueProtection > 0.0 && source_chroma > 1.0e-12) {
        float2 axis = source_lab.yz/source_chroma;
        float2 target = source_lab.yz+lab_delta.yz;
        float2 protected_target = axis*max(dot(target,axis),0.0);
        lab_delta.yz = lerp(target,protected_target,
            HueProtection*smoothstep(0.0,0.02,source_chroma))-source_lab.yz;
    }
    // A luma/gray RGB offset changes perceived chroma in shadows. Oklab does not
    // substitute a gray RGB residual for a requested chroma-preserving correction.
    bool perceptual = LightnessGain != 1.0 || ChromaGain != 1.0 || HueProtection != 0.0;
    controlled = perceptual ? mapped_oklab(source_lab+lab_delta)-source : TotalGain*controlled;

    float peak = asfloat(Analysis.Load(0));
    float unit_floor = PixelBytes == 8u ? 0.0125 : 1.0e-4;
    float y = dot(source,Y709);
    float dark_threshold = PixelBytes == 8u ? 0.005 : 0.02;
    float dark_weight = lerp(1.0,
        saturate(abs(y)/(dark_threshold*max(peak,unit_floor))),DarkProtection);
    float peak_ref = max(peak,unit_floor);
    float near_peak = saturate((max(source.r,max(source.g,source.b))-
                                0.98*peak_ref)/(0.02*peak_ref));
    float highlight_weight = 1.0;
    if (dot(controlled,Y709) > 0.0)
        highlight_weight = 1.0 - HighlightProtection*near_peak;
    float protection_weight = roi_weight(id.xy)*dark_weight*highlight_weight;
    lab_delta *= protection_weight;
    controlled = perceptual ? mapped_oklab(source_lab+lab_delta)-source : controlled*protection_weight;

    if (PixelBytes == 8u && HighlightProtection > 0.0) {
        float3 source2020 = mul(To2020,source);
        float3 delta2020 = mul(To2020,controlled);
        float outward_scale = 1.0;
        [unroll] for (uint channel=0;channel<3;++channel) {
            if (source2020[channel] < 125.0 && delta2020[channel] > 0.0 &&
                source2020[channel]+delta2020[channel] > 125.0) {
                float allowed = max(0.0,125.0-source2020[channel]);
                outward_scale = min(outward_scale,allowed/delta2020[channel]);
            }
        }
        float factor=lerp(1.0,saturate(outward_scale),HighlightProtection);
        lab_delta*=factor;
        controlled = perceptual ? mapped_oklab(source_lab+lab_delta)-source : controlled*factor;
    }

    if (SoftCompression > 0.0) {
        float reference = max(length(source)+peak,unit_floor);
        float knee = 0.25*reference;
        float magnitude = length(controlled);
        if (magnitude > knee) {
            float compressed = knee + (magnitude-knee) /
                (1.0+(magnitude-knee)/reference);
            float factor=lerp(1.0,compressed/magnitude,SoftCompression);
            lab_delta*=factor;
            controlled = perceptual ? mapped_oklab(source_lab+lab_delta)-source : controlled*factor;
        }
    }
    if (DebugMode == 1u) {
        Output[id.xy] = float4(diagnostic(raw.r),diagnostic(raw.g),
                               diagnostic(raw.b),1.0);
    } else if (DebugMode == 2u) {
        Output[id.xy] = float4(diagnostic(controlled.r),diagnostic(controlled.g),
                               diagnostic(controlled.b),1.0);
    } else if (DebugMode == 3u) {
        Output[id.xy] = float4(protection_weight.xxx,1.0);
    } else {
        float3 result = source + controlled;
        if (PixelBytes == 8u && (!finite3(result) ||
            any(abs(result) > 65504.0))) {
            Output[id.xy] = float4(source,source4.a);
            uint ignored;
            Fallback.InterlockedAdd(0,1u,ignored);
        } else {
            Output[id.xy] = float4(result,candidate4.a);
        }
    }
}
)hlsl";

#endif
