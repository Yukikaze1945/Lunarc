#!/usr/bin/env python3
"""Compile and run a short native BT.709 SDR10 correctness probe.

Requires Windows, Visual Studio C++ tools, D3D12/CUDA on one adapter, and NumPy
is not required. It reuses only the existing HDR test's adapter/texture fixtures;
the probe itself uses a synthetic limited-range BT.709 P010 bar/ramp, native SDR
decode, zero-strength reconstruction, P010 encode, and explicit instrumented
readbacks. No video export or performance measurement is involved.
"""
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NATIVE = os.path.join(ROOT, "native")
OUT = os.path.join(ROOT, "bench_out", "sdr10_colour_gate")
SOURCE = os.path.join(OUT, "sdr10_colour_gate.cpp")
EXE = os.path.join(OUT, "sdr10_colour_gate.exe")
VCVARS = r"E:\Visual Studio\VC\Auxiliary\Build\vcvars64.bat"


CPP = r'''#define main hdr_legacy_main
#include <array>
#include "gpu_hdr_scrgb_bridge_test.cpp"
#undef main

namespace {
constexpr uint32_t W = 64, H = 32;
constexpr double Kr = 0.2126, Kb = 0.0722, Kg = 1.0 - Kr - Kb;

struct IdentityPlan {
    GpuHdrTapTable table{};
    std::vector<int32_t> vi, hi;
    std::vector<float> vw, hw;
    IdentityPlan(uint32_t w, uint32_t h) : vi(h), hi(w), vw(h, 1.0f), hw(w, 1.0f) {
        for (uint32_t i = 0; i < h; ++i) vi[i] = int32_t(i);
        for (uint32_t i = 0; i < w; ++i) hi[i] = int32_t(i);
        table.struct_size = sizeof(table); table.taps = 1;
        table.vertical_src_len = table.vertical_dst_len = h;
        table.vertical_indices = vi.data(); table.vertical_weights = vw.data();
        table.horizontal_src_len = table.horizontal_dst_len = w;
        table.horizontal_indices = hi.data(); table.horizontal_weights = hw.data();
    }
};

double oetf(double l) {
    return l < 0.018 ? 4.5 * l : 1.099 * std::pow(l, 0.45) - 0.099;
}
double eotf(double e) {
    return e < 0.081 ? e / 4.5 : std::pow((e + 0.099) / 1.099, 1.0 / 0.45);
}
double half_reference(double value) {
    return std::copysign(from_half(to_half(std::fabs(value))), value);
}
int code_y(double y) { return int(std::floor(std::clamp(y, 0.0, 1.0) * 876.0 + 64.5)); }
int code_c(double c) { return int(std::floor(std::clamp(c, -0.5, 0.5) * 896.0 + 512.5)); }
uint16_t p010(int code) { return uint16_t(code << 6); }

int sdr_gate() {
    constexpr std::array<std::array<double, 3>, 12> bars{{
        {{1,0,0}}, {{0,1,0}}, {{0,0,1}}, {{1,1,0}},
        {{0,1,1}}, {{1,0,1}}, {{0.01,0.01,0.01}}, {{0.05,0.05,0.05}},
        {{0.18,0.18,0.18}}, {{0.5,0.5,0.5}}, {{0.75,0.75,0.75}}, {{1,1,1}}
    }};
    IdentityPlan plan(W, H);
    Machine machine;
    if (!machine.open()) return 1;
    GpuHdrBridge* bridge = nullptr;
    check(gpu_sdr10_abi_version() == 1, "SDR10 ABI version is 1");
    GpuHdrContract contract{}; contract.struct_size = sizeof(contract);
    gpu_sdr10_contract(&contract);
    check(contract.pq_m1 == 0 && contract.pq_m2 == 0 && contract.pq_c1 == 0 &&
          contract.pq_c2 == 0 && contract.pq_c3 == 0 &&
          contract.scrgb_per_nit == 0 && contract.nits_per_scrgb == 0 &&
          std::fabs(contract.kr-Kr) < 1e-12 && std::fabs(contract.kb-Kb) < 1e-12,
          "SDR10 contract declares Rec.709 and no PQ/nits conversion");
    if (gpu_sdr10_open(machine.device.Get(), machine.context, W, H, W, H,
                       &plan.table, &plan.table, 1.0, &bridge) != 0 || !bridge) {
        note(std::string("gpu_sdr10_open: ") + last_error());
        machine.close(); return 1;
    }

    std::vector<uint16_t> y(size_t(W)*H), uv(size_t(W)*H/2u);
    std::vector<std::array<double,3>> decoded(size_t(W)*H);
    std::vector<int> expected_y(size_t(W)*H), expected_uv(size_t(W)*H/2u);
    for (uint32_t by=0; by<H/2; ++by) for (uint32_t bx=0; bx<W/2; ++bx) {
        const auto& rgb = bars[(bx + by) % bars.size()];
        const double rp=oetf(rgb[0]), gp=oetf(rgb[1]), bp=oetf(rgb[2]);
        const double yy=Kr*rp+Kg*gp+Kb*bp;
        const int yc=code_y(yy), uc=code_c((bp-yy)/(2*(1-Kb))),
                  vc=code_c((rp-yy)/(2*(1-Kr)));
        uv[size_t(by)*W + bx*2]=p010(uc); uv[size_t(by)*W + bx*2+1]=p010(vc);
        for(uint32_t dy=0;dy<2;++dy) for(uint32_t dx=0;dx<2;++dx) {
            const uint32_t yyrow=by*2+dy, xx=bx*2+dx;
            y[size_t(yyrow)*W+xx]=p010(yc);
            const double Yn=(yc-64.0)/876.0, Cb=(uc-512.0)/896.0,
                         Cr=(vc-512.0)/896.0;
            const double R=Yn+2*(1-Kr)*Cr, B=Yn+2*(1-Kb)*Cb,
                         G=(Yn-Kr*R-Kb*B)/Kg;
            decoded[size_t(yyrow)*W+xx] = {{
                half_reference(eotf(R)), half_reference(eotf(G)),
                half_reference(eotf(B))} };
            expected_y[size_t(yyrow)*W+xx] = code_y(
                Kr*oetf(decoded[size_t(yyrow)*W+xx][0])+
                Kg*oetf(decoded[size_t(yyrow)*W+xx][1])+
                Kb*oetf(decoded[size_t(yyrow)*W+xx][2]));
        }
        const double ec[3] = {oetf(decoded[size_t(by*2)*W+bx*2][0]),
            oetf(decoded[size_t(by*2)*W+bx*2][1]),
            oetf(decoded[size_t(by*2)*W+bx*2][2])};
        const double eluma=Kr*ec[0]+Kg*ec[1]+Kb*ec[2];
        expected_uv[size_t(by)*W+bx*2]=code_c((ec[2]-eluma)/(2*(1-Kb)));
        expected_uv[size_t(by)*W+bx*2+1]=code_c((ec[0]-eluma)/(2*(1-Kr)));
    }
    DeviceBuffer ydev, uvdev;
    check(ydev.alloc(machine.cuda, y.size()*sizeof(uint16_t)) &&
          uvdev.alloc(machine.cuda, uv.size()*sizeof(uint16_t)) &&
          ydev.upload(machine.cuda, y.data(), y.size()*sizeof(uint16_t)) &&
          uvdev.upload(machine.cuda, uv.data(), uv.size()*sizeof(uint16_t)),
          "synthetic P010 planes uploaded to CUDA");
    uint64_t master_ptr=0; uint32_t master_pitch=0;
    check(gpu_sdr10_decode(bridge, ydev.ptr, W*2, uvdev.ptr, W*2,
        1,1,1,1,2.0,&master_ptr,&master_pitch)==0,
        "strict BT.709 P010 decoded to relative linear FP16");
    expect_refusal(gpu_hdr_decode(bridge, ydev.ptr, W*2, uvdev.ptr, W*2,
        1,9,16,9,2.0,&master_ptr,&master_pitch),
        "HDR decode refuses the SDR10 handle");
    check(gpu_hdr_downscale(bridge,nullptr,nullptr)==0 &&
          gpu_hdr_proxy(bridge,nullptr,nullptr,nullptr,nullptr)==0,
          "identity working plane and independent guidance dispatch");

    std::vector<uint8_t> master_bytes(size_t(W)*H*8u);
    check(gpu_hdr_readback_scrgb(bridge,2,master_bytes.data(),W*8u)==0,
          "FP16 master correctness readback");
    double max_master_error=0;
    for(uint32_t yy=0;yy<H;++yy) for(uint32_t xx=0;xx<W;++xx) {
        for(uint32_t c=0;c<3;++c) {
            uint16_t half=0;
            std::memcpy(&half,master_bytes.data()+size_t(yy)*W*8u+xx*8u+c*2u,2);
            max_master_error=std::max(max_master_error,std::fabs(from_half(half)-
                decoded[size_t(yy)*W+xx][c]));
        }
    }
    check(max_master_error<=0.0015,"decoded FP16 values match the independent reference");

    std::vector<uint8_t> guide(size_t(W)*H*4u);
    check(gpu_hdr_readback_scrgb(bridge,5,guide.data(),W*4u)==0,
          "motion guidance correctness readback");
    int guide_max_error=0;
    bool guide_gray=true;
    for(uint32_t yy=0;yy<H;++yy) for(uint32_t xx=0;xx<W;++xx) {
        const size_t i=(size_t(yy)*W+xx)*4u;
        const auto& p=decoded[size_t(yy)*W+xx];
        const int ref=int(std::floor(std::clamp(oetf(Kr*p[0]+Kg*p[1]+Kb*p[2]),
            0.0,1.0)*255.0+0.5));
        guide_max_error=std::max(guide_max_error,std::abs(int(guide[i])-ref));
        guide_gray = guide_gray && guide[i]==guide[i+1] &&
            guide[i+1]==guide[i+2] && guide[i+3]==255;
    }
    check(guide_gray,"guidance is grayscale RGBA8");
    check(guide_max_error<=1,"guidance uses the BT.709 OETF within one 8-bit code");

    std::vector<float> neural_rgba(size_t(W)*H*4u,0.0f);
    for(size_t i=3;i<neural_rgba.size();i+=4) neural_rgba[i]=1.0f;
    NeuralTexture neural;
    check(neural.build(machine.device.Get(),W,H,neural_rgba),
          "zero neural texture is COPY_SOURCE behind a real fence");
    uint64_t composite=0; uint32_t composite_pitch=0;
    check(gpu_hdr_reconstruct(bridge,neural.texture.Get(),neural.fence.Get(),
        neural.fence_value,0.0,&composite,&composite_pitch)==0,
        "zero-strength composite preserves the decoded master");
    uint64_t outy=0,outuv=0; uint32_t outyp=0,outuvp=0;
    check(gpu_hdr_encode(bridge,1.0,&outy,&outyp,&outuv,&outuvp)==0,
          "relative-linear SDR master encoded to limited BT.709 P010");
    std::vector<uint8_t> actual_y(size_t(outyp)*H), actual_uv(size_t(outuvp)*H/2u);
    check(gpu_hdr_readback_encoded_p010(bridge,actual_y.data(),outyp,
        actual_uv.data(),outuvp)==0,"encoded P010 correctness readback");
    int yerr=0, uverr=0;
    for(uint32_t yy=0;yy<H;++yy) for(uint32_t xx=0;xx<W;++xx) {
        uint16_t code=0; std::memcpy(&code,actual_y.data()+size_t(yy)*outyp+xx*2u,2);
        yerr=std::max(yerr,std::abs(int(code>>6)-expected_y[size_t(yy)*W+xx]));
    }
    for(uint32_t yy=0;yy<H/2;++yy) for(uint32_t xx=0;xx<W;++xx) {
        uint16_t code=0; std::memcpy(&code,actual_uv.data()+size_t(yy)*outuvp+xx*2u,2);
        uverr=std::max(uverr,std::abs(int(code>>6)-expected_uv[size_t(yy)*W+xx]));
    }
    check(yerr<=2,"encoded P010 luma matches FP16-aware BT.709 reference within 2 codes");
    check(uverr<=2,"encoded P010 chroma matches encoded-RGB mean reference within 2 codes");
    GpuHdrTransferCounters counters{}; counters.struct_size=sizeof(counters);
    check(gpu_hdr_get_transfer_counters(bridge,&counters)==0,
          "transfer counters available after test");
    const uint64_t expected_readback=uint64_t(gpu_hdr_plane_pitch(bridge,2))*H+
        uint64_t(gpu_hdr_plane_pitch(bridge,5))*H+
        uint64_t(outyp)*H+uint64_t(outuvp)*(H/2u);
    check(counters.report_readback_bytes==expected_readback,
          "diagnostic image readback bytes are explicitly counted");
    std::printf("[gate] max master error %.8g; guidance error %d; P010 Y/UV "
        "max code error %d/%d; diagnostic image readback bytes %llu\n",
        max_master_error,guide_max_error,yerr,uverr,
        static_cast<unsigned long long>(counters.report_readback_bytes));
    gpu_hdr_close(bridge); ydev.free(machine.cuda); uvdev.free(machine.cuda);
    machine.close();
    return g_failures==0 ? 0 : 1;
}
} // namespace

int main() { setvbuf(stdout,nullptr,_IONBF,0); std::puts("[gate] start SDR10 color test"); return sdr_gate(); }
'''


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SOURCE, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(CPP)
    command = (
        f'call "{VCVARS}" && cl.exe /nologo /EHsc /std:c++17 /W4 /WX /utf-8 '
        f'/I"{NATIVE}" /I"{os.path.join(NATIVE, "include")}" '
        f'"{SOURCE}" "{os.path.join(NATIVE, "gpu_hdr_scrgb_bridge.lib")}" '
        f'd3d12.lib d3dcompiler.lib user32.lib /Fe:"{EXE}"'
    )
    built = subprocess.run(command, cwd=NATIVE, shell=True)
    if built.returncode:
        raise SystemExit(f"native SDR10 probe compile failed ({built.returncode})")
    environment = os.environ.copy()
    environment["PATH"] = NATIVE + os.pathsep + environment.get("PATH", "")
    run = subprocess.run([EXE], cwd=OUT, env=environment)
    raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
