# RenoDX.DLSS5 addon — 静态逆向报告 / Static Reverse-Engineering Report

Subject: `F:\renodx-dlss5.addon64`
Date: 2026-09-29
Method: read-only static analysis (PE parsing, string/resource indexing, `.pdata`-guided disassembly, SPIR-V/DXBC container parsing). No decompiler was used.

---

## 0. 中文摘要（结论先行）

- 样本是 ReShade addon **RenoDX.DLSS5 v8.5-rc2**（PE32+ x86-64 DLL，含 Microsoft Detours），通过自身 `GetProcAddress` 解析 `NVSDK_NGX_D3D12_*` / `NVSDK_NGX_VULKAN_*` 后驱动 NVIDIA NGX；DLSS5 神经渲染走 runtime `nvngx_dlssnr.dll` 310.8.0（Feature 18）。
- **Feature 18 的参数面（`DLSSNR.*`）里没有任何 HDR / 色彩域 / 传递函数 / 基色 / 色空间 / 曝光的正式参数。** 完整键集合见 §5；`Exposure` 类键只有通用 DLSS-SR 时代的 `DLSS.Pre.Exposure`、`DLSS.Exposure.Scale`，且 addon 在 NR 参数块上把它们**硬写成 1.0f**（§6）。
- HDR 处理**不在 NGX 契约里，而在 addon 自己的计算着色器里**：`CodecConstants` / `FeedConstants` 带 `Encoding`、`TransferMode`、`HdrMode`、`PaperWhiteScale`、`DiffuseWhiteNits`、`Divisor`、`ChromaClampStops`、`HighlightGuardStops`、`GovernorMode` 等字段（§8）。这是一套**以“档（stops）”为单位的经验性归一化往返**，不是文档化的 scRGB/PQ 契约。
- NR 往返表面是 **`R16G16B16A16_FLOAT`（DXGI 10）**：`SDR proxy`、`raw neural output`、`look output`、`look transported proxy/output`（§7，直接来自 addon 自己的 `CreateResource` 调用点）。因此格式侧有 FP16 的 >1.0 余量，但喂给 NR 的**数值域被除数（divisor）压回近似 [0,1]**——与本仓库 FP16 探针“Feature 18 会把 >1.0 钳位”的实测互相印证。
- **不做任何基色/色域转换**：addon 自己的 UI 文案写明 “Advisory label only — the neural model works in wide gamut and no conversion is applied”（rva `0x1b9640`）。
- 运动引导只来自游戏的 `DLSSNR.MVec`（+ `MV.Scale.X/Y` 换算），**没有任何 NVOF / 光观光流使用**（§9）。
- **最终判定：在这份二进制里无法证明存在正式的 HDR 色彩/曝光契约（对 NGX 侧而言：不存在）。** 能证明的是一套 addon 私有的、按游戏类型（PQ HDR / float-HDR / relative HDR）分档的经验性亮度处理管线。详见 §11 与 §12。

---

## 1. Constraints honoured

| Constraint | Status |
|---|---|
| 不要执行或加载 addon | Honoured. The file was only read (`pefile`, `objdump`, `strings`, `mmap`). Never loaded, never injected, no process was started against it. |
| 不要联网提交样本 | Honoured. No network access; all SDK cross-references are from the **locally installed** `F:\dlss\DLSS-310.5.3\include\`. |
| 不要修改原文件 / 仓库代码 / 配置 | Honoured. Subject file untouched; all scratch tooling lives outside the repo in `F:\temp_renordx_re\`. This report is the only file added, and it is a new document (no existing file, code or config modified). |
| 不要 commit / push | Honoured. Nothing was committed or pushed. |

## 2. Tool capability limits (stated honestly)

Available and used: `python3 + pefile`, GNU `binutils` (`objdump`, `readelf`, `nm`, `strings`), a hand-written `.pdata` function index, and hand-written SPIR-V/DXBC container parsers (`F:\temp_renordx_re\*.py`).

Not available: **no decompiler** (no Ghidra / Hex-Rays / radare2), no `capstone`, no `spirv-dis`, no `dumpbin`. Therefore:

- Everything labelled FACT below is raw disassembly or raw bytes — reproducible with the commands in §13.
- There is **no decompiled pseudocode** in this report. Function-level *semantics* (what a struct field means) are INFERENCE from data flow, constant pool, and the addon's own embedded text.
- SPIR-V was parsed at the level of types/reflection strings/constant-block layouts. Per-instruction arithmetic of the transfer functions was **not** fully lifted into formulas (see §14, unresolved item U3). This is a capability limit, not a finding.

## 3. Sample provenance — FACT

| Property | Value |
|---|---|
| Path | `F:\renodx-dlss5.addon64` |
| Size | 3,141,632 bytes |
| SHA-256 | `dcd93881e976ad033d83c2bb01f4bc3e4ddc59c15fe0dd4ca165bc5fc7d1ac68` |
| MD5 | `3e758161148174312a30312b00841f6e` |
| Type | PE32+ (DLL), `x86-64`, image base `0x180000000`, entry RVA `0x1247e8` |
| Sections | `.text` 0x1000/0x15de6c, `.rdata` 0x15f000/0x183218, `.data` 0x2e3000, `.pdata` 0x2f0000, `.detourc`, `.detourd`, `.fptable`, `.rsrc`, `.reloc` |
| Debug dir | RSDS PDB `F:\renodx-worktrees\v85-rc2\build\Release\renodx-dlss5.pdb`, GUID `83fb9989-b547-4240-884b-20e1f4c21076`, age `0x1c` |
| Identity | ReShade addon; exports `NAME` → `"RenoDX.DLSS5"`, `DESCRIPTION`, `ReShadeRegisterAddon`, `ReShadeGetConfigValue`, `ReShadeGetImGuiFunctionTable`; INI section `[RenoDX.DLSS5]` |
| Build | RenoDX.DLSS5 **v8.5-rc2** (from the PDB path) |
| NGX target | DLSS5 Neural Rendering = **Feature 18**, runtime `nvngx_dlssnr.dll` **310.8.0** |
| Functions | 3,579 entries in `.pdata` (unwind info present ⇒ MSVC, no packer) |
| `.detourc`/`.detourd` | Microsoft Detours ⇒ the addon hooks NGX entry points in-process |

The presence of complete unwind data and a PDB path means the analysis surface is ordinary, unobfuscated compiler output — high confidence in per-instruction reading.

## 4. How the addon reaches NGX — FACT

Symbol resolution is done by the addon itself (label strings `"PrimeNgxLoaderSymbols/GetProcAddress"` rva `0x29aa50`, `"PrimeNgxLoaderSymbols/core-GetProcAddress"` rva `0x29ac30`):

```
0x1800bb798  call *0x18015f160        ; GetProcAddress (IAT)
0x1800bb79e  mov  %rax,0x1802e5780    ; NVSDK_NGX_D3D12_Init_Ext          (name @0x29aa78)
0x1800bb7af  mov  %rax,0x1802e5788    ; NVSDK_NGX_D3D12_AllocateParameters(@0x29aa98)
0x1800bb7cc  mov  %rax,0x1802e5790    ; NVSDK_NGX_D3D12_DestroyParameters  (@0x29aac0)
0x1800bb7e3  mov  %rax,0x1802e5798    ; NVSDK_NGX_D3D12_CreateFeature      (@0x29aae8)
0x1800bb7fa  mov  %rax,0x1802e57a0    ; NVSDK_NGX_D3D12_EvaluateFeature    (@0x29ab08)
0x1800bb811  mov  %rax,0x1802e57a8    ; NVSDK_NGX_D3D12_ReleaseFeature     (@0x29ab28)
0x1800bb828  mov  %rax,0x1802e57b0    ; NVSDK_NGX_D3D12_Shutdown1          (@0x29ab48)
```

A second, parallel set is resolved at `0x1800bbdc6 … 0x1800bbe22` into `0x1802e5848…0x1802e5870`. Both sets are packed into a loader object built at `0x1800d1890` (0x108 bytes, singleton stored at `0x1802eab80`) with layout `+0x10 Init_Ext, +0x18 AllocateParameters, +0x20 DestroyParameters, +0x28 CreateFeature, +0x30 EvaluateFeature, +0x38 ReleaseFeature, +0x40 Shutdown1`.

Vulkan entry points exist too (`NVSDK_NGX_VULKAN_Init_Ext`, `_CreateFeature1`, `_EvaluateFeature`, `_ReleaseFeature`, `_Shutdown1`, `_GetFeatureDeviceExtensionRequirements`), reached from `0x1800d5058…0x1800d50c4`, plus D3D11 (`NVSDK_NGX_D3D11_EvaluateFeature`, `_EvaluateFeature_C`).

Failure text `"DLSS-NR AllocateParameters failed"` at rva `0x29d2b8` (referenced at `0x180088bc1`) proves the NR path allocates a **real** parameter object through `NVSDK_NGX_D3D12_AllocateParameters`.

## 5. The complete Feature-18 (DLSSNR) parameter set — FACT

Cross-referencing every `lea <key string>,%rdx` immediately followed by `call *0xNN(%rax)` across all of `.text` yields the keys and **the vtable slot actually used** (162 correlated sites, 83 distinct NGX keys). Feature-18 block, filled by `0x1800f5d10`:

| RVA (call site) | Slot | Key | Value source |
|---|---|---|---|
| 0x1800f5d53 | 0x08 | `DLSSNR.Color` | arg5+0x08 (surface ptr) |
| 0x1800f5d67 | 0x08 | `DLSSNR.Output` | arg5+0x10 |
| 0x1800f5d83 | 0x08 | `DLSSNR.MVec` | arg6+0x08 |
| 0x1800f5d97 | 0x08 | `DLSSNR.Depth` | arg6+0x10 |
| 0x1800f5daf…0x1800f5f43 | 0x20 | `DLSSNR.ColorSubrectBaseX/BaseY/Width/Height`, `DepthSubrect*`, `MVecSubrect*`, `OutputSubrect*` | stack args 7–10 / arg6+0x38,+0x3c |
| 0x1800f5f5b…0x1800f5fa3 | 0x20 | `DLSSNR.InputWidth`, `InputHeight`, `OutputWidth`, `OutputHeight` | stack args 7–10 |
| 0x1800f5fd1 | 0x30 | `DLSSNR.Scale` | **1.0f** (see §6) |
| 0x1800f5ff8 | 0x18 | `DLSS.Feature.Create.Flags` | `game_flags \| 0x41` |
| 0x1800f600b | 0x30 | `DLSS.Pre.Exposure` | **1.0f** |
| 0x1800f601e | 0x30 | `DLSS.Exposure.Scale` | **1.0f** |
| 0x1800f609b / 0x1800f60b1 | 0x30 | `DLSSNR.JitterOffsetX` / `Y` | computed |
| 0x1800f61c1 | 0x18 | `DLSSNR.DepthInverted` | bool/uint |
| 0x1800f61d4 | 0x18 | `DLSSNR.Enabled` | uint |
| 0x1800f621e | 0x18 | `DLSSNR.Reset` | uint |

Other `DLSSNR.*` keys present in `.rdata` `0x29d200–0x29d790` and used elsewhere (independently correlated):
`DLSSNR.Width/Height`, `DLSSNR.Output.Width/Output.Height`, `DLSSNR.ScalingRatio` (float ×4 sites), `DLSSNR.Style` (int), `DLSSNR.UICorrection`, `DLSSNR.UseAutoMask` (uint), **`DLSSNR.LocalToneStrength`**, **`DLSSNR.LocalStructureStrength`**, **`DLSSNR.SkinStructureStrength`** (floats — NR's own tone/structure knobs), `DLSSNR.MVecScaleY`, `DLSSNR.Hint.Render.Preset` (int; the NR analogue of the documented `DLSS.Hint.Render.Preset`, `nvsdk_ngx_defs.h:806`).

**Negative result (FACT, exhaustive):** across the whole binary there is **no** parameter key containing `HDR`, `ColorSpace`/`ColourSpace`, `Transfer`, `Primaries`, `PQ`, `scRGB`, `EOTF`, `Nits`, `Gamut`, or any second `Exposure` name beyond `DLSS.Pre.Exposure` / `DLSS.Exposure.Scale` / `ExposureTexture`. Also **absent**: the documented frame-generation HDR flag `DLSSG.ColorBuffersHDR` (`nvsdk_ngx_defs_dlssg.h:127`) — the binary's full `DLSSG.*` set is `Backbuffer, Depth, DepthInverted, DepthSubrectBaseX/Y, HUDLess, MVecs, MVecsSubrect*, MultiFrameCount, MultiFrameIndex, MvecJittered, MvecScaleX, MvecScaleY, NotRenderingGameFrames, OutputInterpolated, OutputReal, Reset, UI, UIAlpha`. And no `Hint.UseFireflySwatter` / `SuperSampling.*` / legacy keys.

Feature-18 call chain: `0x1800beada → 0x1800f5d10` (D3D12 generic path) and `0x1800c24c6 → 0x1800f5d10`; both first invoke `call *0xd0(%rax)` on the workset object (slot 0xd0 is **not** one of the 17 documented `NVSDK_NGX_Parameter` slots) and pass `Width/Height/OutWidth/OutHeight` as stack args 7–10, plus `movl $0x8,0x3a0(%r14)`.

Create-flag site, verbatim:

```
0x1800f5fd4  cmpb  $0x0,0x394(%r14)      ; per-workset "HDR eligible" bool
0x1800f5fdc  je    0x1800f6021
0x1800f5fe0  mov   0x60(%rsp),%r8
0x1800f5ff0  mov   0x44(%r8),%r8d         ; game's create flags
0x1800f5ff4  or    $0x41,%r8d             ; IsHDR(1<<0) | AutoExposure(1<<6)
0x1800f5ff8  call  *0x18(%rax)            ; SetUI("DLSS.Feature.Create.Flags")
```
`1<<0 = NVSDK_NGX_DLSS_Feature_Flags_IsHDR` (`nvsdk_ngx_defs.h:284`), `1<<6 = AutoExposure`. This is **the only HDR signal the addon ever sends to NGX**, and it is a single create-time bit gated by an addon-side bool at workset `+0x394`.

## 6. Exposure parameters — exact names, set sites, lifecycle — FACT / INFERENCE

- **Set sites (NR block):** `DLSS.Pre.Exposure` @ `0x1800f600b`, `DLSS.Exposure.Scale` @ `0x1800f601e`, and `DLSSNR.Scale` @ `0x1800f5fd1` all load the *same* constant:
  ```
  0x1800f5fb0  movss 0x1802afb4c(%rip),%xmm6     ; .rdata f32 = 1.0
  0x1800f5fcb  movaps %xmm6,%xmm2                ; float argument register
  0x1800f6005  movaps %xmm6,%xmm2
  0x1800f6018  movaps %xmm6,%xmm2
  ```
  ⇒ **FACT**: on the Feature-18 parameter block these three keys are hard-wired to `1.0f`; no game exposure value is forwarded into NR there. (`movaps %xmm6,%xmm2` also proves the float argument arrives in `xmm2` at slot 0x30.)
- **Read sites (game's DLSS-SR block):** `0x1800d27a8` and `0x1800d27fa` call slot `0x70` with names `0x29b7e8`/`0x29b800`, i.e. the addon *reads* `DLSS.Pre.Exposure` / `DLSS.Exposure.Scale` out of the game's DLSS-SR parameter block, accepts the result only if it is not an `NVSDK_NGX_Result` `0xBAD000xx` code (`and $0xfff00000 / cmp $0xbad00000`), copies the float into its own record (`+0xcc`, `+0xd0`) and records a "present" flag.
- **Lifecycle:** read at evaluate-time from the game's block → used by the addon's own feed/codec shaders → a neutral `1.0f` is written to the NR block. INFERENCE (supported): the real exposure work is deliberately taken *out* of NGX and done in the addon's shaders.
- **The addon's own exposure parameters are shader uniforms, not NGX keys** — `FeedConstants` (§8) declares `Source`, `PreExposure`, `ExposureScale`, `SnapNow`, `Guard`, `FrameWidth`, `FrameHeight`.

## 7. HDR surfaces and pixel formats — FACT (strongest evidence in this audit)

Generic creator `0x18006bb50` builds a `D3D12_RESOURCE_DESC` at `%rbp+0x68`:

```
movl $0x3,0x68(%…)      ; Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D
movl $0x10001,0x84      ; DepthOrArraySize = 1, MipLevels = 1
mov  %r9d,0x88(%…)      ; Format   (DXGI_FORMAT)
movq $1,0x8c            ; SampleDesc.Count = 1
… call *0xd8(%rax)      ; CreateResource, debug name at 0x38(%rsp)
```
Caller table (call site → format register value → UTF-16 debug name → storage slot in the state object `%rbp`):

| Call site | DXGI format | Debug name | Storage |
|---|---|---|---|
| 0x18008af3a | game format | `DLSS5 Generic original DLSS output` | — |
| **0x18008af83** | **10 = R16G16B16A16_FLOAT** | `DLSS5 Generic SDR proxy` | `%rbp+0xe8` |
| 0x18008afd1 | game format | `DLSS5 Generic pre-SR NR color` | `%rbp+0x490` |
| **0x18008b014** | **10 = R16G16B16A16_FLOAT** | `DLSS5 Generic raw neural output` | `%rbp+0xf0` |
| 0x18008b067 | game format | `DLSS5 Generic decoded DLSS output` | `%rbp+0xf8` |
| 0x18008b0bf | game format | `…decoded DLSS output (stack ping-pong)` | — |
| 0x18008b11b | variable | `DLSS5 Generic NR block means` | — |
| 0x18008b5e6 / 0x8b62e / 0x8b67f | variable | `v6 linear source`, `v6 resolve ping-pong A/B` | — |
| 0x18008b6c2 / 0x8b70a | variable | `v6 frame scale` / `v6 committed frame scale` | — |
| **0x18008b7bb** | **10 RGBA16F** | `DLSS5 Generic look output` | — |
| 0x18008b808 / 0x8b84b | 2 = R32G32B32A32_FLOAT | `DLSS5 Generic look band A/B` | — |
| 0x18008b88e / 0x8b8d1 | 16 = R32G32_FLOAT | `DLSS5 Generic look band peak A/B` | — |
| **0x18008b926 / 0x8b971** | **10 RGBA16F** | `look transported proxy` / `look transported output` | — |
| 0x18008b9ce | 2 RGBA32F | `DLSS5 Generic look trace blocks` | — |
| 0x180067d10 | game format | `DLSS5 Generic pre-FG composite` | — |

Interpretation:
- FACT: the NR round-trip (input proxy → neural output → transported output) surfaces are **FP16 `RGBA16F`**; statistics/trace buffers are FP32; anything that must survive back into the game keeps the game's own format.
- FACT (naming): the buffer handed toward NR is called **“SDR proxy”**, and the addon's own log text says why — rva `0x29c5d0`: *“… without the IsHDR create flag: DLSS reads it as display-referred, NR as scene-linear; NRSourceEncoding=1 declares it display-referred SDR”*; rva `0x29c660`: *“float DLSS surface (format %d)”*.
- INFERENCE: the FP16 format is chosen for headroom in the addon's own round-trip, **not** to hand over-range values to NR. The addon's own tooltip at rva `0x1baf40` states the opposite consequence: *“a pre-exposed buffer fed as it is lets a bright scene overload NR's input, and NR boils.”* This corroborates this repository's independent FP16 probe (Feature 18 clamps >1.0), and it explains the divisor/proxy design in §8.
- Secondary classifier at `0x180093080` (re-verified): with `%ecx` = format and `%dl&1` = mode, it groups **{2, 10, 26}** (RGBA32F, RGBA16F, R11G11B10_FLOAT) against **24** (R10G10B10A2_UNORM) and everything else. This independently confirms the DXGI numbering used above; RGBA16F **is** in the float set (an earlier apparent contradiction was a mis-read of the `cmovb` default branch — the routine returns a 3-valued class, not a bool).

## 8. Where the HDR handling actually lives: the addon's own shaders — FACT (structure) + INFERENCE (semantics)

Shader inventory: **12 Slang-compiled SPIR-V 1.5 compute modules** (Vulkan path; container file offsets `0x180750, 0x183180, 0x185330, 0x188660, 0x18e280, 0x1909a0, 0x192740, 0x1945d0, 0x195dd0, 0x1979c0, 0x198be0, 0x19a020`) and **20 precompiled DXBC containers** (D3D12 path). Registration at `0x18008a460–0x18008a8d2`: nine `call 0x18006ae30` sites, each loading a 16-byte header immediately preceding a module, with names at `0x29bf48…0x29c230` (`v6 same-frame autoscale`, `v6 pedestal reduce, unshaped`, `v6 exposure scale`, `v6 commit, feed v2`, …), followed by the log lines *“created v6 NR codec from build-validated shader bytecode”* (rva `0x29c278`) and *“created the NR look stage from build-validated shader bytecode”* (rva `0x29c520`).

std430 constant blocks, read from the modules' own reflection strings (`SLANG_ParameterGroup_<Name>_std430`):

**`CodecConstants` — v6 codec (modules at 0x18e280, 0x1909a0, 0x192740, 0x1945d0, 0x1979c0, 0x198be0, 0x19a020):**
`SourceSize, SourceBase, ProxySize, NeuralSize,` **`Divisor, PedestalCap, TransferStrength, ColorStrength, Encoding, DarkGate, Curve, PedestalGate, SlewStops, SnapNow, ChromaClampStops, ReleaseStops, GovernorMode, TransferMode, HighlightGuardStops, DarkGateMode`**

**`CodecConstants` — older variant (modules 0x180750, 0x183180):**
`Size, SourceSize, SourceBase, ProxySize, NeuralSize,` **`PaperWhiteScale, TransferStrength, ColorStrength, HdrMode, DiffuseWhiteNits, Padding`**

**`FeedConstants` — feed stage (module 0x195dd0):**
`Source,` **`PreExposure, ExposureScale, SnapNow, Guard, FrameWidth, FrameHeight`**

**`LookConstants` — look stage (modules 0x185330, 0x188660):**
`Brighten, Darken, MaxBrighten, MaxDarken, ColourGain, HueGain, MaxColour, Shadows, Midtones, Highlights, Tone, Detail, Halo, Radius, Alpha, MotionScale, MotionBase, MotionGrid, GfEps, HaloBright, MinRadius, MidRadius, Padding` (images: `Proxy, Neural, BandA, BandMaxA, moments, prev`)

Image descriptors (decoded from `OpTypeImage`/`OpTypeSampledImage`/`OpDecorate` with `DescriptorSet=3`, `Binding=33`): in all 12 modules the colour inputs are **`sampled 2D`, element type `float`, with no `Format` operand** — i.e. the shader reads whatever format the *host view* provides; the only integer image is one `i32` image in module 3 (`BlockMean`). FACT: **the shaders themselves do not encode a storage format**; format selection is entirely the host code's (see §7).

Colour-domain vocabulary the addon actually uses (its own UI/log text, all in `.rdata`):

| RVA | Text (verbatim, abridged) |
|---|---|
| 0x1a8bf8 | `HDR mode` (UI label) |
| 0x1a9760 | “How bright white surfaces glow, in nits (**203 is the HDR reference**). **PQ HDR games only**; other HDR games use Scene white scale.” |
| 0x1a98e0 | “The same for **float-HDR and classic sources (2.5375 is 203-nit paper white)**. PQ HDR games use Paper white and PQ calibration instead.” |
| 0x1b7720 | “Where diffuse white sits in an HDR frame, in nits. 0 follows a loaded RenoDX HDR mod's game brightness, otherwise the SDR content brightness set in Windows.” |
| 0x1b8360 / 0x1b8d80 | `HDR transfer`; “How the game's HDR image is prepared for Neural Rendering and brought back. Auto is right for nearly every game.” |
| 0x1b9050 | “Used only for **calibrated absolute HDR**. **Relative HDR** uses the current frame's GPU-derived scale.” |
| 0x1b9640 | “**Advisory label only — the neural model works in wide gamut and no conversion is applied.**” |
| 0x1ba0a0 | chroma clamp **in stops**, ±2-stop luma bound |
| 0x1ba390 | “Bounded ratio vs Consistent inverse on the Display codec” |
| 0x1baf40 | feed v1 / v2 / Auto; “a pre-exposed buffer fed as it is lets a bright scene overload NR's input, and **NR boils**” |
| 0x1bb710…0x1bc350 | divisor governor (Stable/Slew/Off), attack/release, “0.020 is ~1.2 stops/s at 60 Hz” |
| 0x299040 | trace line `feed=%s exposure[min=%.6g max=%.6g] held=%u guard_v1=%u` |
| 0x299220/0x2992e0/0x299370 | `NR feed guard: engaged / cleared …` |
| 0x29d850 | style note: “measured ~0.44 st on a near-white field”, “Style Highlight Guard”, “Neural transfer = Consistent reshapes brightness edits on the shoulder” |

Reading (INFERENCE, high confidence because the strings, the shader field names and the surface formats agree): the Feature-18 pipeline is
`game HDR buffer → codec (encode/normalise: Divisor in stops, paper white, transfer mode, clamps) → FP16 "SDR proxy" → nvngx_dlssnr (Feature 18) → FP16 "raw neural output" → codec (inverse/"transported") → game format output`, with a frame-rate-aware governor slewing the divisor and a guard that engages when the feed would overload. That is an **empirical, self-calibrating round-trip**, expressed in stops and nits by the addon's own choice of units, and it is *not* an scRGB/PQ contract with NGX.

## 9. Motion guidance — FACT

- The only motion input to Feature 18 is `DLSSNR.MVec` (`Set` at `0x1800f5d83`, slot 0x08) together with `DLSSNR.MVecSubrect*` (ints) and `DLSSNR.MVecScaleY`; the game's velocity scale is read from the SR block keys `MV.Scale.X` / `MV.Scale.Y` (`0x18006d0fd`, `0x18006d12c`, slot 0x70) inside the snapshot reader `0x18006cbb0` (callers `0x1800617d5, 0x180061c1a, 0x18008e6c4, 0x18008f79d, 0x1800bc4b3, 0x1800c0a0c, 0x1800d8e58`), along with `Jitter.Offset.X/Y`, `PerfQualityValue`, `Reset`, `DLSS.Input.*.Subrect.Base.*`, `ExposureTexture`.
- **No NVOF / NVIDIA Optical Flow usage.** `nvo*`, `optical`, `flow` produce 29 string hits, all false positives (`0x19f738 "stack-overflow"`, and French/Russian localisation fragments at `0x1e9bb8, 0x1e9ee2, 0x1ea3b7, 0x1ec8be, 0x1ee0ac`). No `NVOFLOW`/`NvOF*` export, no `DLSSNR.OpticalFlow`-style key.
- Frame generation *is* handled (`DLSSG.*` keys, `nvngx_dlssg`, pre-FG composite surface), but again with **no HDR/colour key**.

## 10. Comparison with the two reference documents (reference only, not binary fact)

| Claim in `MAGPIE_HDR_SCRGB_REFERENCE.md` / `HDR_FP16_FEATURE18_AUDIT.md` | Binary evidence | Verdict |
|---|---|---|
| Feature 18 HDR is announced by `IsHDR` in the create flags | `or $0x41` at `0x1800f5ff4` ⇒ `IsHDR \| AutoExposure`, gated by workset `+0x394` | Consistent — the addon uses exactly that one bit and nothing else |
| There is a documented scRGB / PQ / primaries / colour-space parameter contract | Exhaustive key enumeration found **none** (`DLSSNR.*`, `DLSS.*`, `DLSSG.*`) | **Not supported for this binary**; no such contract is exercised |
| FP16 input/output with >1.0 headroom | NR round-trip surfaces are `RGBA16F` (§7) ⇒ container headroom exists | Format claim supported; **numeric** headroom is deliberately removed by the addon's normalisation ("SDR proxy", `0x29c5d0`, `0x1baf40`) |
| Feature 18 clamps values >1.0 (this repo's probe) | Addon's own text: over-exposed feed “overloads NR's input, and NR boils” (`0x1baf40`); its entire divisor/guard design exists to avoid that | Independent corroboration (text-level, not code-proved) |
| Exposure must be handled by the integrator | `DLSS.Pre.Exposure`/`DLSS.Exposure.Scale` forced to `1.0f` on the NR block; real exposure work in `FeedConstants.PreExposure/ExposureScale` | Supported, with mechanism relocated to the addon's shaders |

## 11. Verdict on the requested questions

1. **How does the addon handle Feature 18 HDR?** It does not negotiate HDR with NGX beyond setting the `IsHDR|0x40` create-flag bit. It performs its own encode→(NR)→decode round-trip in Slang shaders whose colour-domain parameters are `Encoding`, `TransferMode`/`HdrMode`, `PaperWhiteScale`/`DiffuseWhiteNits`, `Divisor`, `TransferStrength`, `ColorStrength`, `ChromaClampStops`, `HighlightGuardStops`, `DarkGate*`, `Pedestal*`, `GovernorMode`, `SlewStops` — all empirical, stop-based normalisation (§8). Grade: FACT for the existence/names of these knobs and the absence of NGX HDR keys; INFERENCE for the exact arithmetic (U3).
2. **Colour domain / transfer function / primaries?** The addon classifies the source as **PQ HDR / float-HDR / relative HDR / classic** and uses 203-nit BT.2408 paper white (2.5375 legacy factor) as its own reference. It performs **no primaries conversion** by its own explicit statement (`0x1b9640`). Grade: FACT (text) + INFERENCE (implementation).
3. **FP16 I/O and >1.0 headroom?** Container: yes, `R16G16B16A16_FLOAT`. Numeric range presented to NR: no — normalised to a display-referred "SDR proxy" scale (§7). Grade: FACT (format), FACT for the intent (own log text).
4. **`Exposure` / `PreExposure` / `HDR` / `ColorSpace` parameters?** Only `DLSS.Pre.Exposure` + `DLSS.Exposure.Scale` + `ExposureTexture` exist as NGX names (generic DLSS-SR names, §5/§6), and there is **no** `HDR` or `ColorSpace` NGX parameter anywhere. Grade: FACT (exhaustive enumeration).
5. **NVOF or other motion guidance?** Game `DLSSNR.MVec` only; zero NVOF (§9). Grade: FACT.
6. **Can a formal HDR colour/exposure contract be proven from this binary?** **No.** What can be proven is the opposite: the Feature-18 contract in use carries exactly one HDR-related bit, and every colour/exposure decision lives in the addon's private shader ABI. Any "HDR contract" one could write down from this binary would be a description of RenoDX's heuristics, not of an NGX-facing specification.

## 12. ABI anomaly worth flagging (relevant to the user's audit)

The slot numbers the addon uses for its parameter traffic are, from all evidence:

| Slot | Observed use | Documented slot meaning in `DLSS-310.5.3/nvsdk_ngx_params.h:53-74` |
|---|---|---|
| 0x00/0x08/0x10 | store pointer/64-bit (surfaces) | Set ULL / **Set float** / Set double |
| 0x18 | Set uint (bools, flags) | Set uint ✔ |
| 0x20 | Set int | Set int ✔ |
| 0x28 | Set (D3D11-ish) | Set ID3D11Resource ✔ |
| 0x30 | **Set float (value in `xmm2`)** | Set ID3D12Resource ✘ |
| 0x40/0x48/0x50 | Get pointer/64-bit | Get ULL / Get float / Get double |
| 0x58 | Get uint (game block) | Get uint ✔ |
| 0x60 | Get int (game block) | Get int ✔ |
| 0x70 | **Get float (game block: `Jitter.Offset.*`, `MV.Scale.*`, `DLSS.Pre.Exposure`)** | Get ID3D12Resource ✘ |
| 0x80 | Reset | Reset ✔ |

Two independent facts make this self-consistent: float is `xmm2`-passed at 0x30 / returned at 0x70 (`movaps %xmm6,%xmm2` at `0x1800f5fcb/0x1800f6005/0x1800f6018`), and the same numbering is used when reading the **game's** block, where results are validated against the `NVSDK_NGX_Result` `0xBAD000xx` family.

Interpretation A (preferred): the addon defines and uses a class of its own — its vtables sit in the addon's `.rdata` at `0x1a0b40` and `0x1a0bd0` (17 entries, `+0x00..+0x38` setters, `+0x40..+0x78` getters, `+0x80` Reset, preceded by an `AddRef`-shaped stub `incl 0x38(%rcx); ret` @ `0x1800dd530`), all `lea`-referenced by the addon's own constructors (`0x180009a58, 0x18003b55b, 0x18004228a`). Setter bodies write a tagged union into a map: `0x1800f5860` stores `%r8` at node `+0x48` with tag `+0x51`; `0x1800f5660` does `cvtss2sd %xmm6 → movsd +0x40`, tag `+0x50`; getter `0x180093c60` does `movsd +0x40 → cvtpd2ps → movss (%r14)`. Grade: FACT (the bodies), INFERENCE (that this is the object used in §5).

Interpretation B: the shipped 310.8 runtime's real parameter layout has the float and `ID3D12Resource` overloads transposed relative to the 310.5.3 public header.

Either way the audit conclusion is the same, and it is a *finding*: **RenoDX is coupled to implementation-level details of the NGX parameter object rather than to a stable documented HDR contract.** See also U1.

## 13. Reproduction

Scratch tooling (outside the repo): `F:\temp_renordx_re\` — `re_util.py` (PE layout, string index, rva↔offset on an `mmap` of the file), `funcs.py` (`.pdata` function index + objdump window printer; `show/disasm` take **RVAs**), `xref.py`, `svwalk.py`/`svimg.py`/`svlayout.py` (SPIR-V containers, image/decoration decoder, constant-block reflection), `dxbc.py`, `docs.py`, `strings_all.txt`, `text.asm`, `tooltips_en.txt`.

```
objdump -d -l --no-show-raw-insn renodx-dlss5.addon64 > text.asm      # .text, VA-based
python -c "import funcs as F; F.show(0xf5d10,0xf6220)"                # Feature-18 fill
python -c "import funcs as F; F.show(0x8af10,0x8b9e0)"                # surface allocation table
sha256sum /f/renodx-dlss5.addon64
```
All offsets quoted in this report are RVAs (image base `0x180000000`; file offset = RVA − section VA + section pointer, per §3 layout).

## 14. Unresolved items

| # | Item | Status / why |
|---|---|---|
| U1 | The map → real `NVSDK_NGX_Parameter` transfer site | Not located. A scan for functions issuing ≥4 distinct documented vtable slots found none, so there is no generic "replay my dictionary onto the NGX object" dispatcher visible in `.text`. The addon does resolve `NVSDK_NGX_D3D12_AllocateParameters` (`0x1800bb7af`) and its failure path is logged (`0x29d2b8`), so the transfer certainly happens; resolving it needs runtime tracing, which the brief forbids. |
| U2 | Which named surface is bound to `DLSSNR.Color` at each hook point | Partly resolved: the fill reads `workset+0x08`; the allocator stores `SDR proxy` at `state+0xe8` and `pre-SR NR color` at `state+0x490`, but the assignment `workset+0x08 ← which slot` was not proven through a 16 KB function without a data-flow engine. Naming + formats make the FP16-proxy reading high-confidence, not certain. |
| U3 | Per-instruction SPIR-V arithmetic of the encode/decode transfer functions | Not lifted into formulas. No `spirv-dis`/decompiler available; only block layouts, image descriptors and reflection names were decoded. The stop-based semantics in §8 come from field names and the addon's own text. |
| U4 | CPU-side write sites for `FeedConstants.PreExposure` / `ExposureScale` and `CodecConstants.Encoding/TransferMode/HdrMode` | Field existence is FACT; the constant-buffer fill sites were not reached. |
| U5 | Exact semantics of the 3-valued format classifier at `0x180093080` | The float set {2,10,26} vs {24} grouping is decoded (which confirms the DXGI numbering in §7), but the meaning of the third class and the `%dl&1` mode bit remain open. |
| U6 | DXBC (D3D12) side equivalence to the SPIR-V modules | Containers enumerated (20) and named, but not diffed against the 12 SPIR-V modules field-by-field. |

## 15. Evidence grades used

- **FACT** — directly readable bytes: hashes, section table, exported/resolved names, key strings, disassembled instructions, `D3D12_RESOURCE_DESC` field writes, constant-pool values, `.rdata` vtables, shader reflection strings.
- **INFERENCE** — structural reading supported by at least two independent kinds of evidence (e.g. surface format + addon's own log text + naming).
- **GUESS** — nothing in §11's conclusions rests on a guess; the only guess-grade thought is that RenoDX's author observed NR's >1.0 clamping empirically rather than from documentation (unprovable statically).
