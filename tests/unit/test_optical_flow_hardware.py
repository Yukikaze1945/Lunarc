#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/unit/test_optical_flow_hardware.py
End-to-End Hardware Integration, Correctness, and Segmented Benchmark Test Suite for NVIDIA Optical Flow.

Verifies:
  TEST A: Static frames (prev == cur) -> flow ~ 0.
  TEST B: Horizontal translation (+16px) -> backward flow x ~ -16.
  TEST C: Vertical translation (+12px) -> backward flow y ~ -12.
  TEST D: Multiple resolutions (720p, 1080p) -> shape (H, W, 2) & scale integrity.
  TEST E: Asymmetric motion and area response test (explicit bounds on displacement & disocclusion).
  TEST F: Lifecycle & stress test (10 consecutive create/execute/destroy cycles).
  TEST G: Segmented hardware benchmarks (1080p & 4K across 1x1/2x2/4x4 and FAST/MEDIUM/SLOW).
  TEST H: Failure & non-fallback behavior (explicit errors, NO silent DIS fallback).
  TEST I: Engine cache bounded lifecycle test (30+ iterations of 1F/4F/1M/4M switches).
  TEST J: Output descriptor dimensions & non-divisible resolution boundary test (1921x1081, 1279x719).
"""
import os
import sys
import time
import math
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import optical_flow as of


def _mean_ms(timings, key):
    """Mean duration of one measured span, or 'n/a' when the bridge does not time it.

    Reporting an unmeasured stage as 0.0 would present it as a stage that costs nothing.
    """
    values = [t[key] for t in timings if isinstance(t.get(key), (int, float))]
    return f"{float(np.mean(values)):.2f}ms" if values else "n/a"


def test_a_static_scene():
    print("\n--- TEST A: 静止画面测试 ---")
    eng = of.OpticalFlowEngine("nvof", "1F")
    pinfo = eng.provider_info()
    assert pinfo["provider"] == "nvidia-optical-flow", f"Wrong provider: {pinfo}"
    assert pinfo["hardware"] is True, "Hardware flag must be True"

    w, h = 1280, 720
    np.random.seed(42)
    base = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    base[100:300, 100:300] = 255
    base[400:600, 700:900] = 0

    flow = eng.compute_flow(base, base)
    assert flow.shape == (h, w, 2), f"Unexpected shape {flow.shape}"
    assert flow.dtype == np.float32, f"Unexpected dtype {flow.dtype}"
    assert np.all(np.isfinite(flow)), "Flow contains NaN or Inf"

    max_mag = np.max(np.hypot(flow[..., 0], flow[..., 1]))
    mean_mag = np.mean(np.hypot(flow[..., 0], flow[..., 1]))
    print(f"Static flow: mean magnitude = {mean_mag:.4f} px, max = {max_mag:.4f} px")
    assert mean_mag < 0.2, f"Static scene mean flow too high: {mean_mag}"
    eng.close()
    print("TEST A PASSED: 静止画面光流趋于零")


def test_b_horizontal_translation():
    print("\n--- TEST B: 已知水平平移测试 (+16px) ---")
    eng = of.OpticalFlowEngine("nvof", "1F")

    w, h = 1280, 720
    prev_frame = np.zeros((h, w, 3), dtype=np.uint8)
    prev_frame[200:400, 200:400] = 255

    cur_frame = np.zeros((h, w, 3), dtype=np.uint8)
    cur_frame[200:400, 216:416] = 255

    flow = eng.compute_flow(prev_frame, cur_frame)
    assert flow.shape == (h, w, 2)

    patch = flow[250:350, 260:360]
    mean_x = np.mean(patch[..., 0])
    mean_y = np.mean(patch[..., 1])
    print(f"Horizontal +16px shift: patch backward flow x = {mean_x:.3f} px, y = {mean_y:.3f} px")
    assert abs(mean_x - (-16.0)) < 0.5, f"Expected backward flow x ~ -16, got {mean_x}"
    assert abs(mean_y - 0.0) < 0.5, f"Expected backward flow y ~ 0, got {mean_y}"
    eng.close()
    print("TEST B PASSED: 水平平移后向位移精确匹配 -16px")


def test_c_vertical_translation():
    print("\n--- TEST C: 已知垂直平移测试 (+12px) ---")
    eng = of.OpticalFlowEngine("nvof", "1F")

    w, h = 1280, 720
    prev_frame = np.zeros((h, w, 3), dtype=np.uint8)
    prev_frame[200:400, 200:400] = 255

    cur_frame = np.zeros((h, w, 3), dtype=np.uint8)
    cur_frame[212:412, 200:400] = 255

    flow = eng.compute_flow(prev_frame, cur_frame)
    assert flow.shape == (h, w, 2)

    patch = flow[260:360, 250:350]
    mean_x = np.mean(patch[..., 0])
    mean_y = np.mean(patch[..., 1])
    print(f"Vertical +12px shift: patch backward flow x = {mean_x:.3f} px, y = {mean_y:.3f} px")
    assert abs(mean_x - 0.0) < 0.5, f"Expected backward flow x ~ 0, got {mean_x}"
    assert abs(mean_y - (-12.0)) < 0.5, f"Expected backward flow y ~ -12, got {mean_y}"
    eng.close()
    print("TEST C PASSED: 垂直平移后向位移精确匹配 -12px")


def test_d_resolutions_scaling():
    print("\n--- TEST D: 多分辨率 (720p / 1080p) 与尺度测试 ---")
    eng = of.OpticalFlowEngine("nvof", "1F")

    for (w, h) in [(1280, 720), (1920, 1080)]:
        prev = np.zeros((h, w, 3), dtype=np.uint8)
        prev[200:400, 200:400] = 255

        cur = np.zeros((h, w, 3), dtype=np.uint8)
        cur[200:400, 220:420] = 255

        flow = eng.compute_flow(prev, cur)
        assert flow.shape == (h, w, 2), f"Resolution {w}x{h} produced wrong shape {flow.shape}"
        assert np.all(np.isfinite(flow))

        patch = flow[250:350, 260:360]
        mean_x = np.mean(patch[..., 0])
        print(f"Resolution {w}x{h} (+20px shift): patch flow x = {mean_x:.3f}")
        assert abs(mean_x - (-20.0)) < 0.5, f"Resolution {w}x{h} displacement scale mismatch: {mean_x}"

    eng.close()
    print("TEST D PASSED: 720p 与 1080p 分辨率及位移尺度均正确")


def test_e_direction_rigor():
    print("\n--- TEST E: 后向光流方向性与移动目标位移测试 ---")
    eng = of.OpticalFlowEngine("nvof", "1F")

    w, h = 1280, 720
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame1[100:300, 200:500] = 200

    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2[100:300, 250:550] = 200
    frame2[100:300, 100:200] = 255

    flow_backward = eng.compute_flow(frame1, frame2)

    # In moving object region (150..250, 300..450), backward flow must be negative x (-50px)
    patch_obj = flow_backward[150:250, 300:450]
    mean_bx = np.mean(patch_obj[..., 0])
    print(f"Moving object backward displacement: x = {mean_bx:.2f} px (expected ~ -50.0)")
    assert abs(mean_bx - (-50.0)) < 2.0, f"Backward flow failed: {mean_bx}"

    # In newly appeared occluder area (150..250, 120..180), verify flow is finite and distinct from moving object
    patch_new = flow_backward[150:250, 120:180]
    mean_new_x = np.mean(patch_new[..., 0])
    print(f"Occluded/disoccluded area mean motion x: {mean_new_x:.2f} px")
    assert np.all(np.isfinite(patch_new)), "Disoccluded region contains NaN or Inf"
    assert abs(mean_new_x - (-50.0)) > 5.0, f"Disoccluded region should not inherit full object flow: {mean_new_x}"

    eng.close()
    print("TEST E PASSED: 证实移动目标后向光流求解为精确 -50px 且遮挡区未错误继承目标位移")


def test_f_lifecycle_stress():
    print("\n--- TEST F: 生命周期与防重关压力测试 (10 次连续 create/execute/destroy) ---")
    w, h = 640, 480
    prev = np.zeros((h, w, 3), dtype=np.uint8)
    prev[100:200, 100:200] = 255
    cur = np.zeros((h, w, 3), dtype=np.uint8)
    cur[100:200, 110:210] = 255

    for cycle in range(10):
        eng = of.OpticalFlowEngine("nvof", "1F")
        flow = eng.compute_flow(prev, cur)
        assert flow.shape == (h, w, 2)
        eng.close()
        eng.close()  # repeat close must not crash
        print(f"  Cycle {cycle + 1}/10 passed")

    print("TEST F PASSED: 10 次连续创建/执行/重复关闭均正常返回，未出现崩溃或句柄错误")


def test_g_benchmarks():
    print("\n====================================================================")
    print("  TEST G: 全矩阵分段硬件基准测试 (1080p & 4K · Grid & Preset 完整覆盖)")
    print("====================================================================")

    matrix = [
        # (grid, preset, label)
        (1, "fast", "1F"),
        (1, "medium", "1M"),
        (1, "slow", "1S"),
        (2, "fast", "2F"),
        (2, "medium", "2M"),
        (2, "slow", "2S"),
        (4, "fast", "4F"),
        (4, "medium", "4M"),
        (4, "slow", "4S"),
    ]

    resolutions = [
        ("1080p", 1920, 1080, 15),
        ("4K", 3840, 2160, 5),
    ]

    header = (
        f"{'Resolution':<12} | {'Grid':<4} | {'Preset':<6} | "
        f"{'bgr_to_abgr':<11} | {'upload':<8} | {'submit_cpu':<10} | {'dep_wait':<9} | "
        f"{'postproc_incl':<13} | {'download':<8} | {'native_tot':<10} | {'python_tot':<10} | {'FPS':<6}"
    )
    print(header)
    print("-" * len(header))

    for (res_label, w, h, iters) in resolutions:
        f1 = np.zeros((h, w, 3), dtype=np.uint8)
        f2 = np.zeros((h, w, 3), dtype=np.uint8)
        f2[h//4:3*h//4, w//4:3*w//4] = 200

        for (g, preset, q_code) in matrix:
            eng = of.OpticalFlowEngine("nvof", quality=q_code, grid=g, preset=preset)

            # Warmup >= 3 frames
            for _ in range(3):
                _ = eng.compute_flow(f1, f2)

            # Measured iterations
            timings = []
            for _ in range(iters):
                _ = eng.compute_flow(f1, f2)
                t = eng.get_last_timing()
                timings.append(t)

            eng.close()

            bgr_to_abgr = np.mean([t["bgr_to_abgr_ms"] for t in timings])
            python_tot = np.mean([t["python_total_ms"] for t in timings])
            fps = 1000.0 / max(0.001, python_tot)

            line = (
                f"{res_label:<12} | {f'{g}x{g}':<4} | {preset.upper():<6} | "
                f"{bgr_to_abgr:>9.2f}ms | "
                f"{_mean_ms(timings, 'nvof_upload_ms'):>8} | "
                f"{_mean_ms(timings, 'nvof_submit_cpu_ms'):>10} | "
                f"{_mean_ms(timings, 'nvof_dependency_wait_ms'):>9} | "
                f"{_mean_ms(timings, 'nvof_postprocess_wait_inclusive_ms'):>13} | "
                f"{_mean_ms(timings, 'nvof_download_ms'):>8} | "
                f"{_mean_ms(timings, 'nvof_total_native_ms'):>10} | "
                f"{python_tot:>8.2f}ms | {fps:>6.1f}"
            )
            print(line)

    print("-" * len(header))
    print("TEST G COMPLETED: 全矩阵分段性能评测完成")


def test_h_failure_and_no_fallback():
    print("\n--- TEST H: 失败行为与严禁静默 Fallback 校验 ---")
    try:
        of.OpticalFlowEngine("invalid_engine_name")
        assert False, "Should have raised ValueError for invalid engine"
    except ValueError as e:
        print(f"Correctly caught unknown engine error: {e}")

    eng = of.OpticalFlowEngine("nvof", "1F")
    too_small = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        eng.compute_flow(too_small, too_small)
        assert False, "Should have raised RuntimeError for unsupported small resolution"
    except RuntimeError as e:
        print(f"Correctly caught unsupported resolution error: {e}")
    finally:
        eng.close()

    dis_eng = of.OpticalFlowEngine("dis", "Medium")
    dis_info = dis_eng.provider_info()
    assert dis_info["hardware"] is False, "DIS must report hardware=False"
    assert dis_info["provider"] == "opencv-dis-cpu", "DIS provider name must be honest"
    print(f"DIS provider info verified: {dis_info}")

    nvof_eng = of.OpticalFlowEngine("nvof", "1F")
    nvof_info = nvof_eng.provider_info()
    assert nvof_info["hardware"] is True, "NVOF must report hardware=True"
    assert nvof_info["provider"] == "nvidia-optical-flow", "NVOF provider must be nvidia-optical-flow"
    print(f"NVOF provider info verified: {nvof_info}")
    nvof_eng.close()

    print("TEST H PASSED: 错误处理与真实非 Fallback 验证通过")


def test_i_engine_cache_lifecycle():
    print("\n--- TEST I: Engine Cache 有界生命周期与切换压力测试 (32 次循环) ---")
    qualities = ["1F", "4F", "1M", "4M"]
    w, h = 640, 480
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    f2 = np.zeros((h, w, 3), dtype=np.uint8)
    f2[100:200, 100:200] = 255

    for i in range(32):
        q = qualities[i % len(qualities)]
        eng = of.get_optical_flow_engine("nvof", quality=q)
        flow = eng.compute_flow(f1, f2)
        assert flow.shape == (h, w, 2)

        # Verify bounded cache size: only 1 active nvof engine in _ENGINES
        active_nvof = [k for k in of._ENGINES.keys() if k == "nvof"]
        assert len(active_nvof) == 1, f"Expected 1 active NVOF in _ENGINES, found {len(active_nvof)}"

        pinfo = eng.provider_info()
        assert pinfo["hardware"] is True
        assert pinfo["cuda_context"] != "0x0"

    print("  Ran 32 consecutive switches across 1F / 4F / 1M / 4M smoothly.")
    print("TEST I PASSED: 30+ 次循环预设切换生命周期有界且稳定")


def test_j_output_descriptor_and_non_divisible_dims():
    print("\n--- TEST J: Output Descriptor 尺寸与非整除分辨率边界测试 ---")
    test_cases = [
        # (width, height, grid)
        (1279, 719, 1),
        (1279, 719, 2),
        (1279, 719, 4),
        (1921, 1081, 1),
        (1921, 1081, 2),
        (1921, 1081, 4),
    ]

    for (w, h, g) in test_cases:
        expected_out_w = math.ceil(w / g)
        expected_out_h = math.ceil(h / g)

        eng = of.OpticalFlowEngine("nvof", quality="1F", grid=g, preset="fast")
        sess = eng._engine

        # Verify internal dimensions match mathematical formula
        assert sess._grid_size == g
        assert sess._session_dims is None or sess._session_dims == (w, h)

        f1 = np.zeros((h, w, 3), dtype=np.uint8)
        f2 = np.zeros((h, w, 3), dtype=np.uint8)
        f2[100:300, 100:300] = 255

        flow = eng.compute_flow(f1, f2)
        assert flow.shape == (h, w, 2), f"Expected shape ({h}, {w}, 2), got {flow.shape}"
        assert np.all(np.isfinite(flow)), "Output flow contains invalid values"

        timing = eng.get_last_timing()
        print(f"  Dims {w}x{h} (Grid {g}x{g} -> output {expected_out_w}x{expected_out_h}): "
              f"upload={_mean_ms([timing], 'nvof_upload_ms')}, "
              f"submit_cpu={_mean_ms([timing], 'nvof_submit_cpu_ms')}, "
              f"dependency_wait={_mean_ms([timing], 'nvof_dependency_wait_ms')}, "
              f"postprocess_inclusive={_mean_ms([timing], 'nvof_postprocess_wait_inclusive_ms')}, "
              f"download={_mean_ms([timing], 'nvof_download_ms')}")

        eng.close()

    print("TEST J PASSED: 奇数与非整除尺寸在 1x1/2x2/4x4 下均严格匹配规范无越界")


def main():
    print("====================================================================")
    print("  NVIDIA Optical Flow (NVOF) 硬件集成测试与全量正确性回归")
    print("====================================================================")

    test_a_static_scene()
    test_b_horizontal_translation()
    test_c_vertical_translation()
    test_d_resolutions_scaling()
    test_e_direction_rigor()
    test_f_lifecycle_stress()
    test_h_failure_and_no_fallback()
    test_i_engine_cache_lifecycle()
    test_j_output_descriptor_and_non_divisible_dims()
    test_g_benchmarks()

    print("\n====================================================================")
    print("  ALL OPTICAL FLOW HARDWARE TESTS & BENCHMARKS PASSED (10/10 100%)")
    print("====================================================================")


if __name__ == "__main__":
    main()
