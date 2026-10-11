"""Motion guidance is optional, while removed depth modes fail clearly."""
import os
import sys

import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from guidance_stubs import StubFlowService, make_core, source_frame
from dlss_service import DLSSService


def test_flow_mode_refuses_missing_guidance_service():
    with pytest.raises(ValueError, match="flow_provider"):
        make_core(guidance_mode=1)


@pytest.mark.parametrize("mode", [2, 3])
def test_removed_depth_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="only 0 .* or 1"):
        make_core(guidance_mode=mode)


def test_flow_free_mode_constructs_without_flow_service():
    core = make_core(guidance_mode=0)
    try:
        _final, record = core.process_frame(0, source_frame())
        assert record["guidance_backend"] != "stub_flow"
        assert not core.uses_flow
    finally:
        core.close()


def test_flow_service_is_used_once_per_frame():
    service = StubFlowService(value=2.0)
    core = make_core(guidance_mode=1, flow_service=service)
    try:
        for idx in range(3):
            core.process_frame(idx, source_frame())
        assert service.calls == 3
    finally:
        core.close()


def test_service_drops_removed_depth_setting_and_rejects_old_modes():
    service = DLSSService()
    settings = service.update_settings({"guidance_mode": 1, "depth_convention": 0})
    assert "depth_convention" not in settings
    job = service.add_job({"settings": {"guidance_mode": 1, "depth_convention": 2}})
    assert "depth_convention" not in job["settings"]
    with pytest.raises(ValueError, match="only 0 .* or 1"):
        service.add_job({"settings": {"guidance_mode": 3}})
