"""Check the installed AI overlay with isolated Python and a real CUDA RAFT pair."""
import json
import os
from pathlib import Path
import sys
import time


resources, components, output = map(Path, sys.argv[1:4])
os.environ['LUNARC_COMPONENT_MODE'] = '1'
os.environ['LUNARC_COMPONENT_ROOT'] = str(components.resolve())
sys.path.insert(0, str(resources.resolve()))
import component_paths
component_paths.configure()
import torch
from torchvision.models.optical_flow import raft_large

assert sys.flags.isolated
assert Path(torch.__file__).resolve().is_relative_to((components / 'raft-v1/site').resolve())
checkpoint = components / 'runtime/torch_home/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth'
model = raft_large(weights=None)
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
model = model.eval().cuda()
previous = torch.full((1, 3, 128, 192), -1.0, device='cuda')
previous[:, :, 32:96, 32:96] = 1.0
current = torch.roll(previous, shifts=4, dims=-1)
torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    flow = model(previous, current, num_flow_updates=2)[-1]
torch.cuda.synchronize()
assert flow.shape == (1, 2, 128, 192)
assert torch.isfinite(flow).all().item()
result = dict(passed=True, isolated=True, python=sys.version,
              torch=torch.__version__, torch_path=torch.__file__,
              device=torch.cuda.get_device_name(), shape=list(flow.shape),
              iterations=2, cold_pair_ms=(time.perf_counter()-started)*1000,
              scope='installation compatibility; not flow accuracy or performance characterization')
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
