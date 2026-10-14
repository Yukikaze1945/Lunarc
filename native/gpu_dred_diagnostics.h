#pragma once
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>

inline void enable_gpu_dred_diagnostics() {
    wchar_t flag[8]{};
    if (!GetEnvironmentVariableW(L"LUNARC_GPU_DIAGNOSTICS", flag, 8) || flag[0] != L'1') return;
    Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> settings;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(settings.GetAddressOf())))) {
        settings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        settings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        settings->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    }
}

inline void describe_gpu_removal(ID3D12Device* device, const char* operation,
                                  HRESULT code, char* text, size_t capacity) {
    const HRESULT reason = device->GetDeviceRemovedReason();
    std::snprintf(text, capacity, "%s: 0x%08lx (removed reason 0x%08lx)", operation,
                  static_cast<unsigned long>(code), static_cast<unsigned long>(reason));
    Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(dred.GetAddressOf())))) {
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 output{};
        if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&output))) {
            unsigned count = 0;
            for (auto* node = output.pHeadAutoBreadcrumbNode; node && count++ < 64; node = node->pNext) {
                if (!node->pLastBreadcrumbValue || !node->pCommandHistory) continue;
                const UINT done = *node->pLastBreadcrumbValue;
                if (done >= node->BreadcrumbCount) continue;
                const size_t used = std::strlen(text);
                if (used + 1 >= capacity) break;
                std::snprintf(text + used, capacity - used, "; DRED list=%s queue=%s done=%u/%u next_op=%u",
                    node->pCommandListDebugNameA ? node->pCommandListDebugNameA : "unnamed",
                    node->pCommandQueueDebugNameA ? node->pCommandQueueDebugNameA : "unnamed",
                    done, node->BreadcrumbCount, unsigned(node->pCommandHistory[done]));
            }
        }
    }
}
