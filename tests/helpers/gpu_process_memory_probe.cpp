// Test-only DXGI query in the caller's process. No model or device ownership.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdint>

using Microsoft::WRL::ComPtr;
struct Memory {
    uint64_t luid, local_usage, local_budget, nonlocal_usage, nonlocal_budget;
};
extern "C" __declspec(dllexport) HRESULT gpu_process_memory_query(void* raw_device, Memory* out) {
    if (!out) return E_INVALIDARG;
    ComPtr<IDXGIFactory4> factory;
    auto hr=CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return hr;
    ComPtr<IDXGIAdapter3> adapter;
    if (raw_device) {
        const auto luid=static_cast<ID3D12Device*>(raw_device)->GetAdapterLuid();
        hr=factory->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter));
    } else {
        ComPtr<IDXGIAdapter1> first;
        hr=factory->EnumAdapters1(0,&first);
        if (SUCCEEDED(hr)) hr=first.As(&adapter);
    }
    if (FAILED(hr)) return hr;
    DXGI_ADAPTER_DESC1 desc{};
    hr=adapter->GetDesc1(&desc);if(FAILED(hr))return hr;
    DXGI_QUERY_VIDEO_MEMORY_INFO local{},nonlocal{};
    hr=adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&local);
    if(FAILED(hr))return hr;
    hr=adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&nonlocal);
    if(FAILED(hr))return hr;
    const uint64_t luid=(uint64_t(uint32_t(desc.AdapterLuid.HighPart))<<32)|desc.AdapterLuid.LowPart;
    *out={luid,local.CurrentUsage,local.Budget,nonlocal.CurrentUsage,nonlocal.Budget};
    return S_OK;
}
