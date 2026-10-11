// Display-only D3D12 readback. Export paths never load or call this module.
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <memory>
using Microsoft::WRL::ComPtr;
namespace {
thread_local char error[256] = "";
struct Reader {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> buffer;
    // Keep borrowed producer resources alive until our queue has finished.
    ComPtr<ID3D12Resource> submitted_source;
    ComPtr<ID3D12Fence> submitted_producer;
    bool poisoned = false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size = 0, serial = 0;
    UINT width = 0, height = 0, bytes = 0;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Reader() { if (event) CloseHandle(event); }
};
int fail(const char* what, HRESULT hr) {
    std::snprintf(error, sizeof(error), "%s: HRESULT 0x%08lx", what,
                  static_cast<unsigned long>(hr));
    return -1;
}
}
#define API extern "C" __declspec(dllexport)
API const char* gpu_preview_last_error() { return error; }
API void gpu_preview_close(Reader* reader) {
    if (!reader) return;
    // A failed signal gives no completion proof. Retain resources on a healthy
    // device rather than release a command allocator or texture still in use.
    if (reader->poisoned && SUCCEEDED(reader->device->GetDeviceRemovedReason())) return;
    delete reader;
}
API int gpu_preview_open(ID3D12Device* device, UINT width, UINT height,
                         UINT format, Reader** out) {
    if (!out) return fail("null output", E_INVALIDARG);
    *out = nullptr;
    if (!device || !width || !height || (format != 28 && format != 10))
        return fail("preview extent/format", E_INVALIDARG);
    auto r = std::make_unique<Reader>();
    r->device = device; r->width = width; r->height = height;
    r->bytes = format == 28 ? 4 : 8;
    if (!r->event) return fail("preview event", HRESULT_FROM_WIN32(GetLastError()));
    D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = device->CreateCommandQueue(&q, IID_PPV_ARGS(&r->queue));
    if (FAILED(hr)) return fail("preview queue", hr);
    hr = device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&r->allocator));
    if (FAILED(hr)) return fail("preview allocator", hr);
    hr = device->CreateCommandList(0, q.Type, r->allocator.Get(), nullptr,
                                   IID_PPV_ARGS(&r->list));
    if (FAILED(hr)) return fail("preview list", hr);
    hr = r->list->Close(); if (FAILED(hr)) return fail("close list", hr);
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&r->fence));
    if (FAILED(hr)) return fail("preview fence", hr);
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width; texture.Height = height;
    texture.DepthOrArraySize = 1; texture.MipLevels = 1;
    texture.Format = static_cast<DXGI_FORMAT>(format); texture.SampleDesc.Count = 1;
    device->GetCopyableFootprints(&texture, 0, 1, 0, &r->footprint, nullptr,
                                  nullptr, &r->size);
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; buffer.Width = r->size;
    buffer.Height = 1; buffer.DepthOrArraySize = 1; buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1; buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r->buffer));
    if (FAILED(hr)) return fail("preview readback", hr);
    *out = r.release(); return 0;
}
API int gpu_preview_read(Reader* r, ID3D12Resource* texture,
                         ID3D12Fence* producer, UINT64 producer_value,
                         uint8_t* host, size_t stride) {
    if (!r || !texture || !host || stride < size_t(r->width) * r->bytes)
        return fail("preview read arguments", E_INVALIDARG);
    if (r->poisoned) return fail("preview queue requires reopening", E_FAIL);
    const auto desc = texture->GetDesc();
    if (desc.Width != r->width || desc.Height != r->height ||
        desc.Format != r->footprint.Footprint.Format)
        return fail("preview source extent/format", E_INVALIDARG);
    ComPtr<ID3D12Device> owner;
    HRESULT hr = texture->GetDevice(IID_PPV_ARGS(&owner));
    if (FAILED(hr) || owner.Get() != r->device.Get())
        return fail("preview device mismatch", E_INVALIDARG);
    if (producer && producer_value) {
        hr = r->queue->Wait(producer, producer_value);
        if (FAILED(hr)) return fail("preview producer wait", hr);
    }
    hr = r->allocator->Reset(); if (FAILED(hr)) return fail("allocator reset", hr);
    hr = r->list->Reset(r->allocator.Get(), nullptr);
    if (FAILED(hr)) return fail("list reset", hr);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = texture; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = r->buffer.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = r->footprint;
    // All shared processing cores return COPY_SOURCE; no state guessing.
    r->list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    hr = r->list->Close(); if (FAILED(hr)) return fail("submit close", hr);
    ID3D12CommandList* lists[] = {r->list.Get()};
    r->submitted_source = texture;
    r->submitted_producer = producer;
    r->poisoned = true;
    r->queue->ExecuteCommandLists(1, lists);
    hr = r->queue->Signal(r->fence.Get(), ++r->serial);
    if (FAILED(hr)) return fail("preview signal", hr);
    hr = r->fence->SetEventOnCompletion(r->serial, r->event);
    if (FAILED(hr)) return fail("preview completion", hr);
    if (WaitForSingleObject(r->event, 30000) != WAIT_OBJECT_0)
        return fail("preview GPU timeout", HRESULT_FROM_WIN32(ERROR_TIMEOUT));
    r->poisoned = false;
    r->submitted_source.Reset();
    r->submitted_producer.Reset();
    void* mapped = nullptr; D3D12_RANGE range{0, size_t(r->size)};
    hr = r->buffer->Map(0, &range, &mapped);
    if (FAILED(hr)) return fail("preview map", hr);
    for (UINT row = 0; row < r->height; ++row)
        std::memcpy(host + row * stride,
            static_cast<uint8_t*>(mapped) + r->footprint.Offset +
                row * r->footprint.Footprint.RowPitch, size_t(r->width) * r->bytes);
    D3D12_RANGE none{0, 0}; r->buffer->Unmap(0, &none);
    return 0;
}
