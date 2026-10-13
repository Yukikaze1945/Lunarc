// Compile the runtime HLSL when building, before publishing a replacement DLL.
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdio>
#include "gpu_post_control_shader.h"
int main() {
    const char* entries[] = {"Analyze", "Apply"};
    for (const char* entry : entries) {
        Microsoft::WRL::ComPtr<ID3DBlob> code, errors;
        HRESULT hr = D3DCompile(gpu_post_control_shader, sizeof(gpu_post_control_shader)-1,
            "gpu_post_control", nullptr, nullptr, entry, "cs_5_1",
            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (FAILED(hr)) {
            std::fprintf(stderr, "%s: %s\n", entry,
                errors ? static_cast<const char*>(errors->GetBufferPointer()) : "compilation failed");
            return 1;
        }
        std::printf("Compiled %s\n", entry);
    }
    return 0;
}
