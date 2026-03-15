# CUDA Path Tracer + Vulkan/ImGui — Project Quickstart

## What You're Building
A real-time path tracer with a professional ImGui UI, where CUDA renders directly into a
Vulkan texture (zero CPU copy). Two panels: left viewport showing live render, right panel
with camera/material/stats controls.

---

## Prerequisites

```bash
# Required
- CUDA Toolkit 12.x         https://developer.nvidia.com/cuda-downloads
- Vulkan SDK 1.3+           https://vulkan.lunarg.com/sdk/home
- CMake 3.25+
- Visual Studio 2022 (Windows) or GCC 13 (Linux)
- GLFW3
- Git

# Verify installs
nvcc --version
cmake --version
vulkaninfo | head -20
```

---

## Repo Setup

```bash
git init cuda-pathtracer
cd cuda-pathtracer

# Clone ImGui (docking branch — required for split panels)
git submodule add -b docking https://github.com/ocornut/imgui external/imgui

# Clone GLFW
git submodule add https://github.com/glfw/glfw external/glfw

git submodule update --init --recursive
```

---

## Project Structure to Create

```
cuda-pathtracer/
├── CMakeLists.txt
├── src/
│   ├── main.cpp              # Vulkan + ImGui init, main loop
│   ├── vulkan_context.cpp    # VkInstance, device, swapchain
│   ├── vulkan_context.h
│   ├── cuda_interop.cu       # CUDA-Vulkan shared texture
│   ├── cuda_interop.h
│   ├── pathtracer.cu         # Path trace kernel
│   ├── pathtracer.h
│   ├── bvh.cu                # BVH build (CPU) + traverse (GPU)
│   ├── bvh.h
│   ├── scene.h               # Sphere, Material, Camera structs
│   ├── ui/
│   │   ├── viewport.cpp      # ImGui Image() viewport panel
│   │   ├── viewport.h
│   │   ├── control_panel.cpp # Sliders, stats, BVH debug
│   │   └── control_panel.h
└── shaders/
    ├── fullscreen.vert.glsl  # Blit CUDA output to screen
    └── fullscreen.frag.glsl
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.25)
project(CudaPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090 = Ada = sm_89

find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(external/glfw)

# ImGui sources (Vulkan + GLFW backend)
set(IMGUI_DIR external/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

add_executable(pathtracer
    src/main.cpp
    src/vulkan_context.cpp
    src/cuda_interop.cu
    src/pathtracer.cu
    src/bvh.cu
    src/ui/viewport.cpp
    src/ui/control_panel.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(pathtracer PRIVATE
    src/
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(pathtracer PRIVATE
    Vulkan::Vulkan
    glfw
    CUDA::cudart
    CUDA::cuda_driver
)

# Compile shaders
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
add_custom_command(TARGET pathtracer POST_BUILD
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/fullscreen.vert.glsl
            -o ${CMAKE_BINARY_DIR}/vert.spv
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/fullscreen.frag.glsl
            -o ${CMAKE_BINARY_DIR}/frag.spv
)
```

---

## Build & Run

### Windows
```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
./Release/pathtracer.exe
```

### Linux / WSL2
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./pathtracer
```

---

## Key Implementation Notes

### CUDA-Vulkan Interop (the hard part)
```cpp
// 1. Create Vulkan image with external memory flag
VkExternalMemoryImageCreateInfo extInfo{};
extInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT; // Windows
//                  = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;    // Linux

// 2. Export the memory handle
HANDLE winHandle; // or int fd on Linux
vkGetMemoryWin32HandleKHR(device, &getInfo, &winHandle);

// 3. Import into CUDA
cudaExternalMemoryHandleDesc desc{};
desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
desc.handle.win32.handle = winHandle;
cudaImportExternalMemory(&extMem, &desc);

// 4. Map as CUDA surface — kernel writes float4 pixels directly
cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
cudaGetMipmappedArrayLevel(&cuArray, extMipmappedArray, 0);
cudaCreateSurfaceObject(&surface, &resDesc);
```

### ImGui Docking Setup
```cpp
ImGuiIO& io = ImGui::GetIO();
io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

// In render loop — full dockspace
ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

// Left panel: viewport
ImGui::Begin("Viewport");
ImGui::Image((ImTextureID)vulkanTextureDescriptorSet, viewportSize);
ImGui::End();

// Right panel: controls
ImGui::Begin("Controls");
ImGui::DragFloat3("Camera", &cam.origin.x, 0.01f);
ImGui::SliderInt("SPP", &spp, 1, 16);
ImGui::Text("%.1f Mrays/sec", mrays_per_sec);
ImGui::PlotLines("ms/frame", frame_times, 128);
ImGui::End();
```

### Path Trace Kernel Signature
```cuda
__global__ void pathtrace(
    cudaSurfaceObject_t surface,   // writes directly to Vulkan texture
    int width, int height,
    Camera cam,
    BVHNode* bvh, Sphere* prims,
    curandState* rng,
    int frame_count,               // for progressive accumulation
    int spp, int max_depth
)
```

---

## Progressive Accumulation Pattern
```cuda
// Blend new sample with history — reset when camera moves
float3 prev = read_accumulation_buffer(x, y);
float3 current = trace_path(ray, scene, rng);
float t = 1.0f / (frame_count + 1);
float3 result = lerp(prev, current, t);
surf2Dwrite(make_float4(result, 1.f), surface, x*16, y);
```

---

## What to Demo / Screenshot for Portfolio

1. **Split comparison** — rasterization vs path tracing same scene
2. **Progressive refinement** — frame 1 vs frame 256 vs frame 1024
3. **ImGui stats panel** — Mrays/sec counter, frame time graph live
4. **Nsight Compute screenshot** — SM occupancy, Tensor/RT Core utilization
5. **BVH debug view** — visualize AABB boxes, node count overlay

---

## Interview Talking Points

- CUDA-Vulkan interop via external memory — zero CPU roundtrip
- BVH traversal in software = understand what RT Cores offload in hardware
- Warp divergence in path tracing — rays hit different materials, diverge
- Progressive accumulation = temporal stability without TAA complexity
- ImGui immediate mode vs retained mode — why immediate mode fits real-time tools