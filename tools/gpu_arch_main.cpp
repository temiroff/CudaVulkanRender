// GPU Architecture Viewer — standalone dev tool
// Build target: gpu_arch_viewer  (CMake)
// Uses GLFW + OpenGL3 backend: no Vulkan, no OptiX, no path tracer required.
// Fast iteration: just run gpu_arch_viewer.exe to develop the UI independently.

#ifndef NOMINMAX
#  define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/gl.h>        // basic GL 1.0 (glViewport, glClearColor, glClear)

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <cuda_runtime.h>
#include <nvml.h>

#include "ui/gpu_arch_window.h"

#include <cstdio>
#include <cstring>
#include <chrono>

static void glfw_error_cb(int err, const char* desc)
{
    fprintf(stderr, "GLFW error %d: %s\n", err, desc);
}

int main()
{
    // ── GLFW + OpenGL context ─────────────────────────────────────────────────
    glfwSetErrorCallback(glfw_error_cb);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* win = glfwCreateWindow(820, 640, "GPU Architecture Viewer", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);  // vsync

    // ── ImGui ─────────────────────────────────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ── CUDA device info (queried once) ───────────────────────────────────────
    char     hw_name[256] = "Unknown GPU";
    int      sm_count     = 0;
    int      cc_major     = 0, cc_minor = 0;
    int      l2_mb        = 0;
    uint64_t vram_total   = 0;

    {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                strncpy_s(hw_name, sizeof(hw_name), prop.name, _TRUNCATE);
                sm_count  = prop.multiProcessorCount;
                cc_major  = prop.major;
                cc_minor  = prop.minor;
                l2_mb     = prop.l2CacheSize / (1024 * 1024);
                vram_total = (uint64_t)prop.totalGlobalMem;
            }
        }
    }

    // ── NVML init ─────────────────────────────────────────────────────────────
    nvmlDevice_t nvml_dev = nullptr;
    bool         nvml_ok  = false;
    if (nvmlInit_v2() == NVML_SUCCESS)
        nvml_ok = (nvmlDeviceGetHandleByIndex(0, &nvml_dev) == NVML_SUCCESS);

    // ── Arch window state ─────────────────────────────────────────────────────
    GpuArchWindowState arch{};
    gpu_arch_window_init(arch, cc_major, cc_minor, l2_mb);
    arch.vram_total = vram_total;

    auto t_last = std::chrono::steady_clock::now();

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // Poll NVML every 500ms
        auto  t_now = std::chrono::steady_clock::now();
        float dt_ms = std::chrono::duration<float, std::milli>(t_now - t_last).count();
        if (dt_ms >= 500.f) {
            t_last = t_now;

            if (nvml_ok) {
                nvmlUtilization_t util{};
                if (nvmlDeviceGetUtilizationRates(nvml_dev, &util) == NVML_SUCCESS) {
                    arch.gpu_util_pct = util.gpu;
                    arch.mem_util_pct = util.memory;
                }
                nvmlMemory_t mem{};
                if (nvmlDeviceGetMemoryInfo(nvml_dev, &mem) == NVML_SUCCESS) {
                    arch.vram_used  = mem.used;
                    arch.vram_total = mem.total;
                }
            }

            // Approximate per-SM activity from overall GPU util:
            // No kernel running here, so we distribute active SMs proportionally.
            memset(arch.sm_active, 0, sizeof(arch.sm_active));
            if (sm_count > 0 && arch.gpu_util_pct > 0) {
                int n = (sm_count * (int)arch.gpu_util_pct + 99) / 100;
                for (int i = 0; i < n && i < sm_count; i++)
                    arch.sm_active[i >> 5] |= (1u << (i & 31));
            }
        }

        // ── ImGui frame ──────────────────────────────────────────────────────
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        bool open = true;
        gpu_arch_window_draw(arch, hw_name, sm_count, &open);
        if (!open) glfwSetWindowShouldClose(win, GLFW_TRUE);

        // ── Render ────────────────────────────────────────────────────────────
        ImGui::Render();

        int fb_w, fb_h;
        glfwGetFramebufferSize(win, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.11f, 0.11f, 0.13f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    if (nvml_ok) nvmlShutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
