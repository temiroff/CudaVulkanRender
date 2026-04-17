cmake_minimum_required(VERSION 3.25)
project(CudaPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090 = Ada Lovelace = sm_89

find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ?????? Optional: CUPTI Profiler (real per-unit hardware counters) ???????????????????????????????????????????????????????????????????
# Ships with CUDA Toolkit in extras/CUPTI.  Auto-detected from CUDAToolkit path.
# Adds "Real counters" checkbox to the GPU arch visualizer.
# Note: enabling profiling serialises GPU execution — expect ~5-15% throughput drop.
get_filename_component(_cuda_root "${CUDAToolkit_INCLUDE_DIRS}" DIRECTORY)
set(_cupti_root "${_cuda_root}/extras/CUPTI")
find_path(CUPTI_INCLUDE_DIR
    NAMES cupti_profiler_target.h
    HINTS "${_cupti_root}/include" "${CUDAToolkit_INCLUDE_DIRS}"
)
find_library(CUPTI_LIBRARY    NAMES cupti    HINTS "${_cupti_root}/lib64" "${_cupti_root}/lib")
find_library(NVPERF_HOST_LIB  NAMES nvperf_host  HINTS "${_cupti_root}/lib64" "${_cupti_root}/lib")
find_library(NVPERF_TARGET_LIB NAMES nvperf_target HINTS "${_cupti_root}/lib64" "${_cupti_root}/lib")

if(CUPTI_INCLUDE_DIR AND CUPTI_LIBRARY AND NVPERF_HOST_LIB)
    message(STATUS "CUPTI found: ${CUPTI_LIBRARY}")
    set(CUPTI_FOUND ON)
else()
    message(STATUS "CUPTI not found (GPU arch real-counters disabled)")
    set(CUPTI_FOUND OFF)
endif()

# ?????? Optional: OptiX SDK ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# Set OPTIX_SDK_DIR in cmake-gui or -DOPTIX_SDK_DIR=... on command line.
# Auto-detected from standard NVIDIA install locations if left empty.
set(OPTIX_SDK_DIR "" CACHE PATH "Path to OptiX SDK root (contains include/optix.h)")

if(NOT OPTIX_SDK_DIR OR NOT EXISTS "${OPTIX_SDK_DIR}/include/optix.h")
    foreach(_ver "9.1.0" "9.0.0" "8.1.0" "8.0.0" "7.7.0" "7.6.0")
        set(_candidate "C:/ProgramData/NVIDIA Corporation/OptiX SDK ${_ver}")
        if(EXISTS "${_candidate}/include/optix.h")
            set(OPTIX_SDK_DIR "${_candidate}" CACHE PATH
                "Path to OptiX SDK root (contains include/optix.h)" FORCE)
            break()
        endif()
    endforeach()
endif()

if(EXISTS "${OPTIX_SDK_DIR}/include/optix.h")
    message(STATUS "OptiX SDK found: ${OPTIX_SDK_DIR}")
    set(OPTIX_FOUND ON)
else()
    message(STATUS "OptiX SDK not found ??? set OPTIX_SDK_DIR in cmake-gui")
    set(OPTIX_FOUND OFF)
endif()

# ?????? Optional: Intel OIDN denoiser ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# Searches common install locations; override with -DOIDN_ROOT=...
# Houdini 20 ships OIDN 1.4 with headers + lib ??? auto-detected if installed.
set(OIDN_ROOT "" CACHE PATH "Path to OIDN SDK root (contains include/OpenImageDenoise/oidn.h)")

# Candidate search paths
set(_OIDN_HINTS
    "${OIDN_ROOT}"
    "C:/Program Files/Intel/OpenImageDenoise"
    "C:/Program Files/Side Effects Software/Houdini 20.0.590"
    "C:/Program Files/Side Effects Software/Houdini 19.5.493"
)
find_path(OIDN_INCLUDE_DIR
    NAMES OpenImageDenoise/oidn.h
    HINTS ${_OIDN_HINTS}
    PATH_SUFFIXES toolkit/include include
)
find_library(OIDN_LIBRARY
    NAMES OpenImageDenoise
    HINTS ${_OIDN_HINTS}
    PATH_SUFFIXES
        custom/houdini/dsolib  # Houdini layout
        lib lib64
)
find_file(OIDN_DLL
    NAMES OpenImageDenoise.dll
    HINTS ${_OIDN_HINTS}
    PATH_SUFFIXES bin
)

if(OIDN_INCLUDE_DIR AND OIDN_LIBRARY)
    message(STATUS "OIDN found: ${OIDN_LIBRARY}")
    set(OIDN_FOUND ON)
else()
    message(STATUS "OIDN not found ??? denoiser disabled (set OIDN_ROOT)")
    set(OIDN_FOUND OFF)
endif()

# ?????? Optional: DLSS via NVIDIA Streamline SDK ??????????????????????????????????????????????????????????????????????????????????????????????????????
# Download prebuilt SDK from: https://github.com/NVIDIA-RTX/Streamline/releases
# Set DLSS_SDK_DIR in cmake-gui or -DDLSS_SDK_DIR=... on command line.
# SDK root must contain include/sl.h and lib/x64/sl.interposer.lib.
set(DLSS_SDK_DIR "" CACHE PATH "Path to Streamline SDK root (contains include/sl.h)")
option(STREAMLINE_USE_DEVELOPMENT_DLLS "Use Streamline development DLLs (shows SDK watermark, no app/project ID required)" ON)

# Auto-detect from bundled external/streamline if not set
if(NOT DLSS_SDK_DIR OR NOT EXISTS "${DLSS_SDK_DIR}/include/sl.h")
    set(_sl_candidate "${CMAKE_SOURCE_DIR}/external/streamline")
    if(EXISTS "${_sl_candidate}/include/sl.h")
        set(DLSS_SDK_DIR "${_sl_candidate}" CACHE PATH
            "Path to Streamline SDK root (contains include/sl.h)" FORCE)
    endif()
endif()

if(EXISTS "${DLSS_SDK_DIR}/include/sl.h")
    message(STATUS "Streamline (DLSS) SDK found at ${DLSS_SDK_DIR}")
    set(DLSS_FOUND ON)
else()
    message(STATUS "Streamline SDK not found ??? DLSS disabled (set DLSS_SDK_DIR)")
    set(DLSS_FOUND OFF)
endif()

add_subdirectory(external/glfw)

# ?????? tinygltf (header-only; bundles stb_image + nlohmann/json) ????????????????????????????????????????????????
include(FetchContent)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.2
    GIT_SHALLOW    TRUE
)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL              OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY          ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# ?????? tinyexr (single-header EXR loader, no build system needed) ?????????????????????????????????????????????
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        v1.0.9
    GIT_SHALLOW    TRUE
)
FetchContent_Populate(tinyexr)

# ImGui sources (docking branch ??? required for split panels)
set(IMGUI_DIR external/imgui)
set(IMGUI_SOURCES_OPENGL3
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

# OpenUSD runtime/development root.
# Prefer locally built Vulkan-enabled USD when available.
set(_USD_ROOT_DEFAULT "${CMAKE_SOURCE_DIR}/external/usd_devkit")
if(EXISTS "${CMAKE_SOURCE_DIR}/external/usd_vulkan/include/pxr/pxr.h")
    set(_USD_ROOT_DEFAULT "${CMAKE_SOURCE_DIR}/external/usd_vulkan")
endif()
set(USD_ROOT "${_USD_ROOT_DEFAULT}" CACHE PATH "Path to USD install root (contains include/pxr and lib/usd_*.lib)")
set(USD_LIB_DIR "${USD_ROOT}/lib")
set(USD_RUNTIME_DIR "${USD_ROOT}/bin")
if(NOT EXISTS "${USD_RUNTIME_DIR}")
    set(USD_RUNTIME_DIR "${USD_LIB_DIR}")
endif()
set(USD_PLUGIN_DIR "${USD_ROOT}/plugin/usd")
set(USD_LIB_PLUGIN_DIR "${USD_ROOT}/lib/usd")

# Keep legacy devkit around for OpenEXR headers/libs used by hdri.cpp.
set(OPENEXR_FALLBACK_DIR "${CMAKE_SOURCE_DIR}/external/usd_devkit")

# miniz.c redefines WIN32_LEAN_AND_MEAN ??? suppress C4005 for that file only
set_source_files_properties(
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
    PROPERTIES COMPILE_OPTIONS "/wd4005"
)

add_executable(pathtracer
    src/main.cpp
    src/vulkan_context.cpp
    src/ui/gpu_arch_window.cpp
    src/ui/gpu_arch_sm_tracker.cu
    src/cuda_interop.cu
    src/pathtracer.cu
    src/bvh.cu
    src/gpu_textures.cu
    src/restir.cu
    src/rasterizer.cu
    src/hdri.cpp
    src/denoiser.cpp
    src/gltf_loader.cpp
    src/usd_loader.cpp
    src/urdf_loader.cpp
    src/mjcf_loader.cpp
    src/post_process.cpp
    src/nim_vlm.cpp
    src/nim_cosmos.cpp
    src/batch_processor.cpp
    src/custom_shader.cpp
    src/tinyexr_impl.cpp
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
    src/optix_denoiser.cpp
    src/optix_renderer.cpp
    src/dlss_upscaler.cpp
    src/slang_shader.cpp
    src/stb_write_impl.cpp
    src/cupti_profiler.cpp
    src/ui/viewport.cpp
    src/ui/control_panel.cpp
    src/ui/outliner.cpp
    src/ui/stats_panel.cpp
    src/ui/material_panel.cpp
    src/ui/anim_panel.cpp
    src/ui/articulation_panel.cpp
    src/ui/robot_demo_panel.cpp
    src/ui/hydra_preview.cpp
    ${IMGUI_SOURCES}
    ${IMGUI_SOURCES_OPENGL3}
)

target_include_directories(pathtracer PRIVATE
    src/
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CUDAToolkit_INCLUDE_DIRS}
    ${tinygltf_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
    ${USD_ROOT}/include
    ${OPENEXR_FALLBACK_DIR}/include
    ${OPENEXR_FALLBACK_DIR}/include/Imath
    ${USD_ROOT}/include/boost-1_81
    "C:/Program Files/Python311/include"   # required by Boost.Python in USD headers
)

target_link_directories(pathtracer PRIVATE
    ${USD_LIB_DIR}
    ${OPENEXR_FALLBACK_DIR}/lib
    "C:/Program Files/Python311/libs"
)

target_link_libraries(pathtracer PRIVATE
    glfw
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvml
    CUDA::nvrtc
    ole32 shell32 uuid winhttp comdlg32
    # OpenUSD ??? core libs needed for geometry + shading traversal
    usd_tf usd_arch usd_vt usd_gf usd_js usd_plug usd_trace
    usd_work usd_sdf usd_pcp usd_usd usd_ar
    usd_usdGeom usd_usdShade usd_usdUtils usd_usdLux
    # OpenEXR (from USD devkit) ??? supports all EXR compression types (DWAA, PIZ, B44, etc.)
    OpenEXR-3_1 OpenEXRCore-3_1 Iex-3_1 IlmThread-3_1 Imath-3_1
    zlibstatic
    # Hydra / HdStorm for USD animation preview window
    usd_hd usd_hdSt usd_hdx usd_hdGp usd_hdMtlx usd_hdsi usd_hdar
    usd_garch usd_glf usd_hgi usd_hgiGL usd_hgiVulkan usd_hgiInterop
    usd_usdImaging usd_usdImagingGL usd_usdSkelImaging usd_usdSkel
    opengl32
)

# Copy USD runtime DLLs next to the exe.
if(EXISTS "${USD_RUNTIME_DIR}")
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${USD_RUNTIME_DIR}"
            "$<TARGET_FILE_DIR:pathtracer>"
        COMMENT "Copying USD runtime DLLs"
    )
endif()

# Copy OpenUSD plugin descriptors + plugin DLLs/resources so Hydra can discover render delegates.
if(EXISTS "${USD_PLUGIN_DIR}")
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${USD_PLUGIN_DIR}"
            "$<TARGET_FILE_DIR:pathtracer>/usd"
        COMMENT "Copying OpenUSD plugins (plugin/usd)"
    )
endif()

# Some USD builds (including Vulkan-enabled source builds) place plugins under lib/usd.
if(EXISTS "${USD_LIB_PLUGIN_DIR}")
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${USD_LIB_PLUGIN_DIR}"
            "$<TARGET_FILE_DIR:pathtracer>/usd"
        COMMENT "Copying OpenUSD plugins (lib/usd)"
    )
endif()

# Copy full Blender color management folder next to the exe
add_custom_command(TARGET pathtracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/colormanagement"
        "$<TARGET_FILE_DIR:pathtracer>/colormanagement"
    COMMENT "Copying Blender color management (OCIO config + LUTs)"
)

# Enable device-side linking so __device__ functions can be called across .cu TUs
set_target_properties(pathtracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE_DIR_0)

target_compile_definitions(pathtracer PRIVATE
    NOMINMAX          # prevent Windows min/max macros clashing with std::numeric_limits
    WIN32_LEAN_AND_MEAN
    _SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING  # USD hd/dataSource.h uses deprecated std::atomic_load<shared_ptr>
    _CRT_SECURE_NO_WARNINGS  # sscanf etc. are safe in our usage
)

if(OPTIX_FOUND)
    target_compile_definitions(pathtracer PRIVATE
        OPTIX_ENABLED
        OPTIX_INCLUDE_DIR="${OPTIX_SDK_DIR}/include"
        CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR_0}")
    target_include_directories(pathtracer PRIVATE "${OPTIX_SDK_DIR}/include")
    message(STATUS "Building with OptiX SDK support (denoiser and RT renderer integrated)")
endif()

if(CUPTI_FOUND)
    target_compile_definitions(pathtracer PRIVATE CUPTI_ENABLED)
    target_include_directories(pathtracer PRIVATE "${CUPTI_INCLUDE_DIR}")
    target_link_libraries(pathtracer PRIVATE "${CUPTI_LIBRARY}" "${NVPERF_HOST_LIB}")
    if(NVPERF_TARGET_LIB)
        target_link_libraries(pathtracer PRIVATE "${NVPERF_TARGET_LIB}")
    endif()

    # Copy CUPTI + NvPerf runtime DLLs next to the exe.
    # cupti64_YYYY.X.Y.dll must be in the same directory as the executable.
    get_filename_component(_cupti_lib_dir "${CUPTI_LIBRARY}" DIRECTORY)
    file(GLOB _CUPTI_DLLS "${_cupti_lib_dir}/*.dll")
    if(_CUPTI_DLLS)
        add_custom_command(TARGET pathtracer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_CUPTI_DLLS}
                    "$<TARGET_FILE_DIR:pathtracer>"
            COMMENT "Copying CUPTI runtime DLLs (cupti64, nvperf_host, nvperf_target)")
    endif()

    message(STATUS "Building with CUPTI real hardware counters")
endif()

if(OIDN_FOUND)
    target_compile_definitions(pathtracer PRIVATE OIDN_ENABLED)
    target_include_directories(pathtracer PRIVATE "${OIDN_INCLUDE_DIR}")
    target_link_libraries(pathtracer PRIVATE "${OIDN_LIBRARY}")
    # Copy OIDN DLL next to exe if found
    if(OIDN_DLL)
        add_custom_command(TARGET pathtracer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OIDN_DLL}" "$<TARGET_FILE_DIR:pathtracer>"
            COMMENT "Copying OIDN DLL"
        )
    endif()
    message(STATUS "Building with OIDN denoiser support")
endif()

if(DLSS_FOUND)
    target_compile_definitions(pathtracer PRIVATE DLSS_ENABLED)
    target_include_directories(pathtracer PRIVATE "${DLSS_SDK_DIR}/include")
    # sl.interposer.dll exports all Vulkan symbols and intercepts vkCreateInstance/
    # vkCreateDevice to inject SL extensions.  It MUST be the first Vulkan provider
    # in the link order ??? so we removed Vulkan::Vulkan from the base link above and
    # add it HERE after sl.interposer so sl.interposer wins every Vulkan symbol.
    target_link_directories(pathtracer PRIVATE "${DLSS_SDK_DIR}/lib/x64")
    target_link_libraries(pathtracer PRIVATE sl.interposer Vulkan::Vulkan)
else()
    # No DLSS ??? use the plain Vulkan loader
    target_link_libraries(pathtracer PRIVATE Vulkan::Vulkan)
endif()

if(DLSS_FOUND)
    # Copy Streamline plugin DLLs next to the exe.
    # Default to development DLLs for local runs. Production DLLs are intended
    # for signed/onboarded titles and can report the feature as unsupported
    # without a valid NVIDIA application ID or project ID.
    set(_sl_dev_dir "${DLSS_SDK_DIR}/bin/x64/development")
    if(STREAMLINE_USE_DEVELOPMENT_DLLS AND EXISTS "${_sl_dev_dir}/sl.common.dll")
        file(GLOB SL_DLLS "${_sl_dev_dir}/*.dll")
        message(STATUS "Using Streamline development DLLs (no app-ID required)")
    else()
        file(GLOB SL_DLLS "${DLSS_SDK_DIR}/bin/x64/*.dll")
        message(STATUS "Using Streamline production DLLs")
    endif()
    if(SL_DLLS)
        add_custom_command(TARGET pathtracer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SL_DLLS}
                    "$<TARGET_FILE_DIR:pathtracer>"
            COMMENT "Copying Streamline plugin DLLs")
    endif()
    message(STATUS "Building with DLSS via Streamline SDK")
endif()

target_compile_options(pathtracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extended-lambda
        -Xcompiler /W3
        -Xcompiler /DNOMINMAX
    >
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -O0
    >
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:
        -O3
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
        # Suppress noisy but harmless warnings from third-party headers (USD devkit, Imath, tinyexr)
        /wd4244   # double???float/int conversion (USD/Imath)
        /wd4267   # size_t???int (USD ndr)
        /wd4251   # dll-interface (USD TfSingleton std::atomic)
        /wd4305   # double???float truncation (Imath half)
    >
)

# Compile GLSL shaders to SPIR-V
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
if(GLSLC)
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/fullscreen.vert
                -o $<TARGET_FILE_DIR:pathtracer>/vert.spv
        COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/fullscreen.frag
                -o $<TARGET_FILE_DIR:pathtracer>/frag.spv
        COMMENT "Compiling shaders"
    )
else()
    message(WARNING "glslc not found ??? shaders won't be compiled automatically")
endif()

# ?????? Optional: Slang shader compiler ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
set(SLANG_SDK_DIR "" CACHE PATH "Path to Slang SDK root (contains bin/slangc.exe)")
find_program(SLANGC slangc
    HINTS "${SLANG_SDK_DIR}/bin"
          "$ENV{SLANG_SDK_DIR}/bin"
          "$ENV{VULKAN_SDK}/Bin"          # Vulkan SDK bundles slangc since 1.3.x
          "C:/VulkanSDK/1.4.328.1/Bin"
          "C:/slang/bin"
    DOC "Path to slangc executable"
)
if(SLANGC)
    message(STATUS "slangc found: ${SLANGC}")
    # Expose path at runtime so the material shader editor can call slangc
    target_compile_definitions(pathtracer PRIVATE
        SLANGC_PATH="${SLANGC}"
    )
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${SLANGC}
            ${CMAKE_SOURCE_DIR}/shaders/post.slang
            -target spirv
            -stage compute
            -entry main
            -o $<TARGET_FILE_DIR:pathtracer>/post.spv
        COMMENT "Compiling post.slang -> post.spv"
    )
else()
    message(WARNING "slangc not found ??? post.slang won't be compiled (set SLANG_SDK_DIR)")
endif()

# ── GPU Architecture Viewer (standalone dev tool) ─────────────────────────────
# Fast-iteration target: no Vulkan, no OptiX, no path tracer.
# Run gpu_arch_viewer.exe to develop the arch UI independently.
set(IMGUI_OPENGL_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(gpu_arch_viewer
    tools/gpu_arch_main.cpp
    src/ui/gpu_arch_window.cpp
    ${IMGUI_OPENGL_SOURCES}
)

target_include_directories(gpu_arch_viewer PRIVATE
    src/
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gpu_arch_viewer PRIVATE
    glfw
    CUDA::cudart
    CUDA::nvml
    opengl32
)

target_compile_definitions(gpu_arch_viewer PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

target_compile_options(gpu_arch_viewer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /wd4244 /wd4267>
)
