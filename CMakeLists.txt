cmake_minimum_required(VERSION 3.25)
project(CudaPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090 = Ada Lovelace = sm_89

find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ── Optional: OptiX SDK ───────────────────────────────────────────────────────
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
    message(STATUS "OptiX SDK not found — set OPTIX_SDK_DIR in cmake-gui")
    set(OPTIX_FOUND OFF)
endif()

# ── Optional: Intel OIDN denoiser ────────────────────────────────────────────
# Searches common install locations; override with -DOIDN_ROOT=...
# Houdini 20 ships OIDN 1.4 with headers + lib — auto-detected if installed.
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
    message(STATUS "OIDN not found — denoiser disabled (set OIDN_ROOT)")
    set(OIDN_FOUND OFF)
endif()

# ── Optional: DLSS via NVIDIA Streamline SDK ──────────────────────────────────
# Download prebuilt SDK from: https://github.com/NVIDIA-RTX/Streamline/releases
# Set DLSS_SDK_DIR in cmake-gui or -DDLSS_SDK_DIR=... on command line.
# SDK root must contain include/sl.h and lib/x64/sl.interposer.lib.
set(DLSS_SDK_DIR "" CACHE PATH "Path to Streamline SDK root (contains include/sl.h)")

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
    message(STATUS "Streamline SDK not found — DLSS disabled (set DLSS_SDK_DIR)")
    set(DLSS_FOUND OFF)
endif()

add_subdirectory(external/glfw)

# ── tinygltf (header-only; bundles stb_image + nlohmann/json) ────────────────
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

# ── tinyexr (single-header EXR loader, no build system needed) ───────────────
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        v1.0.9
    GIT_SHALLOW    TRUE
)
FetchContent_Populate(tinyexr)

# ImGui sources (docking branch — required for split panels)
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

# ── OpenUSD devkit (extracted from MayaUSD 2025) ─────────────────────────────
set(USD_DEVKIT_DIR "${CMAKE_SOURCE_DIR}/external/usd_devkit")
set(MAYA_USD_LIB_DIR "C:/Program Files/Autodesk/MayaUSD/Maya2025/0.29.0/mayausd/USD/lib")

add_executable(pathtracer
    src/main.cpp
    src/vulkan_context.cpp
    src/cuda_interop.cu
    src/pathtracer.cu
    src/bvh.cu
    src/gpu_textures.cu
    src/restir.cu
    src/hdri.cpp
    src/denoiser.cpp
    src/gltf_loader.cpp
    src/usd_loader.cpp
    src/post_process.cpp
    src/nim_vlm.cpp
    src/batch_processor.cpp
    src/custom_shader.cpp
    src/tinyexr_impl.cpp
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
    src/optix_denoiser.cpp
    src/dlss_upscaler.cpp
    src/slang_shader.cpp
    src/stb_write_impl.cpp
    src/ui/viewport.cpp
    src/ui/control_panel.cpp
    src/ui/outliner.cpp
    src/ui/stats_panel.cpp
    src/ui/material_panel.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(pathtracer PRIVATE
    src/
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CUDAToolkit_INCLUDE_DIRS}
    ${tinygltf_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
    ${USD_DEVKIT_DIR}/include
    ${USD_DEVKIT_DIR}/include/Imath         # Imath headers (ImathVec, ImathBox, half…)
    ${USD_DEVKIT_DIR}/include/boost-1_81   # USD's bundled Boost headers
    "C:/Program Files/Python311/include"   # required by Boost.Python in USD headers
)

target_link_directories(pathtracer PRIVATE
    ${USD_DEVKIT_DIR}/lib
    "C:/Program Files/Python311/libs"
)

target_link_libraries(pathtracer PRIVATE
    Vulkan::Vulkan
    glfw
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvml
    CUDA::nvrtc
    ole32 shell32 uuid winhttp
    # OpenUSD — core libs needed for geometry + shading traversal
    usd_tf usd_arch usd_vt usd_gf usd_js usd_plug usd_trace
    usd_work usd_sdf usd_pcp usd_usd usd_ar
    usd_usdGeom usd_usdShade usd_usdUtils
    # OpenEXR (from USD devkit) — supports all EXR compression types (DWAA, PIZ, B44, etc.)
    OpenEXR-3_1 OpenEXRCore-3_1 Iex-3_1 IlmThread-3_1 Imath-3_1
    zlibstatic
)

# Copy MayaUSD runtime DLLs next to the exe so it can find them
add_custom_command(TARGET pathtracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${MAYA_USD_LIB_DIR}"
        "$<TARGET_FILE_DIR:pathtracer>"
    COMMENT "Copying MayaUSD runtime DLLs"
)

# Copy full Blender color management folder next to the exe
add_custom_command(TARGET pathtracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/colormanagement"
        "$<TARGET_FILE_DIR:pathtracer>/colormanagement"
    COMMENT "Copying Blender color management (OCIO config + LUTs)"
)

# Enable device-side linking so __device__ functions can be called across .cu TUs
set_target_properties(pathtracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(pathtracer PRIVATE
    NOMINMAX          # prevent Windows min/max macros clashing with std::numeric_limits
    WIN32_LEAN_AND_MEAN
)

if(OPTIX_FOUND)
    target_compile_definitions(pathtracer PRIVATE OPTIX_ENABLED)
    target_include_directories(pathtracer PRIVATE "${OPTIX_SDK_DIR}/include")
    message(STATUS "Building with OptiX RT core support")
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
    # Streamline interposer replaces vulkan-1.lib as the Vulkan dispatch layer.
    # Link sl.interposer.lib BEFORE Vulkan::Vulkan so it wins symbol resolution.
    target_link_directories(pathtracer PRIVATE "${DLSS_SDK_DIR}/lib/x64")
    target_link_libraries(pathtracer PRIVATE sl.interposer)
    # Copy all Streamline plugin DLLs next to the exe
    file(GLOB SL_DLLS "${DLSS_SDK_DIR}/bin/x64/*.dll")
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
        -O3
        --extended-lambda
        -Xcompiler /W3
        -Xcompiler /DNOMINMAX
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
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
    message(WARNING "glslc not found — shaders won't be compiled automatically")
endif()

# ── Optional: Slang shader compiler ─────────────────────────────────────────
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
    message(WARNING "slangc not found — post.slang won't be compiled (set SLANG_SDK_DIR)")
endif()
