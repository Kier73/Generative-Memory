cmake_minimum_required(VERSION 3.10)
project(GenerativeMemoryCore C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include_directories(include)

add_library(gmem SHARED 
    src/gmem.c 
    src/gmem_alloc.c
    src/gmem_filter_json.c
    src/gmem_filter_block.c
    src/gmem_vrns.c
    src/gmem_trinity.c
    src/gmem_aro.c
    src/gmem_manager.c
    src/gmem_net.c
    src/gmem_archetype.c
)

if(WIN32)
    target_link_libraries(gmem PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(gmem PRIVATE Threads::Threads)
    # Check for librt (for clock_gettime if needed, mostly implied in modern glibc)
endif()

# Enable AVX2 if available
# Enable AVX2 if available
if(MSVC)
    add_compile_options(/arch:AVX2)
    add_definitions(-DGMEM_USE_AVX2)
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
    add_definitions(-DGMEM_USE_AVX2)
endif()

# WinFsp / FUSE Integration (Windows Only)
if(WIN32)
    set(WINFSP_DIR "C:/Program Files (x86)/WinFsp")
    if(EXISTS "${WINFSP_DIR}/inc/fuse/fuse.h")
        message(STATUS "WinFsp SDK detected. Enabling gmem_mount.")
        include_directories("${WINFSP_DIR}/inc/fuse")
        link_directories("${WINFSP_DIR}/lib")
        
        add_executable(gmem_mount src/gmem_winfsp.c)
        target_link_libraries(gmem_mount gmem winfsp-x64)
    else()
        message(WARNING "WinFsp SDK not found at ${WINFSP_DIR}. gmem_mount will not be built. Please install the WinFsp SDK (Developer options) to enable OS integration.")
    endif()
endif()
