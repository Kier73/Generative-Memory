# GVM: Generative Memory Core - Setup & Usage Guide

## Overview
GVM is an **Inductive Computational Substrate** that exposes up to **100TB of virtual memory** as a native OS drive. Unlike traditional storage, GVM synthesizes data on-demand using mathematical variety kernels, achieving multi-GB/s throughput without physical NAND overhead.

---

## 1. Prerequisites (Windows)

To build and run GVM, you need:
- **WinFsp**: The Windows File System Proxy (Runtime + SDK). [Download here](https://github.com/winfsp/winfsp/releases).
- **CMake 3.15+**: For build configuration.
- **C Compiler**: MSVC (Visual Studio) or MinGW-w64 (GCC).
- **Python 3.8+**: For running benchmarks and verification scripts.

---

## 2. Installation & Build

1. **Clone the project** and navigate to the `generative_memory` directory.
2. **Create a build directory**:
   ```powershell
   mkdir build
   cd build
   ```
3. **Configure and Build**:
   ```powershell
   cmake ..
   cmake --build . --config Release
   ```
   This will produce:
   - `gmem.dll`: The core synthesis engine.
   - `gmem_mount.exe`: The OS-level bridge service.

---

## 3. Mounting the Virtual Drive

To expose the 100TB substrate as a drive (e.g., `G:`), run the following command from the project root:

```powershell
# Add WinFsp to path if not already present
$env:PATH += ";C:\Program Files (x86)\WinFsp\bin"

# Launch the mount service
.\build\Release\gmem_mount.exe G: -f -o max_read=1048576 -o max_write=1048576 -o FileInfoTimeout=-1
```

### Mount Parameters:
- `G:`: The target drive letter.
- `-f`: Run in foreground (recommended for monitoring).
- `max_read=1048576`: Optimizes for 1MB read buffers (essential for > 2GB/s throughput).
- `FileInfoTimeout=-1`: Minimizes OS metadata requests for synthetic files.

---

## 4. Usage

Once mounted, a file named **`gmem_100tb.raw`** will appear on the `G:` drive.

### Key Workflows:
- **Massive Datasets**: Open the `.raw` file in any application (e.g., NumPy, Blender, Unreal Engine). It will act as a 100TB flat array of `float32`.
- **Inductive Synthesis**: Reading from the drive triggers on-demand AVX2-accelerated math variety generation.
- **Sparse Modification**: Any data you **WRITE** to the drive is stored in a high-speed RAM overlay. The mathematical base remains unchanged, but your modifications are persistent during the session.

### Performance Verification:
Run the comparative benchmark to see GVM vs. your physical hardware:
```powershell
python benchmarks/vfs_vs_nvme.py
```

---

## 5. Troubleshooting
- **Permission Denied**: Ensure you are running the terminal as an Administrator, or that WinFsp is correctly configured with user-access permissions (the current bridge handles this via `uid` mapping).
- **Lower than expected speed**: Check that `-o max_read=1048576` is used during mount and that your CPU supports AVX2.

---
> [!NOTE]
> This is a developmental release. Future updates will include persistent serialization and advanced manifold filtering.
