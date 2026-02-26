use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// Phase 10: Semantic Autoencoder CUDA Gielis GPU Compiler.
/// Compiles PTX directly via NVRTC (Nvidia Runtime Compilation) in pure Rust,
/// executing the Lattice Lock Mass Geometry Search in bare-metal VRAM.
pub struct GielisGPUCompiler {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    active: bool,
}

impl GielisGPUCompiler {
    pub fn new() -> Result<Self, String> {
        // Initialize CUDA context for GPU 0
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(_) => {
                println!("  [WARN] CUDA Context could not be spun up or NVidia drivers missing. Gielis Lattice Lock disabled in Rust.");
                return Err("No CUDA Context".to_string());
            }
        };

        // The PTX source string natively embedded into the Rust binary at compile time
        let ptx_src = include_str!("../../../bindings-c/gielis_kernel.cu");

        // Compile to PTX using NVRTC via cudarc
        let ptx = compile_ptx(ptx_src).map_err(|e| format!("NVRTC Compile Error: {:?}", e))?;

        // Load into the context
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module Load Error: {:?}", e))?;

        Ok(Self {
            ctx,
            module,
            active: true,
        })
    }

    /// Executes the mass-parallel geometry search to find the 4 Morph Anchors.
    pub fn search(&self, target_signal: &[f32], freq: f32) -> Result<(f32, f32, f32, f32), String> {
        if !self.active {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }

        let l = target_signal.len();

        let mut phi_arr = vec![0.0f32; l];
        let freq_2pi = 2.0 * std::f32::consts::PI * freq;
        for i in 0..l {
            let t = -1.0 + 2.0 * (i as f32) / ((l.max(2) - 1) as f32);
            phi_arr[i] = freq_2pi * t;
        }

        // Search Grid (80,000 topographies)
        let mut m_flat = Vec::with_capacity(80_000);
        let mut n1_flat = Vec::with_capacity(80_000);
        let mut n2_flat = Vec::with_capacity(80_000);
        let mut n3_flat = Vec::with_capacity(80_000);

        let m_vals = (1..=10).map(|x| x as f32).collect::<Vec<_>>();
        let logspace_n = || {
            let start = -1.0f32;
            let end = 1.0f32;
            let step = (end - start) / 19.0;
            (0..20)
                .map(move |i| 10.0f32.powf(start + (i as f32) * step))
                .collect::<Vec<_>>()
        };

        let n1_vals = logspace_n();
        let n2_vals = logspace_n();
        let n3_vals = logspace_n();

        for &m in &m_vals {
            for &n1 in &n1_vals {
                for &n2 in &n2_vals {
                    for &n3 in &n3_vals {
                        m_flat.push(m);
                        n1_flat.push(n1);
                        n2_flat.push(n2);
                        n3_flat.push(n3);
                    }
                }
            }
        }

        let num_topologies = m_flat.len();
        let stream = self.ctx.default_stream();

        // 3. Move pointers to VRAM via the CudaStream
        let d_target = stream
            .clone_htod(target_signal)
            .map_err(|e| format!("{:?}", e))?;
        let d_phi = stream
            .clone_htod(&phi_arr)
            .map_err(|e| format!("{:?}", e))?;
        let d_m = stream.clone_htod(&m_flat).map_err(|e| format!("{:?}", e))?;
        let d_n1 = stream
            .clone_htod(&n1_flat)
            .map_err(|e| format!("{:?}", e))?;
        let d_n2 = stream
            .clone_htod(&n2_flat)
            .map_err(|e| format!("{:?}", e))?;
        let d_n3 = stream
            .clone_htod(&n3_flat)
            .map_err(|e| format!("{:?}", e))?;

        let mut d_mse_out = stream
            .alloc_zeros::<f32>(num_topologies)
            .map_err(|e| format!("{:?}", e))?;

        // 4. Launch configuration
        let threads_per_block = 256;
        let blocks_per_grid = (num_topologies as u32 + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (blocks_per_grid, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get function from pre-loaded module
        let f = self
            .module
            .load_function("gielis_lattice_lock")
            .map_err(|_| "Kernel not found".to_string())?;

        // 5. Fire SM compute threads simultaneously
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_target)
                .arg(&d_phi)
                .arg(&d_m)
                .arg(&d_n1)
                .arg(&d_n2)
                .arg(&d_n3)
                .arg(&mut d_mse_out)
                .arg(&(l as i32))
                .arg(&(num_topologies as i32))
                .launch(cfg)
        }
        .map_err(|e| format!("Launch failed: {:?}", e))?;

        // Copy back to CPU Memory DtoH
        let mse_cpu = stream
            .clone_dtoh(&d_mse_out)
            .map_err(|e| format!("{:?}", e))?;

        // 6. Find minimum MSE
        let mut best_idx = 0;
        let mut min_mse = f32::MAX;
        for (i, &mse) in mse_cpu.iter().enumerate() {
            if mse < min_mse {
                min_mse = mse;
                best_idx = i;
            }
        }

        Ok((
            m_flat[best_idx],
            n1_flat[best_idx],
            n2_flat[best_idx],
            n3_flat[best_idx],
        ))
    }
}
