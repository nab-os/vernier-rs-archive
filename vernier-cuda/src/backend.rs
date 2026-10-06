use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, ComputeJob, Real, Result, VernierError};

use crate::buffer::CudaBuffer;
use crate::kernels::KERNEL_SRC;

// ---------------------------------------------------------------------------
// cuFFT raw FFI
// ---------------------------------------------------------------------------

type CufftHandle = u32;
const CUFFT_C2C: i32 = 0x29; // from cufft.h: CUFFT_C2C (complex-to-complex interleaved)
const CUFFT_FORWARD: i32 = -1;
const CUFFT_INVERSE: i32 = 1;

unsafe extern "C" {
    fn cufftCreate(plan: *mut CufftHandle) -> i32;
    fn cufftMakePlan2d(plan: CufftHandle, nx: i32, ny: i32, ctype: i32, work_size: *mut usize) -> i32;
    fn cufftExecC2C(plan: CufftHandle, idata: *mut f32, odata: *mut f32, direction: i32) -> i32;
    fn cufftDestroy(plan: CufftHandle) -> i32;
}

// ---------------------------------------------------------------------------
// CudaContext: holds the device, all compiled functions, and the FFT plan cache
// ---------------------------------------------------------------------------

pub struct CudaContext {
    pub dev: Arc<CudaDevice>,
    scale_fn: CudaFunction,
    bandpass_fn: CudaFunction,
    bandpass_from_peaks_fn: CudaFunction,
    filter_annulus_fn: CudaFunction,
    extract_phase_fn: CudaFunction,
    magnitude_fn: CudaFunction,
    blur_h_fn: CudaFunction,
    blur_v_fn: CudaFunction,
    argmax_local_fn: CudaFunction,
    argmax_global_fn: CudaFunction,
    band_angular_filter_fn: CudaFunction,
    peak_order_fn: CudaFunction,
    spectral_partial_fn: CudaFunction,
    spectral_global_fn: CudaFunction,
    /// key = (height, width), value = CufftHandle
    fft_plans: Mutex<HashMap<(usize, usize), CufftHandle>>,
}

// CudaContext wraps raw pointers (device handles); we assert Send + Sync
// because cudarc's device is internally synchronized.
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new() -> Result<Arc<Self>> {
        let dev = CudaDevice::new(0).map_err(|e| VernierError::Backend(e.to_string()))?;

        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| VernierError::Backend(e.to_string()))?;

        let kernel_names = &[
            "scale",
            "bandpass",
            "bandpass_from_peaks",
            "filter_annulus",
            "extract_phase",
            "magnitude_inplace",
            "gaussian_blur_h",
            "gaussian_blur_v",
            "argmax_local",
            "argmax_global",
            "band_angular_filter",
            "peak_order",
            "spectral_plane_fit_partial",
            "spectral_plane_fit_global",
        ];
        dev.load_ptx(ptx, "vernier", kernel_names)
            .map_err(|e| VernierError::Backend(e.to_string()))?;

        let get = |name: &str| -> Result<CudaFunction> {
            dev.get_func("vernier", name)
                .ok_or_else(|| VernierError::Backend(format!("kernel not found: {}", name)))
        };

        Ok(Arc::new(Self {
            scale_fn: get("scale")?,
            bandpass_fn: get("bandpass")?,
            bandpass_from_peaks_fn: get("bandpass_from_peaks")?,
            filter_annulus_fn: get("filter_annulus")?,
            extract_phase_fn: get("extract_phase")?,
            magnitude_fn: get("magnitude_inplace")?,
            blur_h_fn: get("gaussian_blur_h")?,
            blur_v_fn: get("gaussian_blur_v")?,
            argmax_local_fn: get("argmax_local")?,
            argmax_global_fn: get("argmax_global")?,
            band_angular_filter_fn: get("band_angular_filter")?,
            peak_order_fn: get("peak_order")?,
            spectral_partial_fn: get("spectral_plane_fit_partial")?,
            spectral_global_fn: get("spectral_plane_fit_global")?,
            dev,
            fft_plans: Mutex::new(HashMap::new()),
        }))
    }
}

// ---------------------------------------------------------------------------
// CudaBackend
// ---------------------------------------------------------------------------

pub struct CudaBackend {
    ctx: Arc<CudaContext>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        Ok(Self { ctx: CudaContext::new()? })
    }
}

impl ComputeBackend for CudaBackend {
    type Buffer2D = CudaBuffer;
    type Job<'a> = CudaJob<'a>;

    fn begin(&self) -> Result<CudaJob<'_>> {
        Ok(CudaJob { ctx: &self.ctx })
    }

    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<CudaBuffer> {
        layout.check_upload(data.len())?;
        let floats: &[f32] = bytemuck::cast_slice(data);
        let dev_slice = self.ctx.dev.htod_sync_copy(floats)
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        Ok(CudaBuffer {
            data: dev_slice,
            width: layout.width,
            height: layout.height,
        })
    }

    fn download(&self, buf: &CudaBuffer) -> Result<Vec<Complex32>> {
        let floats = self.ctx.dev.dtoh_sync_copy(&buf.data)
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        Ok(bytemuck::cast_vec(floats))
    }

    fn name(&self) -> &str {
        "cuda-cufft"
    }
}

// ---------------------------------------------------------------------------
// CudaJob
// ---------------------------------------------------------------------------

pub struct CudaJob<'a> {
    ctx: &'a CudaContext,
}

impl<'a> CudaJob<'a> {
    // -----------------------------------------------------------------------
    // cuFFT helper
    // -----------------------------------------------------------------------
    fn cufft_2d(&self, data: &mut CudaSlice<f32>, width: usize, height: usize,
                direction: i32) -> Result<()> {
        let handle = {
            let mut plans = self.ctx.fft_plans.lock().unwrap();
            if let Some(&h) = plans.get(&(height, width)) {
                h
            } else {
                let mut h: CufftHandle = 0;
                unsafe {
                    let r = cufftCreate(&mut h);
                    if r != 0 {
                        return Err(VernierError::Backend(
                            format!("cufftCreate failed: {}", r)));
                    }
                    let mut ws: usize = 0;
                    let r = cufftMakePlan2d(h, height as i32, width as i32, CUFFT_C2C, &mut ws);
                    if r != 0 {
                        cufftDestroy(h);
                        return Err(VernierError::Backend(
                            format!("cufftMakePlan2d failed: {}", r)));
                    }
                }
                plans.insert((height, width), h);
                h
            }
        };

        // Get the raw CUDA device pointer. DevicePtr::device_ptr() returns &CUdeviceptr
        // which is &u64. Dereference to get the u64 address, then cast to *mut f32.
        unsafe {
            let ptr: *mut f32 = *data.device_ptr() as *mut f32;
            let r = cufftExecC2C(handle, ptr, ptr, direction);
            if r != 0 {
                return Err(VernierError::Backend(
                    format!("cufftExecC2C failed: {}", r)));
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Allocation helper: allocate n f32 values, zeroed.
    // -----------------------------------------------------------------------
    fn alloc_f32(&self, n: usize) -> Result<CudaSlice<f32>> {
        self.ctx.dev.alloc_zeros::<f32>(n)
            .map_err(|e| VernierError::Backend(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // internal gaussian_blur_2d operating on raw CudaBuffer (mutable ref)
    // -----------------------------------------------------------------------
    fn gaussian_blur_2d_inner(&self, buf: &mut CudaBuffer, sigma: f32) -> Result<()> {
        let (width, height) = (buf.width, buf.height);
        let n_floats = buf.n_floats();
        let mut temp = self.alloc_f32(n_floats)?;

        let gx = ((width + 7) / 8) as u32;
        let gy = ((height + 7) / 8) as u32;
        let cfg_2d = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };

        // Horizontal pass: buf -> temp
        unsafe {
            self.ctx.blur_h_fn.clone().launch(
                cfg_2d,
                (&buf.data, &mut temp,
                 width as u32, height as u32, sigma),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // Vertical pass: temp -> buf.data (allocate another temp for output)
        let mut out = self.alloc_f32(n_floats)?;
        unsafe {
            self.ctx.blur_v_fn.clone().launch(
                cfg_2d,
                (&temp, &mut out,
                 width as u32, height as u32, sigma),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        buf.data = out;
        Ok(())
    }
}

impl ComputeJob for CudaJob<'_> {
    type Buffer2D = CudaBuffer;

    fn upload(&mut self, data: &[Complex32], layout: BufferLayout) -> Result<CudaBuffer> {
        layout.check_upload(data.len())?;
        let floats: &[f32] = bytemuck::cast_slice(data);
        let dev_slice = self.ctx.dev.htod_sync_copy(floats)
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        Ok(CudaBuffer {
            data: dev_slice,
            width: layout.width,
            height: layout.height,
        })
    }

    fn copy_buffer(&mut self, src: &CudaBuffer) -> Result<CudaBuffer> {
        let mut dst = self.alloc_f32(src.n_floats())?;
        self.ctx.dev.dtod_copy(&src.data, &mut dst)
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        Ok(CudaBuffer {
            data: dst,
            width: src.width,
            height: src.height,
        })
    }

    fn fft2d(&mut self, buf: &mut CudaBuffer) -> Result<()> {
        self.cufft_2d(&mut buf.data, buf.width, buf.height, CUFFT_FORWARD)
    }

    fn ifft2d(&mut self, buf: &mut CudaBuffer) -> Result<()> {
        let (width, height) = (buf.width, buf.height);
        self.cufft_2d(&mut buf.data, width, height, CUFFT_INVERSE)?;

        // cuFFT inverse is unnormalized: scale by 1/(width*height)
        let n_floats = buf.n_floats() as u32;
        let scale_factor = 1.0_f32 / (width * height) as f32;
        let n_blocks = (n_floats + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.scale_fn.clone().launch(
                cfg,
                (&mut buf.data, n_floats, scale_factor),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))
    }

    fn bandpass_filter(&mut self, buf: &mut CudaBuffer, cx: usize, cy: usize,
                       sigma: Real) -> Result<()> {
        let (width, height) = (buf.width, buf.height);
        let gx = ((width + 7) / 8) as u32;
        let gy = ((height + 7) / 8) as u32;
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.bandpass_fn.clone().launch(
                cfg,
                (&mut buf.data, width as u32, height as u32,
                 cx as u32, cy as u32, sigma as f32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))
    }

    fn bandpass_from_peaks(&mut self, buf: &mut CudaBuffer, peaks: &CudaBuffer,
                           direction: u32, sigma: Real) -> Result<()> {
        let (width, height) = (buf.width, buf.height);
        let gx = ((width + 7) / 8) as u32;
        let gy = ((height + 7) / 8) as u32;
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.bandpass_from_peaks_fn.clone().launch(
                cfg,
                (&mut buf.data, &peaks.data,
                 width as u32, height as u32,
                 direction, sigma as f32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))
    }

    fn filter(&mut self, buf: &mut CudaBuffer, min_frequency: usize,
              max_frequency: usize) -> Result<()> {
        let (width, height) = (buf.width, buf.height);
        let gx = ((width + 7) / 8) as u32;
        let gy = ((height + 7) / 8) as u32;
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.filter_annulus_fn.clone().launch(
                cfg,
                (&mut buf.data, width as u32, height as u32,
                 min_frequency as u32, max_frequency as u32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))
    }

    fn gaussian_blur_2d(&mut self, buf: &mut CudaBuffer, sigma: Real) -> Result<()> {
        self.gaussian_blur_2d_inner(buf, sigma as f32)
    }

    fn extract_phase(&mut self, buf: &CudaBuffer) -> Result<CudaBuffer> {
        let n = buf.n_complex() as u32;
        let mut out = self.alloc_f32(buf.n_floats())?;
        let n_blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.extract_phase_fn.clone().launch(
                cfg,
                (&buf.data, &mut out, n),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;
        Ok(CudaBuffer {
            data: out,
            width: buf.width,
            height: buf.height,
        })
    }

    fn peak_search(&mut self, buffer: &mut CudaBuffer, min_frequency: usize,
                   max_frequency: usize, smoothing_sigma: Real,
                   sigma: Real) -> Result<Option<CudaBuffer>> {
        let (width, height) = (buffer.width, buffer.height);
        let n = width * height;
        let n_groups = (n + 255) / 256;

        // 1. Copy input into working magnitude buffer
        let mut mag_data = self.alloc_f32(buffer.n_floats())?;
        self.ctx.dev.dtod_copy(&buffer.data, &mut mag_data)
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        let mut magnitude = CudaBuffer { data: mag_data, width, height };

        // 2. Convert to magnitude in-place
        let n_elements = n as u32;
        let cfg_1d = LaunchConfig {
            grid_dim: ((n_elements + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.magnitude_fn.clone().launch(
                cfg_1d,
                (&mut magnitude.data, n_elements),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // 3. Annulus mask
        self.filter(&mut magnitude, min_frequency, max_frequency)?;

        // 4. Optional Gaussian blur
        if smoothing_sigma > 0.0 {
            self.gaussian_blur_2d_inner(&mut magnitude, smoothing_sigma as f32)?;
        }

        // 5. First argmax: local reduction
        // intermediate has 2 floats per group: [best_mag, best_index]
        let mut partials1 = self.alloc_f32(n_groups * 2)?;
        let cfg_local = LaunchConfig {
            grid_dim: (n_groups as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.argmax_local_fn.clone().launch(
                cfg_local,
                (&magnitude.data, &mut partials1,
                 width as u32, height as u32, n_elements),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // 6. First argmax: global reduction -> peak1 = [cx, cy]
        let mut peak1 = self.alloc_f32(2)?;
        let cfg_global = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.argmax_global_fn.clone().launch(
                cfg_global,
                (&partials1, &mut peak1,
                 n_groups as u32, width as u32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // 7. Band + angular filter (modifies magnitude in-place)
        let gx = ((width + 7) / 8) as u32;
        let gy = ((height + 7) / 8) as u32;
        let cfg_2d = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.band_angular_filter_fn.clone().launch(
                cfg_2d,
                (&mut magnitude.data, &peak1,
                 width as u32, height as u32, sigma as f32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // 8. Second argmax: local reduction
        let mut partials2 = self.alloc_f32(n_groups * 2)?;
        unsafe {
            self.ctx.argmax_local_fn.clone().launch(
                cfg_local,
                (&magnitude.data, &mut partials2,
                 width as u32, height as u32, n_elements),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // 9. Second argmax: global reduction -> peak2 = [cx, cy]
        let mut peak2 = self.alloc_f32(2)?;
        unsafe {
            self.ctx.argmax_global_fn.clone().launch(
                cfg_global,
                (&partials2, &mut peak2,
                 n_groups as u32, width as u32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // 10. Order peaks -> 8-float result: [cx1, 0, cy1, 0, cx2, 0, cy2, 0]
        let mut result = self.alloc_f32(8)?;
        let cfg_single = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.peak_order_fn.clone().launch(
                cfg_single,
                (&peak1, &peak2, &mut result,
                 width as u32, height as u32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        Ok(Some(CudaBuffer { data: result, width: 2, height: 2 }))
    }

    fn spectral_plane_fit_two(&mut self, spectrum: &CudaBuffer, peaks: &CudaBuffer,
                               sigma: Real) -> Result<CudaBuffer> {
        let (width, height) = (spectrum.width, spectrum.height);
        let n = width * height;
        let n_groups = (n + 63) / 64;

        // 10 floats per group
        let mut partials = self.alloc_f32(n_groups * 10)?;

        let cfg_partial = LaunchConfig {
            grid_dim: (n_groups as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.spectral_partial_fn.clone().launch(
                cfg_partial,
                (&spectrum.data, &peaks.data, &mut partials,
                 width as u32, height as u32, n as u32, sigma as f32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        // 12 floats: [a1,0, b1,0, c1,0, a2,0, b2,0, c2,0]
        let mut result = self.alloc_f32(12)?;
        let cfg_global = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.ctx.spectral_global_fn.clone().launch(
                cfg_global,
                (&partials, &mut result,
                 n_groups as u32, width as f32, height as f32),
            )
        }.map_err(|e| VernierError::Backend(e.to_string()))?;

        Ok(CudaBuffer { data: result, width: 6, height: 1 })
    }

    fn submit(self) -> Result<()> {
        self.ctx.dev.synchronize()
            .map_err(|e| VernierError::Backend(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn checkerboard(w: usize, h: usize) -> (Vec<Complex32>, BufferLayout) {
        let layout = BufferLayout::packed(w, h);
        let mut v = Vec::with_capacity(layout.len());
        for r in 0..h {
            for c in 0..w {
                let val = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                v.push(Complex32::new(val, 0.0));
            }
        }
        (v, layout)
    }

    #[test]
    fn fft_then_ifft_is_identity() {
        let backend = CudaBackend::new().unwrap();
        let (data, layout) = checkerboard(8, 8);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-4);
        }
    }

    #[test]
    fn non_power_of_two_height_round_trips() {
        // cuFFT handles non-power-of-two natively.
        let backend = CudaBackend::new().unwrap();
        let (data, layout) = checkerboard(8, 6);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-3);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-3);
        }
    }
}
