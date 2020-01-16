mod error;
pub use self::error::*;

#[cfg(feature = "gpu")]
mod locks;
#[cfg(feature = "gpu")]
pub use self::locks::*;

#[cfg(feature = "gpu")]
mod sources;
#[cfg(feature = "gpu")]
pub use self::sources::*;

#[cfg(feature = "gpu")]
mod utils;
#[cfg(feature = "gpu")]
pub use self::utils::*;

#[cfg(feature = "gpu")]
mod structs;
#[cfg(feature = "gpu")]
pub use self::structs::*;

#[cfg(feature = "gpu")]
mod fft;
#[cfg(feature = "gpu")]
pub use self::fft::*;

#[cfg(feature = "gpu")]
mod multiexp;
#[cfg(feature = "gpu")]
pub use self::multiexp::*;

#[cfg(not(feature = "gpu"))]
mod nogpu;
#[cfg(not(feature = "gpu"))]
pub use self::nogpu::*;

#[cfg(feature = "gpu")]
use ocl::Device;
#[cfg(feature = "gpu")]
lazy_static::lazy_static! {
    pub static ref GPU_NVIDIA_DEVICES: Vec<Device> = get_devices(GPU_NVIDIA_PLATFORM_NAME).unwrap_or_default();
}

pub struct LockedKernel<K, F>
where
    F: Fn() -> Option<K>,
{
    _f: F,
    kernel: Option<K>,
}

impl<K, F> LockedKernel<K, F>
where
    F: Fn() -> Option<K>,
{
    pub fn new(f: F) -> LockedKernel<K, F> {
        LockedKernel::<K, F> {
            _f: f,
            kernel: None,
        }
    }
    pub fn get(&mut self) -> &mut Option<K> {
        #[cfg(feature = "gpu")]
        {
            use log::{info, warn};
            if !PriorityLock::can_lock() {
                if let Some(_kernel) = self.kernel.take() {
                    warn!("GPU acquired by a high priority process! Freeing up kernels...");
                }
            } else if self.kernel.is_none() {
                info!("GPU is available!");
                self.kernel = (self._f)();
            }
        }
        &mut self.kernel
    }
}
