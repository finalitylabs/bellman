use fs2::FileExt;
use log::{info, warn};
use std::fs::File;
use std::path::PathBuf;

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";
fn tmp_path(filename: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(filename);
    p
}

#[derive(Debug)]
pub struct GPULock(File);
impl GPULock {
    pub fn lock() -> GPULock {
        info!("Acquiring GPU lock...");
        let f = File::create(tmp_path(GPU_LOCK_NAME)).unwrap();
        f.lock_exclusive().unwrap();
        info!("GPU lock acquired!");
        GPULock(f)
    }
}
impl Drop for GPULock {
    fn drop(&mut self) {
        info!("GPU lock released!");
    }
}

use std::cell::RefCell;
thread_local!(static IS_ME: RefCell<bool> = RefCell::new(false));

#[derive(Debug)]
pub struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        info!("Acquiring priority lock...");
        let f = File::create(tmp_path(PRIORITY_LOCK_NAME)).unwrap();
        f.lock_exclusive().unwrap();
        IS_ME.with(|f| *f.borrow_mut() = true);
        info!("Priority lock acquired!");
        PriorityLock(f)
    }
    pub fn can_lock() -> bool {
        // Either taken by me or not taken by somebody else
        let is_me = IS_ME.with(|f| *f.borrow());
        is_me
            || File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .try_lock_exclusive()
                .is_ok()
    }
}
impl Drop for PriorityLock {
    fn drop(&mut self) {
        IS_ME.with(|f| *f.borrow_mut() = false);
        info!("Priority lock released!");
    }
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
