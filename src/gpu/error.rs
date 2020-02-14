use std::fmt;

#[derive(Debug, Clone)]
pub enum GPUError {
    GPUTaken,
    Msg(String),
}

pub type GPUResult<T> = std::result::Result<T, GPUError>;

impl fmt::Display for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let GPUError::Msg(ref e) = *self {
            e.fmt(f)
        } else {
            write!(f, "GPU taken by a high priority process!")
        }
    }
}

#[cfg(feature = "gpu")]
use ocl;

#[cfg(feature = "gpu")]
impl From<ocl::Error> for GPUError {
    fn from(error: ocl::Error) -> Self {
        GPUError::Msg(error.to_string())
    }
}

#[cfg(feature = "gpu")]
impl From<std::boxed::Box<dyn std::any::Any + std::marker::Send>> for GPUError {
    fn from(e: std::boxed::Box<dyn std::any::Any + std::marker::Send>) -> Self {
        match &e.downcast_ref::<Self>() {
            &Some(err) => err.clone(),
            &None => GPUError::Msg("An unknown GPU error happened!".to_string()),
        }
    }
}
