use super::error::{GPUError, GPUResult};
use crate::multicore::Worker;
use ff::{PrimeField, ScalarEngine};
use groupy::CurveAffine;
use std::marker::PhantomData;
use std::sync::Arc;

// This module is compiled instead of `fft.rs` and `multiexp.rs` if `gpu` feature is disabled.

pub struct FFTKernel<E>(PhantomData<E>)
where
    E: ScalarEngine;

impl<E> FFTKernel<E>
where
    E: ScalarEngine,
{
    pub fn create(_: u32, _: bool) -> GPUResult<FFTKernel<E>> {
        return Err(GPUError::Msg("GPU accelerator is not enabled!".to_string()));
    }

    pub fn radix_fft(&mut self, _: &mut [E::Fr], _: &E::Fr, _: u32) -> GPUResult<()> {
        return Err(GPUError::Msg("GPU accelerator is not enabled!".to_string()));
    }
}

pub struct MultiexpKernel<E>(PhantomData<E>)
where
    E: ScalarEngine;

impl<E> MultiexpKernel<E>
where
    E: ScalarEngine,
{
    pub fn create(_: bool) -> GPUResult<MultiexpKernel<E>> {
        return Err(GPUError::Msg("GPU accelerator is not enabled!".to_string()));
    }

    pub fn multiexp<G>(
        &mut self,
        _: &Worker,
        _: Arc<Vec<G>>,
        _: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        _: usize,
        _: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        return Err(GPUError::Msg("GPU accelerator is not enabled!".to_string()));
    }
}

use paired::Engine;

pub struct LockedFFTKernel<E>
where
    E: Engine,
{
    kernel: Option<FFTKernel<E>>,
}

impl<E> LockedFFTKernel<E>
where
    E: Engine,
{
    pub fn new(_: bool, _: u32) -> LockedFFTKernel<E> {
        LockedFFTKernel::<E> { kernel: None }
    }
    pub fn get(&mut self) -> &mut Option<FFTKernel<E>> {
        &mut self.kernel
    }
}

pub struct LockedMultiexpKernel<E>
where
    E: Engine,
{
    kernel: Option<MultiexpKernel<E>>,
}

impl<E> LockedMultiexpKernel<E>
where
    E: Engine,
{
    pub fn new(_: bool) -> LockedMultiexpKernel<E> {
        LockedMultiexpKernel::<E> { kernel: None }
    }
    pub fn get(&mut self) -> &mut Option<MultiexpKernel<E>> {
        &mut self.kernel
    }
}
