use log::info;
use ocl::{ProQue, Buffer, MemFlags, Device};
use paired::Engine;
use std::sync::Arc;
use ff::{PrimeField, ScalarEngine};
use paired::{CurveAffine, CurveProjective};
use super::error::{GPUResult, GPUError};
use super::sources;
use super::structs;
use super::utils;
use crossbeam::thread;
use futures::Future;

use crate::multicore::Worker;
use crate::multiexp::{multiexp as cpu_multiexp, FullDensity};

// NOTE: Please read `structs.rs` for an explanation for unsafe transmutes of this code!

// Best params for RTX 2080Ti + AMD Ryzen
const NUM_GROUPS : usize = 334; // Partition the bases into `NUM_GROUPS` groups
const WINDOW_SIZE : usize = 10; // Exponents are 255bit long, divide exponents into `WINDOW_SIZE` bit windows
const NUM_WINDOWS : usize = 26; // Then we will have Ceil(256/`WINDOW_SIZE`) windows per exponent
const CHUNK_SIZE : usize = 15_000_000; // Maximum number of base elements we can pass to a GPU
const SPEEDUP : f64 = 3.1f64; // Speedup of a single GPU compared to CPU
// So each group will have `NUM_WINDOWS` threads and as there are `NUM_GROUPS` groups, there will
// be `NUM_GROUPS` * `NUM_WINDOWS` threads in total.

const LOCAL_WORK_SIZE : usize = 256;
const BUCKET_LEN : usize = 1 << WINDOW_SIZE;

// Multiexp kernel for a single GPU
pub struct SingleMultiexpKernel<E> where E: Engine {
    proque: ProQue,

    g1_base_buffer: Buffer<structs::CurveAffineStruct<E::G1Affine>>,
    g1_bucket_buffer: Buffer<structs::CurveProjectiveStruct<E::G1>>,
    g1_result_buffer: Buffer<structs::CurveProjectiveStruct<E::G1>>,

    g2_base_buffer: Buffer<structs::CurveAffineStruct<E::G2Affine>>,
    g2_bucket_buffer: Buffer<structs::CurveProjectiveStruct<E::G2>>,
    g2_result_buffer: Buffer<structs::CurveProjectiveStruct<E::G2>>,

    exp_buffer: Buffer<structs::PrimeFieldStruct<E::Fr>>
}

impl<E> SingleMultiexpKernel<E> where E: Engine {

    pub fn create(d: Device, n: u32) -> GPUResult<SingleMultiexpKernel<E>> {
        let src = sources::multiexp_kernel::<E>();
        let pq = ProQue::builder().device(d).src(src).dims(n).build()?;

        let g1basebuff = Buffer::builder().queue(pq.queue().clone()).flags(MemFlags::new().read_write()).len(n).build()?;
        let g1buckbuff = Buffer::builder().queue(pq.queue().clone()).flags(MemFlags::new().read_write()).len(BUCKET_LEN * NUM_WINDOWS * NUM_GROUPS).build()?;
        let g1resbuff = Buffer::builder().queue(pq.queue().clone()).flags(MemFlags::new().read_write()).len(NUM_WINDOWS * NUM_GROUPS).build()?;

        let g2basebuff = Buffer::builder().queue(pq.queue().clone()).flags(MemFlags::new().read_write()).len(n).build()?;
        let g2buckbuff = Buffer::builder().queue(pq.queue().clone()).flags(MemFlags::new().read_write()).len(BUCKET_LEN * NUM_WINDOWS * NUM_GROUPS).build()?;
        let g2resbuff = Buffer::builder().queue(pq.queue().clone()).flags(MemFlags::new().read_write()).len(NUM_WINDOWS * NUM_GROUPS).build()?;

        let expbuff = Buffer::builder().queue(pq.queue().clone()).flags(MemFlags::new().read_write()).len(n).build()?;

        Ok(SingleMultiexpKernel {proque: pq,
            g1_base_buffer: g1basebuff, g1_bucket_buffer: g1buckbuff, g1_result_buffer: g1resbuff,
            g2_base_buffer: g2basebuff, g2_bucket_buffer: g2buckbuff, g2_result_buffer: g2resbuff,
            exp_buffer: expbuff})
    }

    pub fn multiexp<G>(&mut self,
            bases: &[G],
            exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
            n: usize)
            -> GPUResult<(<G as CurveAffine>::Projective)>
            where G: CurveAffine {

        let exp_bits = std::mem::size_of::<E::Fr>() * 8;

        let mut res = vec![<G as CurveAffine>::Projective::zero(); NUM_WINDOWS * NUM_GROUPS];
        let texps = unsafe { std::mem::transmute::<&[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr], &[structs::PrimeFieldStruct::<E::Fr>]>(exps) };
        self.exp_buffer.write(texps).enq()?;

        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let mut gws = NUM_WINDOWS * NUM_GROUPS;
        gws += (LOCAL_WORK_SIZE - (gws % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let sz = std::mem::size_of::<G>(); // Trick, used for dispatching between G1 and G2!
        if sz == std::mem::size_of::<E::G1Affine>() {
            let tbases = unsafe { std::mem::transmute::<&[G], &[structs::CurveAffineStruct<E::G1Affine>]>(bases) };
            self.g1_base_buffer.write(tbases).enq()?;
            let kernel = self.proque.kernel_builder("G1_bellman_multiexp")
                .global_work_size([gws])
                .arg(&self.g1_base_buffer)
                .arg(&self.g1_bucket_buffer)
                .arg(&self.g1_result_buffer)
                .arg(&self.exp_buffer)
                .arg(n as u32)
                .arg(NUM_GROUPS as u32)
                .arg(NUM_WINDOWS as u32)
                .arg(WINDOW_SIZE as u32)
                .build()?;
            unsafe { kernel.enq()?; }
            let tres = unsafe { std::mem::transmute::<&mut [<G as CurveAffine>::Projective], &mut [structs::CurveProjectiveStruct::<E::G1>]>(&mut res) };
            self.g1_result_buffer.read(tres).enq()?;

        } else if sz == std::mem::size_of::<E::G2Affine>() {
            let tbases = unsafe { std::mem::transmute::<&[G], &[structs::CurveAffineStruct<E::G2Affine>]>(bases) };
            self.g2_base_buffer.write(tbases).enq()?;
            let kernel = self.proque.kernel_builder("G2_bellman_multiexp")
                .global_work_size([gws])
                .arg(&self.g2_base_buffer)
                .arg(&self.g2_bucket_buffer)
                .arg(&self.g2_result_buffer)
                .arg(&self.exp_buffer)
                .arg(n as u32)
                .arg(NUM_GROUPS as u32)
                .arg(NUM_WINDOWS as u32)
                .arg(WINDOW_SIZE as u32)
                .build()?;
            unsafe { kernel.enq()?; }
            let tres = unsafe { std::mem::transmute::<&mut [<G as CurveAffine>::Projective], &mut [structs::CurveProjectiveStruct::<E::G2>]>(&mut res) };
            self.g2_result_buffer.read(tres).enq()?;
        } else {
            return Err(GPUError {msg: "Only E::G1 and E::G2 are supported!".to_string()} );
        }

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..NUM_WINDOWS {
            let w = std::cmp::min(WINDOW_SIZE, exp_bits - bits);
            for _ in 0..w { acc.double(); }
            for g in 0..NUM_GROUPS {
                acc.add_assign(&res[g * NUM_WINDOWS + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

// A struct that containts several multiexp kernels for different devices
pub struct MultiexpKernel<E>(Vec<SingleMultiexpKernel<E>>) where E: Engine;

impl<E> MultiexpKernel<E> where E: Engine {

    pub fn create() -> GPUResult<MultiexpKernel<E>> {
        let devices = utils::get_devices(utils::GPU_NVIDIA_PLATFORM_NAME)?;
        if devices.len() == 0 { return Err(GPUError {msg: "No working GPUs found!".to_string()} ); }
        let mut kernels = Vec::new();
        for dev in devices.into_iter().map(|device| { SingleMultiexpKernel::<E>::create(device, CHUNK_SIZE as u32) }) {
            kernels.push(dev?);
        }
        info!("Multiexp: {} working device(s) selected.", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!("Multiexp: Device {}: {}", i, k.proque.device().name()?);
        }
        return Ok(MultiexpKernel::<E>(kernels));
    }

    pub fn multiexp<G>(&mut self,
            pool: &Worker,
            bases: Arc<Vec<G>>,
            exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
            skip: usize,
            n: usize)
            -> GPUResult<(<G as CurveAffine>::Projective)>
            where G: CurveAffine {

        use std::time::Instant;

        let num_devices = self.0.len();

        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..];

        let exps = &exps[..n];

        // Observations show that a GPU can calculate multiexp of `SPEEDUP * n` elements in the
        // same amount of time a CPU calculates multiexp of `n` elements.
        // Therefore we should calculate `cpu_n` so that: `N = (cpu_n * SPEEDUP) * num_gpus + cpu_n`
        // This yields: `N = cpu_n * (SPEEDUP * num_gpus + 1) => cpu_n = N / (SPEEDUP * num_gpus + 1)`
        let cpu_n = ((n as f64) / ((num_devices as f64) * SPEEDUP + 1.0f64)).ceil() as usize;
        let n = n - cpu_n;

        let now = Instant::now();
        let (cpu_bases, bases) = bases.split_at(cpu_n);
        let (cpu_exps, exps) = exps.split_at(cpu_n);
        println!("Chunking: took {}ms.", now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64);
        println!("CPU: {} GPUs: {}", cpu_bases.len(), bases.len());

        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

        match thread::scope(|s| {
            let mut threads = Vec::new();
            if chunk_size > 0 {
                for ((bases, exps), kern) in bases.chunks(chunk_size).zip(exps.chunks(chunk_size)).zip(self.0.iter_mut()) {
                    threads.push(s.spawn(move |s| {
                        let now = Instant::now();
                        let mut acc = <G as CurveAffine>::Projective::zero();
                        for (bases, exps) in bases.chunks(CHUNK_SIZE).zip(exps.chunks(CHUNK_SIZE)) {
                            let result = kern.multiexp(bases, exps, bases.len()).unwrap();
                            acc.add_assign(&result);
                        }
                        println!("Single GPU Multiexp: took {}ms.", now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64);
                        acc
                    }));
                }
            }

            let now = Instant::now();
            let mut acc = cpu_multiexp(pool, (Arc::new(cpu_bases.to_vec()), 0), FullDensity, Arc::new(cpu_exps.to_vec()), &mut None).wait().unwrap();
            println!("CPU Multiexp: took {}ms.", now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64);

            for t in threads {
                let result = t.join().unwrap();
                acc.add_assign(&result);
            }
            acc
        }) {
            Ok(acc) => { Ok(acc) },
            Err(e) => {
                match &e.downcast_ref::<GPUError>() {
                    &Some(err) => Err(err.clone()),
                    &None => Err(GPUError {msg: "Multigpu Multiexp failed!".to_string()})
                }
            }
        }
    }
}
