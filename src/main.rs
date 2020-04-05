#![allow(unused_imports)]
#![allow(unused_variables)]
extern crate bellperson;
extern crate ff;
extern crate paired;
extern crate rand;
use bellperson::groth16::Parameters;
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use ff::{Field, PrimeField};
use paired::Engine;

use std::fs::File;
use std::io::prelude::*;

use std::thread::sleep;
use std::time::{Duration, Instant};

mod dummy;

fn main() {
    use num_cpus;
    println!("Num cpus {}", num_cpus::get());
    //env_logger::init();
    use bellperson::groth16::{
        create_random_proof, create_random_proof_batch, generate_random_parameters,
        prepare_batch_verifying_key, verify_proofs_batch, Proof,
    };
    use paired::bls12_381::{Bls12, Fr};
    use rand::thread_rng;

    println!("I know the value of 2^(2^1000)");

    let rng = &mut thread_rng();

    println!("Creating parameters...");

    let load_parameters = false;
    let parameters_path = "parameters.dat";

    // Create parameters for our circuit
    let params = if load_parameters {
        let param_file = File::open(parameters_path).expect("Unable to open parameters file!");
        Parameters::<Bls12>::read(param_file, false /* false for better performance*/)
            .expect("Unable to read parameters file!")
    } else {
        let c = dummy::DummyDemo::<Bls12> { xx: None };

        let p = generate_random_parameters(c, rng).unwrap();
        let mut param_file =
            File::create(parameters_path).expect("Unable to create parameters file!");
        p.write(param_file)
            .expect("Unable to write parameters file!");
        p
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_batch_verifying_key(&params.vk);

    // Create an instance of circuit
    let c = dummy::DummyDemo::<Bls12> {
        xx: Fr::from_str("3"),
    };

    const SAMPLES: usize = 500;
    println!("Batch proof");

    let proofs = create_random_proof_batch(vec![c; SAMPLES], &params, rng).unwrap();
    let pref = proofs.iter().collect::<Vec<&_>>();
    println!("Verifying...");
    let now = Instant::now();
    let mut haha = vec![Fr::from_str("2").unwrap()];
    for i in 0..100 {
        haha.push(haha[i]);
        let cop = haha[i];
        haha[i + 1].mul_assign(&cop);
    }
    println!(
        "{}",
        verify_proofs_batch(&pvk, rng, &pref[..], &vec![haha.clone(); 500]).unwrap()
    );
    println!(
        "Verification finished in {}s and {}ms for {} proofs",
        now.elapsed().as_secs(),
        now.elapsed().subsec_nanos() / 1000000,
        SAMPLES
    );
}
