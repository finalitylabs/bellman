#![allow(unused_imports)]
#![allow(unused_variables)]
extern crate bellperson;
extern crate ff;
extern crate paired;
extern crate rand;

// For randomness (during paramgen and proof generation)
use self::rand::{thread_rng, Rng};

// Bring in some tools for using pairing-friendly curves
use self::paired::Engine;

use self::ff::{Field, PrimeField};

// We're going to use the BLS12-381 pairing-friendly elliptic curve.
use self::paired::bls12_381::{Bls12, Fr};

// We'll use these interfaces to construct our circuit.
use self::bellperson::{Circuit, ConstraintSystem, SynthesisError};

// We're going to use the Groth16 proving system.
use self::bellperson::groth16::{
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof, Proof,
};

#[derive(Clone)]
pub struct DummyDemo<E: Engine> {
    pub xx: Option<E::Fr>,
}

impl<E: Engine> Circuit<E> for DummyDemo<E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let mut x_val = E::Fr::from_str("2");
        let mut x = cs.alloc_input(|| "", || x_val.ok_or(SynthesisError::AssignmentMissing))?;

        for k in 0..100000 {
            // Allocate: x * x = x2
            let x2_val = x_val.map(|mut e| {
                e.square();
                e
            });
            let x2 = cs.alloc_input(|| "", || x2_val.ok_or(SynthesisError::AssignmentMissing))?;

            // Enforce: x * x = x2
            cs.enforce(|| "", |lc| lc + x, |lc| lc + x, |lc| lc + x2);

            x = x2;
            x_val = x2_val;
        }

        cs.enforce(
            || "",
            |lc| lc + (x_val.unwrap(), CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + x,
        );

        Ok(())
    }
}
