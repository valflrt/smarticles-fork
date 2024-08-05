use std::f32::consts::TAU;

use egui::Vec2;
use rand::distributions::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::mat::Mat2D;
use crate::{CLASS_COUNT, MAX_PARTICLE_COUNT};

pub const FIRST_THRESHOLD: f32 = 10.;
pub const SECOND_THRESHOLD: f32 = 15.;

pub const PROXIMITY_FORCE: f32 = -80.;

pub const FORCE_REDUCING_FACTOR: f32 = -0.00001;

pub const DAMPING_FACTOR: f32 = 0.01;

/// Radius of the spawn area.
const SPAWN_AREA_RADIUS: f32 = 50.;

#[derive(Debug, Clone)]
pub struct Simulation {
    pub particle_counts: [usize; CLASS_COUNT],
    /// Matrix containing force and radius for each particle class
    /// with respect to each other.
    pub force_matrix: Mat2D<f32>,

    pub particle_prev_positions: Mat2D<Vec2>,
    pub particle_positions: Mat2D<Vec2>,
}

impl Simulation {
    pub fn move_particles(&mut self) {
        for c1 in 0..CLASS_COUNT {
            for c2 in 0..CLASS_COUNT {
                let force_factor = -self.force_matrix[(c1, c2)];

                (0..self.particle_counts[c1])
                    .into_par_iter()
                    .map(|p1| {
                        let mut force = Vec2::ZERO;

                        let pos = self.particle_positions[(c1, p1)];
                        for p2 in 0..self.particle_counts[c2] {
                            let other_pos = self.particle_positions[(c2, p2)];
                            let direction = other_pos - pos;
                            force += calculate_force(direction.length(), force_factor)
                                * direction.normalized();
                        }

                        let prev_pos = self.particle_prev_positions[(c1, p1)];

                        force += (prev_pos - pos) * DAMPING_FACTOR;

                        let new_pos = 2. * pos - prev_pos + force;

                        (pos, new_pos)
                    })
                    .collect::<Vec<(Vec2, Vec2)>>()
                    .iter()
                    .enumerate()
                    .for_each(|(p1, (pos, new_pos))| {
                        self.particle_prev_positions[(c1, p1)] = *pos;
                        self.particle_positions[(c1, p1)] = *new_pos;
                    });
            }
        }
    }

    /// Sets all particle positions to zero.
    pub fn reset_particles_positions(&mut self) {
        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                self.particle_positions[(c, p)] = Vec2::ZERO;
                self.particle_prev_positions[(c, p)] = Vec2::ZERO
            }
        }
    }

    pub fn spawn(&mut self) {
        self.reset_particles_positions();

        let mut rand = SmallRng::from_entropy();

        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                let pos = SPAWN_AREA_RADIUS
                    * Vec2::angled(TAU * rand.sample::<f32, _>(Open01))
                    * rand.sample::<f32, _>(Open01);
                self.particle_positions[(c, p)] = pos;
                self.particle_prev_positions[(c, p)] = pos;
            }
        }
    }
}
impl Default for Simulation {
    fn default() -> Self {
        let particle_positions = Mat2D::filled_with(Vec2::ZERO, CLASS_COUNT, MAX_PARTICLE_COUNT);
        Self {
            particle_counts: [200; CLASS_COUNT],
            force_matrix: Mat2D::filled_with(0., CLASS_COUNT, CLASS_COUNT),

            particle_prev_positions: particle_positions.to_owned(),
            particle_positions,
        }
    }
}

pub fn calculate_force(radius: f32, force: f32) -> f32 {
    (if radius < FIRST_THRESHOLD {
        (radius / FIRST_THRESHOLD - 1.) * PROXIMITY_FORCE
    } else if radius < FIRST_THRESHOLD + SECOND_THRESHOLD {
        (radius / SECOND_THRESHOLD - FIRST_THRESHOLD / SECOND_THRESHOLD) * force
    } else if radius < FIRST_THRESHOLD + 2. * SECOND_THRESHOLD {
        (-radius / SECOND_THRESHOLD + FIRST_THRESHOLD / SECOND_THRESHOLD + 2.) * force
    } else {
        0.
    }) * FORCE_REDUCING_FACTOR
}
