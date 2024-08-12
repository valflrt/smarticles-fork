use std::collections::HashMap;
use std::f32::consts::TAU;

use egui::Vec2;
use rand::random;
use rayon::prelude::*;

use crate::mat::Mat2D;
use crate::{CLASS_COUNT, MAX_PARTICLE_COUNT};

pub const FIRST_THRESHOLD: f32 = 10.;
pub const SECOND_THRESHOLD: f32 = 10.;

pub const PROXIMITY_FORCE: f32 = -60.;

const DAMPING_FACTOR: f32 = 0.1;
const FORCE_SCALING_FACTOR: f32 = 0.0005;

const SPAWN_DENSITY: f32 = 0.03;

pub fn spawn_area_radius(total_particles: f32) -> f32 {
    total_particles * SPAWN_DENSITY
}

pub fn calculate_force(radius: f32, force: f32) -> f32 {
    if radius < FIRST_THRESHOLD {
        (radius / FIRST_THRESHOLD - 1.) * PROXIMITY_FORCE
    } else if radius < FIRST_THRESHOLD + SECOND_THRESHOLD {
        (radius / SECOND_THRESHOLD - FIRST_THRESHOLD / SECOND_THRESHOLD) * force
    } else if radius < FIRST_THRESHOLD + 2. * SECOND_THRESHOLD {
        (-radius / SECOND_THRESHOLD + FIRST_THRESHOLD / SECOND_THRESHOLD + 2.) * force
    } else {
        0.
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct Cell(i32, i32);

impl Cell {
    pub const CELL_SIZE: f32 = FIRST_THRESHOLD + 2. * SECOND_THRESHOLD + 2.;

    pub fn from_position(position: Vec2) -> Self {
        Self(
            (position.x / Self::CELL_SIZE) as i32,
            (position.y / Self::CELL_SIZE) as i32,
        )
    }

    pub const fn get_neighbors(&self) -> [Cell; 9] {
        let Cell(x, y) = *self;
        [
            Cell(x - 1, y - 1),
            Cell(x - 1, y),
            Cell(x - 1, y + 1),
            Cell(x, y - 1),
            Cell(x, y),
            Cell(x, y + 1),
            Cell(x + 1, y - 1),
            Cell(x + 1, y),
            Cell(x + 1, y + 1),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Simulation {
    pub particle_counts: [usize; CLASS_COUNT],
    /// Matrix containing force and radius for each particle class
    /// with respect to each other.
    pub force_matrix: Mat2D<f32>,

    pub particle_prev_positions: Mat2D<Vec2>,
    pub particle_positions: Mat2D<Vec2>,

    cell_map: HashMap<Cell, Vec<(usize, usize)>>,
}

impl Simulation {
    fn get_neighboring_particles(&self, cell: Cell) -> Vec<(usize, usize)> {
        cell.get_neighbors()
            .iter()
            .flat_map(|neighbor| {
                if let Some(particles) = self.cell_map.get(neighbor) {
                    particles.iter().copied()
                } else {
                    [].iter().copied()
                }
            })
            .collect::<Vec<_>>()
    }

    fn calculate_particle_update(&self) -> Vec<((usize, usize), (Vec2, Vec2))> {
        // for c1 in 0..CLASS_COUNT {
        // for c2 in 0..CLASS_COUNT {

        (0..CLASS_COUNT)
            .into_par_iter()
            .flat_map(|c1| {
                (0..self.particle_counts[c1])
                    .into_par_iter()
                    .map(move |p1| {
                        let mut force = Vec2::ZERO;

                        let pos = self.particle_positions[(c1, p1)];
                        let cell = Cell::from_position(pos);

                        let neighboring_particles = self.get_neighboring_particles(cell);
                        // println!("{:?}: {}", cell, neighboring_particles.len());
                        for (c2, p2) in neighboring_particles {
                            let force_factor = -self.force_matrix[(c1, c2)];
                            let other_pos = self.particle_positions[(c2, p2)];

                            let direction = other_pos - pos;
                            force -= direction.normalized()
                                * calculate_force(direction.length(), force_factor)
                                * FORCE_SCALING_FACTOR;
                        }

                        let prev_pos = self.particle_prev_positions[(c1, p1)];

                        force += (prev_pos - pos) * DAMPING_FACTOR;

                        let new_pos = 2. * pos - prev_pos + force;

                        ((c1, p1), (pos, new_pos))
                    })
            })
            .collect::<Vec<((usize, usize), (Vec2, Vec2))>>()

        // }
        // }
    }

    pub fn move_particles(&mut self) {
        self.calculate_particle_update()
            .iter()
            .for_each(|(index, (pos, new_pos))| {
                self.particle_prev_positions[*index] = *pos;
                self.particle_positions[*index] = *new_pos;
            });

        self.organize_particles();
    }

    /// Sets all particle positions to zero.
    pub fn reset_particles_positions(&mut self) {
        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                self.particle_positions[(c, p)] = Vec2::ZERO;
                self.particle_prev_positions[(c, p)] = Vec2::ZERO
            }
        }

        self.organize_particles();
    }

    pub fn spawn(&mut self) {
        self.reset_particles_positions();

        let spawn_radius = spawn_area_radius(self.particle_counts.iter().sum::<usize>() as f32);

        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                let angle = TAU * random::<f32>();
                let distance = random::<f32>().sqrt() * spawn_radius;

                let pos = Vec2::new(distance * angle.cos(), distance * angle.sin());
                self.particle_positions[(c, p)] = pos;
                self.particle_prev_positions[(c, p)] = pos;
            }
        }

        self.organize_particles();
    }

    pub fn organize_particles(&mut self) {
        self.cell_map
            .values_mut()
            .for_each(|particles| particles.clear());
        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                let particle_index = (c, p);
                let cell = Cell::from_position(self.particle_positions[particle_index]);
                self.cell_map
                    .entry(cell)
                    .or_insert(Vec::new())
                    .push(particle_index);
            }
        }
    }
}

impl Default for Simulation {
    fn default() -> Self {
        let particle_positions = Mat2D::filled_with(Vec2::ZERO, CLASS_COUNT, MAX_PARTICLE_COUNT);
        let mut sim = Self {
            particle_counts: [200; CLASS_COUNT],
            force_matrix: Mat2D::filled_with(0., CLASS_COUNT, CLASS_COUNT),

            particle_prev_positions: particle_positions.to_owned(),
            particle_positions,

            cell_map: HashMap::new(),
        };
        sim.organize_particles();
        sim
    }
}
