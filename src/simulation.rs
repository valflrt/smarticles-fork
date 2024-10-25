use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

use eframe::egui::Vec2;
use rand::random;
use rayon::prelude::*;

use crate::{mat::Mat2D, CLASS_COUNT, MAX_PARTICLE_COUNT};

pub const FIRST_THRESHOLD: f32 = 10.; // 10.
pub const SECOND_THRESHOLD: f32 = 20.; // 12.

pub const PROXIMITY_POWER: f32 = -60.; // -60.

const DAMPING_FACTOR: f32 = 0.08; // 0.06
const FORCE_SCALING_FACTOR: f32 = 0.0004; // 0.0008

const SPAWN_DENSITY: f32 = 12.;

pub fn compute_force(radius: f32, power: f32) -> f32 {
    if radius < FIRST_THRESHOLD {
        (radius / FIRST_THRESHOLD - 1.) * PROXIMITY_POWER
    } else if radius < FIRST_THRESHOLD + SECOND_THRESHOLD {
        (radius / SECOND_THRESHOLD - FIRST_THRESHOLD / SECOND_THRESHOLD) * power
    } else if radius < FIRST_THRESHOLD + 2. * SECOND_THRESHOLD {
        (-radius / SECOND_THRESHOLD + FIRST_THRESHOLD / SECOND_THRESHOLD + 2.) * power
    } else {
        0.
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Cell(pub i32, pub i32);

impl Cell {
    pub const CELL_SIZE: f32 = FIRST_THRESHOLD + 2. * SECOND_THRESHOLD + 0.1;

    pub fn from_position(position: Vec2) -> Self {
        Self(
            (position.x / Self::CELL_SIZE).floor() as i32,
            (position.y / Self::CELL_SIZE).floor() as i32,
        )
    }

    #[inline]
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
    pub enabled_classes: [bool; CLASS_COUNT],
    pub particle_counts: [usize; CLASS_COUNT],
    /// Matrix containing the power for each particle class with
    /// respect to each other.
    pub power_matrix: Mat2D<i8>,

    pub particle_prev_positions: Mat2D<Vec2>,
    pub particle_positions: Mat2D<Vec2>,

    pub cell_map: HashMap<Cell, Vec<(usize, usize)>>,
}

impl Simulation {
    fn get_neighboring_particles(&self, cell: Cell) -> Vec<(usize, usize)> {
        cell.get_neighbors()
            .iter()
            .filter_map(|neighbor| self.cell_map.get(neighbor))
            .flat_map(|particles| particles.iter().copied())
            // only handle enabled classes
            .filter(|(c, _)| self.enabled_classes[*c])
            .collect()
    }

    fn compute_position_updates(&self) -> Vec<((usize, usize), (Vec2, Vec2))> {
        self.cell_map
            .par_iter()
            .map(|(cell, particles)| {
                // Fetch the particles of neighboring cells
                let neighboring_particules = self.get_neighboring_particles(*cell);

                let mut new_positions: Vec<((usize, usize), (Vec2, Vec2))> = Vec::new();

                for &(c1, p1) in particles.iter().filter(|(c, _)| self.enabled_classes[*c]) {
                    let mut force = Vec2::ZERO;

                    let pos = self.particle_positions[(c1, p1)];

                    // there are only particles from enabled classes in
                    // `neighboring_particules` (see `get_neighboring_particles`)
                    for &(c2, p2) in &neighboring_particules {
                        let power = -self.power_matrix[(c1, c2)];
                        let other_pos = self.particle_positions[(c2, p2)];

                        let distance = other_pos - pos;
                        force -= distance.normalized()
                            * compute_force(distance.length(), power as f32)
                            * FORCE_SCALING_FACTOR;
                    }

                    let prev_pos = self.particle_prev_positions[(c1, p1)];

                    // add damping force
                    force += (prev_pos - pos) * DAMPING_FACTOR;

                    // Verlet integration
                    let new_pos = 2. * pos - prev_pos + force;

                    new_positions.push(((c1, p1), (pos, new_pos)));
                }

                new_positions
            })
            .reduce(
                || Vec::new(),
                |mut acc, v| {
                    acc.extend(v);
                    acc
                },
            )
    }

    pub fn move_particles(&mut self) {
        self.compute_position_updates()
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

        let spawn_radius =
            (self.particle_counts.iter().sum::<usize>() as f32 / PI).sqrt() * SPAWN_DENSITY;

        for c in (0..CLASS_COUNT).filter(|c| self.enabled_classes[*c]) {
            for p in 0..self.particle_counts[c] {
                let angle = TAU * random::<f32>();
                let distance = random::<f32>().sqrt() * spawn_radius;

                let pos = Vec2::new(
                    distance * angle.cos() + (0.5 - random::<f32>()) * spawn_radius,
                    distance * angle.sin() + (0.5 - random::<f32>()) * spawn_radius,
                );

                self.particle_positions[(c, p)] = pos;
                self.particle_prev_positions[(c, p)] = pos;
            }
        }

        self.organize_particles();
    }

    pub fn organize_particles(&mut self) {
        // Remove empty cells from the hashmap and clear non-empty
        // ones
        self.cell_map.retain(|_, particles| {
            if !particles.is_empty() {
                particles.clear();
                true
            } else {
                false
            }
        });
        for c in (0..CLASS_COUNT).filter(|c| self.enabled_classes[*c]) {
            for p in 0..self.particle_counts[c] {
                let particle_index = (c, p);
                let cell = Cell::from_position(self.particle_positions[particle_index]);
                self.cell_map.entry(cell).or_default().push(particle_index);
            }
        }
    }
}

impl Default for Simulation {
    fn default() -> Self {
        let particle_positions = Mat2D::filled_with(Vec2::ZERO, CLASS_COUNT, MAX_PARTICLE_COUNT);
        let mut sim = Self {
            enabled_classes: [true; CLASS_COUNT],
            particle_counts: [200; CLASS_COUNT],
            power_matrix: Mat2D::filled_with(0, CLASS_COUNT, CLASS_COUNT),

            particle_prev_positions: particle_positions.to_owned(),
            particle_positions,

            cell_map: HashMap::new(),
        };
        sim.organize_particles();
        sim
    }
}
