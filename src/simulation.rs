use std::f32::consts::PI;

use eframe::egui::Vec2;
use fnv::FnvHashMap;
use rand::random;
use rayon::prelude::*;

use crate::{mat::Mat2D, CLASS_COUNT, MAX_PARTICLE_COUNT};

pub const PROXIMITY_POWER: f32 = -160.;

// TODO make this interactive (changeable from ui)
const DAMPING_FACTOR: f32 = 100.;
const DT: f32 = 0.0004;

const SPAWN_DENSITY: f32 = 0.035;

pub const FIRST_THRESHOLD: f32 = 4.;
pub const SECOND_THRESHOLD: f32 = 20.;
/// Range in which particles interact. If two particles are
/// farther away than this distance, the will never interact.
const INTERACTION_RANGE: f32 = FIRST_THRESHOLD + 2. * SECOND_THRESHOLD;

pub fn compute_force(radius: f32, power: f32) -> f32 {
    if radius < FIRST_THRESHOLD {
        (-radius / FIRST_THRESHOLD + 1.) * PROXIMITY_POWER
    } else if radius < FIRST_THRESHOLD + SECOND_THRESHOLD {
        (-radius / SECOND_THRESHOLD + FIRST_THRESHOLD / SECOND_THRESHOLD) * power
    } else if radius < FIRST_THRESHOLD + 2. * SECOND_THRESHOLD {
        (radius / SECOND_THRESHOLD - FIRST_THRESHOLD / SECOND_THRESHOLD - 2.) * power
    } else {
        0.
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum Quarter {
    NorthEast,
    SouthEast,
    SouthWest,
    NorthWest,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Cell(pub i32, pub i32, pub Option<Quarter>);

impl Cell {
    pub const CELL_SIZE: f32 = INTERACTION_RANGE / 3.;

    pub fn from_position(position: Vec2) -> Self {
        Self(
            (position.x / Self::CELL_SIZE).floor() as i32,
            (position.y / Self::CELL_SIZE).floor() as i32,
            None,
        )
    }

    pub fn get_neighbors(&self) -> impl Iterator<Item = Cell> {
        // Note: The value of R must always be greater than
        // `ceil(INTERACTION_RANGE / Cell::CELL_SIZE)`
        const R: i32 = (INTERACTION_RANGE / Cell::CELL_SIZE) as i32;

        let Cell(x, y, quarter) = *self;

        let (rows, columns) = if let Some(quarter) = quarter {
            match quarter {
                //     + > i
                //     v
                //     j
                //                     i      j
                Quarter::NorthEast => (0..=R, -R..=-1),
                Quarter::SouthEast => (0..=R, 0..=R),
                Quarter::SouthWest => (-R..=-1, 0..=R),
                Quarter::NorthWest => (-R..=-1, -R..=-1),
            }
        } else {
            (-R..=R, -R..=R)
        };

        rows.flat_map(move |i| columns.clone().map(move |j| (i, j)))
            .map(move |(i, j)| Cell(x + i, y + j, quarter))
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

    pub cell_map: FnvHashMap<Cell, Vec<(usize, usize)>>,
}

impl Simulation {
    pub fn move_particles(&mut self) {
        self.compute_position_updates()
            .iter()
            .for_each(|(index, (pos, new_pos))| {
                self.particle_prev_positions[*index] = *pos;
                self.particle_positions[*index] = *new_pos;
            });

        self.organize_particles();
    }

    pub fn spawn(&mut self) {
        let spawn_radius =
            (self.particle_counts.iter().sum::<usize>() as f32 / PI).sqrt() / SPAWN_DENSITY;

        for c in (0..CLASS_COUNT).filter(|c| self.enabled_classes[*c]) {
            for p in 0..self.particle_counts[c] {
                let mut pos =
                    Vec2::new(0.5 - random::<f32>(), 0.5 - random::<f32>()) * spawn_radius;

                for i in 2..=4 {
                    pos += Vec2::new(0.5 - random::<f32>(), 0.5 - random::<f32>()) * spawn_radius
                        / i as f32
                }

                self.particle_positions[(c, p)] = pos;
                self.particle_prev_positions[(c, p)] = pos;
            }
        }

        self.organize_particles();
    }

    fn compute_position_updates(&self) -> Vec<((usize, usize), (Vec2, Vec2))> {
        self.cell_map
            .par_iter()
            .map(|(&cell, particles)| {
                // Fetch the particles of neighboring cells
                let neighboring_particles = self.get_neighboring_particles(cell);

                let mut new_positions: Vec<((usize, usize), (Vec2, Vec2))> = Vec::new();

                for &(c1, p1) in particles.iter().filter(|(c, _)| self.enabled_classes[*c]) {
                    let mut force = Vec2::ZERO;

                    let pos = self.particle_positions[(c1, p1)];

                    // there are only particles from enabled classes in
                    // `neighboring_particules` (see `get_neighboring_particles`)
                    for &(c2, p2) in &neighboring_particles {
                        let power = -self.power_matrix[(c2, c1)]; // power of the force applied by c2 on c1
                        let other_pos = self.particle_positions[(c2, p2)];

                        let distance = other_pos - pos;
                        force +=
                            distance.normalized() * compute_force(distance.length(), power as f32);
                    }

                    let prev_pos = self.particle_prev_positions[(c1, p1)];

                    // scale calculated force and add damping
                    force += (prev_pos - pos) * DAMPING_FACTOR;

                    // Verlet integration
                    let new_pos = 2. * pos - prev_pos + force * DT;

                    new_positions.push(((c1, p1), (pos, new_pos)));
                }

                new_positions
            })
            .reduce(Vec::new, |mut acc, v| {
                acc.extend(v);
                acc
            })
    }

    fn get_neighboring_particles(&self, cell: Cell) -> Vec<(usize, usize)> {
        cell.get_neighbors()
            // .iter()
            // get non-empty cells
            .filter_map(|neighbor| self.cell_map.get(&neighbor))
            .flat_map(|particles| particles.iter().copied())
            // keep particles from enabled classes only
            .filter(|&(c, _)| self.enabled_classes[c])
            .collect()
    }

    pub fn organize_particles(&mut self) {
        // Remove empty cells from the hashmap and clear non-empty
        // ones
        self.cell_map.retain(|_, particles| {
            if !particles.is_empty() {
                particles.clear(); // Clear retained cells
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
            particle_counts: [0; CLASS_COUNT],
            power_matrix: Mat2D::filled_with(0, CLASS_COUNT, CLASS_COUNT),

            particle_prev_positions: particle_positions.to_owned(),
            particle_positions,

            cell_map: FnvHashMap::default(),
        };
        sim.spawn();
        sim
    }
}
