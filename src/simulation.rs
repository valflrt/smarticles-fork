use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

use eframe::egui::Vec2;
use rand::random;
use rayon::prelude::*;

use crate::{mat::Mat2D, CLASS_COUNT, MAX_PARTICLE_COUNT};

const DAMPING_FACTOR: f32 = 0.06;
const FORCE_SCALING_FACTOR: f32 = 0.0008;

pub const FIRST_THRESHOLD: f32 = 10.;
pub const SECOND_THRESHOLD: f32 = 12.;

pub const INTERACTION_THRESHOLD: f32 = FIRST_THRESHOLD + 2. * SECOND_THRESHOLD;

pub const PROXIMITY_POWER: f32 = -60.;

const SPAWN_DENSITY: f32 = 12.;

const DEFAULT_PARTICLE_COUNT: usize = 200;

/// Calcul de la force (force centrale ne dépendant que du
/// rayon)
pub fn compute_force(radius: f32, power: f32) -> f32 {
    if radius < FIRST_THRESHOLD {
        // 1er cas: zone de proximité: la particule repousse les
        // particules trop proches
        (radius / FIRST_THRESHOLD - 1.) * PROXIMITY_POWER
    } else if radius < FIRST_THRESHOLD + SECOND_THRESHOLD {
        // 2ème cas: zone "croissante": la force dépend de la distance
        // entre les deux particules (et croît jusqu'à atteindre le
        // 3ème cas)
        (radius / SECOND_THRESHOLD - FIRST_THRESHOLD / SECOND_THRESHOLD) * power
    } else if radius < INTERACTION_THRESHOLD {
        // 3ème cas: zone "décroissante": la force dépend de la distance
        // entre les deux particules (et décroît jusqu'à atteindre le
        // 4ème cas)
        (-radius / SECOND_THRESHOLD + FIRST_THRESHOLD / SECOND_THRESHOLD + 2.) * power
    } else {
        // 4ème cas: au delà de cette limite, plus aucune force n'est
        // appliquée
        0.
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct Cell(i32, i32);

impl Cell {
    pub const CELL_SIZE: f32 = INTERACTION_THRESHOLD + 0.1;

    /// Créé une cellule à partir d'une position
    pub fn from_position(position: Vec2) -> Self {
        Self(
            (position.x / Self::CELL_SIZE) as i32,
            (position.y / Self::CELL_SIZE) as i32,
        )
    }

    /// Récupère les cellules voisines
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
    /// Nombre de particule pour chaque classe
    pub particle_counts: [usize; CLASS_COUNT],
    /// Matrice contenant les forces de la classe i sur la classe
    /// j
    pub power_matrix: Mat2D<i8>,

    pub particle_prev_positions: Mat2D<Vec2>,
    pub particle_positions: Mat2D<Vec2>,

    /// HashMap associant à chaque cellule une liste d'identifiant
    /// de particule (les particules dans la cellule)
    cell_map: HashMap<Cell, Vec<(usize, usize)>>,
}

impl Simulation {
    /// Récupère les particules des cellules voisines
    fn get_neighboring_particles(&self, cell: Cell) -> Vec<(usize, usize)> {
        // Itération sur les cellules voisines
        cell.get_neighbors()
            .iter()
            // Récupère les particules associées à chaque cellule voisine
            // dans la HashMap
            .filter_map(|neighbor| self.cell_map.get(neighbor))
            .flat_map(|particles| particles.iter().copied())
            .collect()
    }

    /// Calcule les nouvelles positions des particules
    fn compute_positions(&self) -> Vec<((usize, usize), (Vec2, Vec2))> {
        // Itération sur l'ensemble des particules
        (0..CLASS_COUNT)
            .into_par_iter()
            .flat_map(|c1| {
                (0..self.particle_counts[c1])
                    .into_par_iter() // itérateur parallèle
                    .map(move |p1| {
                        // Initialise la force à 0
                        let mut force = Vec2::ZERO;

                        let pos = self.particle_positions[(c1, p1)];
                        let cell = Cell::from_position(pos);

                        // Récupère les particules des cellules voisines
                        let neighboring_particles = self.get_neighboring_particles(cell);

                        for (c2, p2) in neighboring_particles {
                            let power = -self.power_matrix[(c1, c2)];
                            let other_pos = self.particle_positions[(c2, p2)];

                            // Calcul de la distance (vectorielle) entre les deux particules
                            let distance = other_pos - pos;
                            force -= distance.normalized()
                                * compute_force(distance.length(), power as f32)
                                * FORCE_SCALING_FACTOR;
                        }

                        let prev_pos = self.particle_prev_positions[(c1, p1)];

                        // ajout d'une force de frottement
                        force += (prev_pos - pos) * DAMPING_FACTOR;

                        // Calcul de la nouvelle position à l'aide de l'integration
                        // de Verlet:
                        // Si P(n), V(n) et A(n) le vecteur position, vitesse et
                        // acceleration respectivement à l'instant n, on a:
                        //
                        // - V(n+1) = V(n) + A(n+1)
                        // - P(n+1) = P(n) + V(n+1)
                        //
                        // Donc:
                        //
                        // P(n+1) = P(n) + V(n+1)
                        //        = P(n) + ( V(n) + A(n+1) )
                        //        = P(n) + ( P(n) - P(n-1) + A(n+1) )
                        //        = 2*P(n) - P(n-1) + A(n+1)
                        //
                        // On peut ensuite calculer l'accélération à l'aide de la
                        // seconde loi de Newton: m*A(n) = F(n) et en prenant m = 1:
                        //
                        // P(n+1) = 2*P(n) - P(n-1) + F(n)
                        let new_pos = 2. * pos - prev_pos + force;

                        ((c1, p1), (pos, new_pos))
                    })
            })
            .collect::<Vec<((usize, usize), (Vec2, Vec2))>>()
    }

    /// Déplace les particules
    pub fn move_particles(&mut self) {
        self.organize_particles();

        // Mise à jour des positions
        self.compute_positions()
            .iter()
            .for_each(|(index, (pos, new_pos))| {
                self.particle_prev_positions[*index] = *pos;
                self.particle_positions[*index] = *new_pos;
            });
    }

    /// Remise à 0 des particules
    pub fn reset_particles_positions(&mut self) {
        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                self.particle_positions[(c, p)] = Vec2::ZERO;
                self.particle_prev_positions[(c, p)] = Vec2::ZERO
            }
        }
    }

    /// Dispose les particules aléatoirement
    pub fn spawn(&mut self) {
        self.reset_particles_positions();

        let spawn_radius = (self.particle_count() as f32 / PI).sqrt() * SPAWN_DENSITY;

        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                let angle = TAU * random::<f32>();
                let distance = random::<f32>().sqrt() * spawn_radius;

                let pos = Vec2::new(distance * angle.cos(), distance * angle.sin());
                self.particle_positions[(c, p)] = pos;
                self.particle_prev_positions[(c, p)] = pos;
            }
        }
    }

    /// Organise les particules: les particules sont mises dans
    /// la cellule correspondante
    pub fn organize_particles(&mut self) {
        // Supprime les cellules vides
        self.cell_map.retain(|_, particles| {
            if !particles.is_empty() {
                particles.clear();
                true
            } else {
                false
            }
        });
        for c in 0..CLASS_COUNT {
            for p in 0..self.particle_counts[c] {
                let particle_index = (c, p);
                let cell = Cell::from_position(self.particle_positions[particle_index]);
                self.cell_map.entry(cell).or_default().push(particle_index);
            }
        }
    }

    pub fn particle_count(&self) -> usize {
        self.particle_counts.iter().sum::<usize>()
    }
}

impl Default for Simulation {
    /// Configuration initiale du struct Simulation
    fn default() -> Self {
        let particle_positions = Mat2D::filled_with(Vec2::ZERO, CLASS_COUNT, MAX_PARTICLE_COUNT);
        let mut sim = Self {
            particle_counts: [DEFAULT_PARTICLE_COUNT; CLASS_COUNT],
            power_matrix: Mat2D::filled_with(0, CLASS_COUNT, CLASS_COUNT),

            particle_prev_positions: particle_positions.to_owned(),
            particle_positions,

            cell_map: HashMap::new(),
        };
        sim.organize_particles();
        sim
    }
}
