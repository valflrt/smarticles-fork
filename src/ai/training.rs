use std::{
    array,
    f32::consts::{PI, TAU},
};

use eframe::egui::Vec2;
use rand::random;

use crate::{mat::Mat2D, simulation::Simulation, CLASS_COUNT, MAX_POWER};

use super::net::Network;

pub const BATCH_SIZE: usize = 50;

pub const NETWORK_INPUT_SIZE: usize = 1 + CLASS_COUNT * CLASS_COUNT;
pub const NETWORK_OUTPUT_SIZE: usize = CLASS_COUNT * CLASS_COUNT;

pub const INFERENCE_TICK_INTERVAL: usize = 20;

/// Formate les entrées en un vecteur
pub fn adapt_input(target_angle: f32, power_matrix: Mat2D<i8>) -> Vec<f32> {
    [
        &[10. * target_angle / TAU],
        &power_matrix
            .iter()
            .map(|x| *x as f32 / 10.)
            .collect::<Vec<_>>()[..],
    ]
    .concat()
}
/// Change de `power_matrix` selon la sortie du réseau donnée
pub fn apply_output(output: Vec<f32>, power_matrix: &mut Mat2D<i8>) {
    power_matrix
        .vec_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, power)| {
            *power = (*power + (output[i] * 10.) as i8).clamp(-MAX_POWER, MAX_POWER);
        });
}

pub fn random_target_angle() -> f32 {
    PI * (random::<f32>() * 2. - 1.)
}

/// Prépare la simulation pour les réseaux de neurones
pub fn setup_simulation_for_networks(sim: &mut Simulation) {
    sim.particle_counts = [8; CLASS_COUNT];

    sim.power_matrix.vec_mut().iter_mut().for_each(|p| *p = 0);

    sim.reset_particles_positions();

    let spawn_radius = (sim.particle_count() as f32 / PI).sqrt() * 3.;

    for c in 0..CLASS_COUNT {
        let class_angle = TAU * (c as f32 / CLASS_COUNT as f32);

        for p in 0..sim.particle_counts[c] {
            let particle_angle = TAU * random::<f32>();
            let distance = random::<f32>().sqrt() * spawn_radius * 0.5;

            let pos =
                spawn_radius * Vec2::angled(class_angle) + distance * Vec2::angled(particle_angle);
            sim.particle_positions[(c, p)] = pos;
            sim.particle_prev_positions[(c, p)] = pos;
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationData {
    /// The average of the projections of the movement direction
    /// on the target direction
    movement_projection_avg: f32,
    /// Distance maximale entre les particules
    max_distance_between_particles: f32,
    /// Distance moyenne entre les particules
    distance_between_particles_avg: f32,
}

pub fn evaluation_fn(network: Network) -> EvaluationData {
    let mut evaluation_data = EvaluationData {
        movement_projection_avg: 0.,
        max_distance_between_particles: 0.,
        distance_between_particles_avg: 0.,
    };

    let mut sim = Simulation::default();
    setup_simulation_for_networks(&mut sim);

    let mut target_angle = random_target_angle();

    let mut prev_ggc = calc_geometric_center(&sim);

    const MAX_TICK_COUNT: usize = 6000;
    const MAX_INFERENCE_TICK_COUNT: usize = MAX_TICK_COUNT / INFERENCE_TICK_INTERVAL;

    // boucle permettant d'évaluer le réseau pour un certain temps
    for tick_count in 0..MAX_INFERENCE_TICK_COUNT {
        let _tick_count: usize = tick_count * INFERENCE_TICK_INTERVAL;

        // lance INFERENCE_TICK_INTERVAL mises à jour de la position
        // des particules
        for _ in 0..INFERENCE_TICK_INTERVAL {
            sim.move_particles();
            // petite modification de l'angle cible
            target_angle += (random::<f32>() * 2. - 1.) * 0.1;
            if target_angle > PI {
                target_angle -= 2. * PI;
            } else if target_angle < -PI {
                target_angle += 2. * PI;
            }
        }

        // calcul du centre géométrique
        let gc = calc_geometric_center(&sim);

        // calcul du vecteur déplacement
        let movement = gc - prev_ggc;

        // calcul du produit scalaire
        evaluation_data.movement_projection_avg += (movement.x * target_angle.cos()
            + movement.y * target_angle.sin())
            / MAX_INFERENCE_TICK_COUNT as f32;

        // calcul de la distance moyenne entre les particules
        // complexité en O(N^2) avec N le nombre de particules
        let mut distance_between_particles_sum = 0.;
        for c1 in 0..CLASS_COUNT {
            for p1 in 0..sim.particle_counts[c1] {
                for c2 in 0..CLASS_COUNT {
                    for p2 in 0..sim.particle_counts[c2] {
                        let distance = (sim.particle_positions[(c1, p1)]
                            - sim.particle_positions[(c2, p2)])
                            .length();
                        evaluation_data.max_distance_between_particles =
                            distance.max(evaluation_data.max_distance_between_particles);
                        distance_between_particles_sum += distance;
                    }
                }
            }
        }
        evaluation_data.distance_between_particles_avg += distance_between_particles_sum
            / (sim.particle_count().pow(2) * MAX_INFERENCE_TICK_COUNT) as f32;

        prev_ggc = gc;

        // applique les paramètres donnés par le réseau aux paramètres
        // de la simulation
        let output = network.infer(adapt_input(target_angle, sim.power_matrix.to_owned()));
        apply_output(output, &mut sim.power_matrix);
    }

    evaluation_data
}

/// Calcule le score d'un réseau à partir des données récoltées
/// durant l'évaluation
pub fn compute_score(evaluation_data: EvaluationData) -> f32 {
    println!("{:#?}", evaluation_data);
    let EvaluationData {
        // speed,
        movement_projection_avg,
        max_distance_between_particles,
        distance_between_particles_avg,
        ..
    } = evaluation_data;

    let particle_distance_coefficient =
        max_distance_between_particles * 0.4 + distance_between_particles_avg * 0.6;

    let score = if movement_projection_avg.is_sign_positive() {
        movement_projection_avg.powi(3) / particle_distance_coefficient
    } else {
        movement_projection_avg.powi(3) * particle_distance_coefficient / 10.
    };

    println!("{:?}", score);
    score
}

/// Returns the geometric centers of each class, the global
/// geometric center, the mean distances between the geometric
/// center each particle of each class and the mean distance
/// between each particle and the global geomtric center:
/// `(gcs, ggc, mean_distances, global_mean_distance)`
pub fn calc_geometric_center(simulation: &Simulation) -> Vec2 {
    let mut gc_sum = Vec2::ZERO;
    let _gcs: [Vec2; CLASS_COUNT] = array::from_fn(|c| {
        let gc = (0..simulation.particle_counts[c]).fold(Vec2::ZERO, |acc, p| {
            acc + simulation.particle_positions[(c, p)]
        }) / simulation.particle_counts[c] as f32;
        gc_sum += gc;
        gc
    });
    gc_sum / CLASS_COUNT as f32
}