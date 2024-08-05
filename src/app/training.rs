use std::{sync::mpsc::Receiver, time::Instant};

use egui::Vec2;
use humantime::format_duration;
use rand::random;

use crate::{
    ai::{batch_training::Batch, net::Network},
    mat::Mat2D,
    simulation::Simulation,
    Senders, SmarticlesEvent, CLASS_COUNT, MAX_FORCE,
};

pub const NETWORK_INPUT_SIZE: usize = 2 + CLASS_COUNT * 2 + CLASS_COUNT * CLASS_COUNT + CLASS_COUNT;
pub const NETWORK_OUTPUT_SIZE: usize = CLASS_COUNT * CLASS_COUNT;

pub const INFERENCE_TICK_INTERVAL: u8 = 10;

fn evaluation_fn(network: Network) -> f32 {
    let mut score = 0.;

    let mut sim = Simulation::default();
    setup_simulation_for_network(&mut sim);
    sim.spawn();

    let target_position =
        Vec2::new((random::<f32>() * 2.) - 1., (random::<f32>() * 2.) - 1.) * 5000.;

    let (_, ggc, _, _) = calc_geometric_centers_and_mean_distances(&sim);

    const MAX_TICK_COUNT: usize = 4000;

    let mut prev_ggc = ggc;

    for _tick_count in 0..MAX_TICK_COUNT / INFERENCE_TICK_INTERVAL as usize {
        for _ in 0..INFERENCE_TICK_INTERVAL {
            sim.move_particles();
        }

        let (gcs, ggc, mean_distances, global_mean_distance) =
            calc_geometric_centers_and_mean_distances(&sim);

        let movement = ggc - prev_ggc;
        let target_distance = target_position - ggc;

        score += movement.dot(target_distance);
        score -= global_mean_distance.powi(3);

        prev_ggc = ggc;

        if target_distance.length() < 10. {
            println!("target reached");
            score += 10000.;
        }

        let mut output = network.infer(adapt_input(
            target_distance.normalized(),
            gcs,
            mean_distances,
            sim.force_matrix.to_owned(),
        ));

        output.vec_mut().iter_mut().for_each(|x| *x *= MAX_FORCE);

        *sim.force_matrix.vec_mut() = output.vec();
    }

    score /= MAX_TICK_COUNT as f32;

    // score += traveled_distance_to_target;
    // score -= global_mean_distance_sum / (MAX_TICK_COUNT as f32);

    score
}

pub fn adapt_input(
    normalized_ggc_to_target_direction: Vec2,
    gcs: Vec<Vec2>,
    mean_distances: Vec<f32>,
    force_matrix: Mat2D<f32>,
) -> Mat2D<f32> {
    Mat2D::from_rows(
        [
            &[
                normalized_ggc_to_target_direction.x,
                normalized_ggc_to_target_direction.y,
            ],
            &gcs.iter()
                .zip(mean_distances.iter().copied())
                .flat_map(|(gc, mean_distance)| vec![gc.x, gc.y, mean_distance])
                .collect::<Vec<_>>()[..],
            &force_matrix.vec()[..],
        ]
        .concat(),
        NETWORK_INPUT_SIZE,
        1,
    )
}

pub struct TrainingBackend {
    batch: Batch,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,
}

impl TrainingBackend {
    pub fn new(batch: Batch, senders: Senders, receiver: Receiver<SmarticlesEvent>) -> Self {
        senders.send_ui(SmarticlesEvent::Networks(batch.networks.to_owned()));

        Self {
            batch,
            senders,
            receiver,
        }
    }

    pub fn train(&mut self, gen_count: usize) {
        for _n in 0..gen_count {
            println!("generation {}\n", self.batch.generation_count);
            let start_timestamp = Instant::now();

            let ranking = self.batch.rank(evaluation_fn);
            for (i, score) in &ranking {
                println!("{} -> {}", i, score);
            }
            self.batch.evolve(1., ranking);

            println!("{} elapsed\n", format_duration(start_timestamp.elapsed()))
        }

        self.senders
            .send_ui(SmarticlesEvent::Networks(self.batch.networks.to_owned()));
    }

    pub fn update(&mut self) -> bool {
        let events = self.receiver.try_iter().collect::<Vec<_>>();

        for event in events {
            match event {
                SmarticlesEvent::StartTraining(gen_count) => {
                    self.train(gen_count);
                }

                SmarticlesEvent::Quit => return false,

                _ => {}
            }
        }

        true
    }
}

pub fn setup_simulation_for_network(sim: &mut Simulation) {
    sim.particle_counts = [12; CLASS_COUNT];
}

/// Returns the geometric centers of each class, the global
/// geometric center, the mean distances between the geometric
/// center each particle of each class and the mean distance
/// between each particle and the global geomtric center:
/// `(gcs, ggc, mean_distances, global_mean_distance)`
pub fn calc_geometric_centers_and_mean_distances(
    simulation: &Simulation,
) -> (Vec<Vec2>, Vec2, Vec<f32>, f32) {
    let mut gc_sum = Vec2::ZERO;
    let gcs = (0..CLASS_COUNT)
        .map(|c| {
            let gc = (0..simulation.particle_counts[c]).fold(Vec2::ZERO, |acc, p| {
                acc + simulation.particle_positions[(c, p)]
            }) / simulation.particle_counts[c] as f32;
            gc_sum += gc;
            gc
        })
        .collect::<Vec<_>>();
    let mut mean_distances_sum = 0.;
    let mean_distances = (0..CLASS_COUNT)
        .map(|c| {
            let mean_distance = (0..simulation.particle_counts[c]).fold(0., |acc, p| {
                acc + (gcs[c] - simulation.particle_positions[(c, p)]).length()
            }) / simulation.particle_counts[c] as f32;
            mean_distances_sum += mean_distance;
            mean_distance
        })
        .collect();
    (
        gcs,
        gc_sum / CLASS_COUNT as f32,
        mean_distances,
        mean_distances_sum / CLASS_COUNT as f32,
    )
}
