use core::f32;
use std::{f32::consts::TAU, sync::mpsc::Receiver, time::Instant};

use egui::Vec2;
use humantime::format_duration;
use rand::random;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    ai::{batch_training::Batch, net::Network},
    mat::Mat2D,
    simulation::Simulation,
    Senders, SmarticlesEvent, CLASS_COUNT, MAX_FORCE,
};

pub const BATCH_SIZE: usize = 30;

pub const NETWORK_INPUT_SIZE: usize = 1 + CLASS_COUNT * 2 + CLASS_COUNT * CLASS_COUNT;
pub const NETWORK_OUTPUT_SIZE: usize = CLASS_COUNT * CLASS_COUNT;

pub const INFERENCE_TICK_INTERVAL: u8 = 20;

pub fn random_target_position(origin: Vec2) -> Vec2 {
    origin + 600. * Vec2::angled(TAU * random::<f32>())
}

pub fn setup_simulation_for_network(sim: &mut Simulation) {
    sim.particle_counts = [12; CLASS_COUNT];
}

#[derive(Debug, Clone)]
struct Context {
    sim: Simulation,
}

fn setup_training_context() -> Context {
    let mut sim = Simulation::default();
    setup_simulation_for_network(&mut sim);
    sim.spawn();
    Context { sim }
}

pub fn adapt_input(
    normalized_ggc_to_target_direction: Vec2,
    gcs: Vec<Vec2>,
    force_matrix: Mat2D<f32>,
) -> Vec<f32> {
    [
        &[normalized_ggc_to_target_direction.angle()],
        &gcs.iter()
            .flat_map(|gc| vec![gc.x, gc.y])
            .collect::<Vec<_>>()[..],
        &force_matrix.vec()[..],
    ]
    .concat()
}

#[derive(Debug, Clone)]
struct EvaluationData {
    total_movement: Vec2,
    /// the smaller this is the more the movement is a straight
    /// line
    disorder_indicator: f32,
    movement_projection_on_target_direction: f32,
    mean_distance_between_particles: f32,
    mean_distance_to_target: f32,
    ticks_to_target: Vec<usize>,
    target_hits: usize,
}

const MAX_TICK_COUNT: usize = 5000;
fn evaluation_fn(network: Network, context: Context) -> EvaluationData {
    let mut evaluation_data = EvaluationData {
        total_movement: Vec2::ZERO,
        disorder_indicator: 0.,
        movement_projection_on_target_direction: 0.,
        mean_distance_between_particles: 0.,
        mean_distance_to_target: 0.,
        ticks_to_target: vec![0],
        target_hits: 0,
    };

    let Context { mut sim } = context;
    let mut target_position = random_target_position(Vec2::ZERO);

    let (_, ggc) = calc_geometric_centers_and_mean_distances(&sim);

    let mut prev_ggc = ggc;

    let mut movements = Vec::new();

    for _tick_count in 0..MAX_TICK_COUNT / INFERENCE_TICK_INTERVAL as usize {
        for _ in 0..INFERENCE_TICK_INTERVAL {
            sim.move_particles();
            *evaluation_data.ticks_to_target.last_mut().unwrap() += 1;
        }

        let (gcs, ggc) = calc_geometric_centers_and_mean_distances(&sim);

        let target_distance = target_position - ggc;

        let movement = ggc - prev_ggc;
        evaluation_data.total_movement += movement;
        movements.push(movement);

        let mut mean_distance_between_particles = 0.;
        for c1 in 0..CLASS_COUNT {
            for p1 in 0..sim.particle_counts[c1] {
                for c2 in 0..CLASS_COUNT {
                    for p2 in 0..sim.particle_counts[c2] {
                        mean_distance_between_particles += (sim.particle_positions[(c1, p1)]
                            - sim.particle_positions[(c2, p2)])
                            .length();
                    }
                }
            }
        }
        evaluation_data.mean_distance_between_particles += mean_distance_between_particles
            / (sim.particle_counts.iter().sum::<usize>()).pow(2) as f32;

        evaluation_data.movement_projection_on_target_direction +=
            movement.dot(target_distance.normalized());

        evaluation_data.mean_distance_to_target += sim
            .particle_positions
            .vec()
            .iter()
            .copied()
            .map(|pos| (pos - target_position).length())
            .sum::<f32>()
            / (MAX_TICK_COUNT * sim.particle_counts.iter().sum::<usize>()) as f32;

        prev_ggc = ggc;

        let mut target_reached = false;
        for c in 0..CLASS_COUNT {
            for p in 0..sim.particle_counts[c] {
                target_reached &= (target_position - sim.particle_positions[(c, p)]).length() < 30.;
            }
        }
        if target_reached {
            evaluation_data.target_hits += 1;
            evaluation_data.ticks_to_target.push(0);
            target_position = random_target_position(ggc);
        }

        let mut output = network.infer(adapt_input(
            target_distance.normalized(),
            gcs,
            sim.force_matrix.to_owned(),
        ));

        output.iter_mut().for_each(|x| *x *= MAX_FORCE);
        *sim.force_matrix.vec_mut() = output;
    }

    for movement in movements {
        evaluation_data.disorder_indicator +=
            (movement.angle() - evaluation_data.total_movement.angle()).abs();
    }

    evaluation_data
}

fn compute_score(evaluation_data: EvaluationData) -> f32 {
    // println!("{:#?}", evaluation_data);
    let EvaluationData {
        total_movement,
        disorder_indicator,
        movement_projection_on_target_direction,
        mean_distance_between_particles,
        mean_distance_to_target,
        ticks_to_target,
        target_hits,
        ..
    } = evaluation_data;

    total_movement.length()
        + (target_hits as f32) * 1000.
        + movement_projection_on_target_direction * 50.
        - mean_distance_between_particles / 100.
        - disorder_indicator / 2.
        + 100. / mean_distance_to_target
        + (MAX_TICK_COUNT as f32
            - ticks_to_target.iter().sum::<usize>() as f32 / ticks_to_target.len() as f32)
            / 100.
}

pub struct TrainingBackend {
    batch: Batch,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,
}

impl TrainingBackend {
    pub fn new(batch: Batch, senders: Senders, receiver: Receiver<SmarticlesEvent>) -> Self {
        senders.send_ui(SmarticlesEvent::NetworkRanking(
            batch
                .networks
                .iter()
                .map(|network| (0., network.to_owned()))
                .collect(),
        ));

        Self {
            batch,
            senders,
            receiver,
        }
    }

    pub fn train(&mut self, gen_count: usize) {
        for n in 0..gen_count {
            println!("generation {}\n", self.batch.generation + 1);

            let context = setup_training_context();

            let start_timestamp = Instant::now();

            let evaluation_data = self
                .batch
                .networks
                .par_iter()
                .map(|network| evaluation_fn(network.to_owned(), context.to_owned()))
                .collect::<Vec<_>>();

            let evaluation_duration = start_timestamp.elapsed();

            let ranking = self.batch.rank(evaluation_data.to_owned(), compute_score);

            let start_timestamp = Instant::now();

            self.batch.evolve(1., ranking.to_owned());

            let evolve_duration = start_timestamp.elapsed();

            let min = ranking
                .iter()
                .map(|(_, score)| score)
                .copied()
                .fold(f32::INFINITY, f32::min);
            let max = ranking
                .iter()
                .map(|(_, score)| score)
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            for (k, (i, score)) in ranking.iter().copied().enumerate() {
                if (10..ranking.len().checked_sub(10).unwrap_or(10)).contains(&k) {
                    if k == 10 {
                        println!("..");
                    }
                    continue;
                }
                println!(
                    "{:2} -> position: {:.3}, score: {:10.2}, target hits: {:2}",
                    i,
                    (score - min) / (max - min),
                    score,
                    evaluation_data[i].target_hits
                );
            }

            println!(
                "evaluated networks in {}\nevolved networks in {}\ntotal elapsed: {}\n",
                format_duration(evaluation_duration),
                format_duration(evolve_duration),
                format_duration(evaluation_duration + evolve_duration)
            );

            if (n + 1) % 10 == 0 {
                self.batch.save();
            }

            self.senders
                .send_ui(SmarticlesEvent::GenerationChange(self.batch.generation));
            if n + 1 == gen_count {
                self.senders.send_ui(SmarticlesEvent::NetworkRanking(
                    ranking
                        .iter()
                        .copied()
                        .map(|(i, score)| (score, self.batch.networks[i].to_owned()))
                        .collect(),
                ));
            };
        }
    }

    pub fn update(&mut self) -> bool {
        let events = self.receiver.try_iter().collect::<Vec<_>>();

        for event in events {
            match event {
                SmarticlesEvent::StartTraining(gen_count) => {
                    self.train(gen_count);
                }

                SmarticlesEvent::EvaluateNetworks => {
                    let context = setup_training_context();
                    let evaluation_data = self
                        .batch
                        .networks
                        .par_iter()
                        .map(|network| evaluation_fn(network.to_owned(), context.to_owned()))
                        .collect::<Vec<_>>();
                    self.senders.send_ui(SmarticlesEvent::NetworkRanking(
                        self.batch
                            .rank(evaluation_data, compute_score)
                            .iter()
                            .copied()
                            .map(|(i, score)| (score, self.batch.networks[i].to_owned()))
                            .collect(),
                    ))
                }

                SmarticlesEvent::Quit => return false,

                _ => {}
            }
        }

        true
    }
}

/// Returns the geometric centers of each class, the global
/// geometric center, the mean distances between the geometric
/// center each particle of each class and the mean distance
/// between each particle and the global geomtric center:
/// `(gcs, ggc, mean_distances, global_mean_distance)`
pub fn calc_geometric_centers_and_mean_distances(simulation: &Simulation) -> (Vec<Vec2>, Vec2) {
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

    (gcs, gc_sum / CLASS_COUNT as f32)
}
