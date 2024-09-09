use core::f32;
use std::{sync::mpsc::Receiver, time::Instant};

use humantime::format_duration;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    ai::{
        batch::Batch,
        training::{compute_score, evaluation_fn},
    },
    Senders, SmarticlesEvent,
};

pub struct TrainingManager {
    batch: Batch,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,
}

impl TrainingManager {
    pub fn new(batch: Batch, senders: Senders, receiver: Receiver<SmarticlesEvent>) -> Self {
        senders.send_to_app(SmarticlesEvent::NetworkRanking(
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

            let start_timestamp = Instant::now();

            let evaluation_data = self
                .batch
                .networks
                .par_iter()
                .cloned()
                .map(evaluation_fn)
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
                    "{:2} -> position: {:.3}, score: {:>20.3}",
                    i,
                    (score - min) / (max - min),
                    score,
                );
            }

            println!(
                "evaluated networks in {}\nevolved networks in {}\ntotal elapsed: {}\n",
                format_duration(evaluation_duration),
                format_duration(evolve_duration),
                format_duration(evaluation_duration + evolve_duration)
            );

            if self.batch.generation % 25 == 0 {
                self.batch.save();
            }

            self.senders
                .send_to_app(SmarticlesEvent::GenerationChange(self.batch.generation));

            if n + 1 == gen_count {
                self.senders.send_to_app(SmarticlesEvent::NetworkRanking(
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
                    let evaluation_data = self
                        .batch
                        .networks
                        .par_iter()
                        .cloned()
                        .map(evaluation_fn)
                        .collect::<Vec<_>>();

                    self.senders.send_to_app(SmarticlesEvent::NetworkRanking(
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
