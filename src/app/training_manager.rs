use std::{
    sync::mpsc::Receiver,
    thread::sleep,
    time::{Duration, Instant},
};

use humantime::format_duration;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    ai::{
        batch::Batch,
        training::{compute_score, evaluation_fn},
    },
    events::{Recipient, StateUpdate},
    Event, Senders,
};

pub struct TrainingManager {
    batch: Batch,

    senders: Senders,
    receiver: Receiver<Event>,
}

impl TrainingManager {
    pub fn start(batch: Batch, senders: Senders, receiver: Receiver<Event>) {
        senders.send(
            Recipient::App,
            Event::StateUpdate(
                StateUpdate::new().network_ranking(
                    &batch
                        .networks
                        .iter()
                        .map(|network| (0., network.to_owned()))
                        .collect(),
                ),
            ),
        );

        let mut slf = TrainingManager {
            batch,
            senders,
            receiver,
        };

        sleep(Duration::from_millis(500));

        while slf.update() {}
    }

    pub fn update(&mut self) -> bool {
        let events = self.receiver.try_iter().collect::<Vec<_>>();

        for ev in events {
            match ev {
                Event::StartTraining(gen_count) => {
                    self.train(gen_count);
                    self.senders.send(Recipient::App, Event::TrainingStopped);
                }

                Event::EvaluateNetworks => {
                    let evaluation_data = self
                        .batch
                        .networks
                        .par_iter()
                        .cloned()
                        .map(evaluation_fn)
                        .collect::<Vec<_>>();

                    self.senders.send(
                        Recipient::App,
                        Event::StateUpdate(
                            StateUpdate::new().network_ranking(
                                &self
                                    .batch
                                    .rank(evaluation_data, compute_score)
                                    .iter()
                                    .copied()
                                    .map(|(i, score)| (score, self.batch.networks[i].to_owned()))
                                    .collect(),
                            ),
                        ),
                    );

                    self.senders.send(Recipient::App, Event::TrainingStopped);
                }

                Event::Exit => return false,

                _ => {}
            }
        }

        true
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
            self.batch.evolve(ranking.to_owned(), 2.);
            let evolve_duration = start_timestamp.elapsed();

            for (k, (i, score)) in ranking.iter().copied().enumerate() {
                if (10..ranking.len().checked_sub(10).unwrap_or(10)).contains(&k) {
                    if k == 10 {
                        println!("..");
                    }
                    continue;
                }
                println!("{:2} -> position: {:2}, score: {:>20.3}", i, k, score,);
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

            self.senders.send(
                Recipient::App,
                Event::StateUpdate(StateUpdate::new().training_generation(self.batch.generation)),
            );

            if n + 1 == gen_count {
                self.senders.send(
                    Recipient::App,
                    Event::StateUpdate(
                        StateUpdate::new().network_ranking(
                            &ranking
                                .iter()
                                .copied()
                                .map(|(i, score)| (score, self.batch.networks[i].to_owned()))
                                .collect(),
                        ),
                    ),
                );
            };
        }
    }
}
