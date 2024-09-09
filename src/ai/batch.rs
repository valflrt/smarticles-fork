use core::f32;
use std::{fs::OpenOptions, io::Write};

use postcard::to_allocvec;
use rand::{distributions::WeightedIndex, prelude::Distribution};
use serde::{Deserialize, Serialize};

use super::net::Network;

#[derive(Debug, Serialize, Deserialize)]
pub struct Batch {
    pub networks: Vec<Network>,
    pub generation: usize,
}

impl Batch {
    pub fn new<N>(networks: N) -> Self
    where
        N: Into<Vec<Network>>,
    {
        Self {
            networks: networks.into(),
            generation: 0,
        }
    }

    /// Classe les réseaux par score décroissant
    pub fn rank<T, F>(&self, evaluation_data: Vec<T>, compute_score: F) -> Vec<(usize, f32)>
    where
        T: Clone,
        F: Fn(T) -> f32,
    {
        let mut ranking = evaluation_data
            .iter()
            .enumerate()
            .map(|(i, evaluation_data)| (i, compute_score(evaluation_data.to_owned())))
            .collect::<Vec<_>>();
        ranking.sort_by(|(_, score_1), (_, score_2)| score_2.total_cmp(score_1));
        ranking
    }

    /// Evolution des réseau à partir d'un classement: selection
    /// et mutation
    pub fn evolve(&mut self, mutation_rate: f32, ranking: Vec<(usize, f32)>) {
        assert!(
            (0. ..=1.).contains(&mutation_rate),
            "mutation rate must be between 0 and 1"
        );

        let mut new_networks = Vec::with_capacity(self.networks.len());

        let mut rng = rand::thread_rng();

        let l = ranking.len();
        let scores = ranking
            .iter()
            .take(l / 2 + 1)
            .map(|(_, score)| score)
            .copied();

        let min = scores.clone().fold(f32::INFINITY, f32::min);
        let max = scores.fold(f32::NEG_INFINITY, f32::max);

        let weights = (0..l)
            .map(|i| {
                if i < l / 2 {
                    let (_, score) = ranking[i];
                    // for the first half of the ranking, assign the scores of
                    // the networks mapped between 0 and 1 as the weight
                    (score - min) / (max - min)
                } else if i < 3 * l / 4 {
                    let (_, score) = ranking[l / 2 - 1];
                    // for the third quarter of the ranking, assign a weight equal
                    // to half the weight of the last network in the first half
                    // of the ranking
                    0.5 * (score - min) / (max - min)
                } else {
                    // for the last quarter assign a weight of 0
                    0.
                }
            })
            .collect::<Vec<_>>();
        let dist = WeightedIndex::new(weights.to_owned()).unwrap();

        while new_networks.len() < l {
            let (i1, i2) = (dist.sample(&mut rng), dist.sample(&mut rng));

            let mut new_network = if i1 != i2 {
                let (network1, network2) = (
                    self.networks[ranking[i1].0].clone(),
                    self.networks[ranking[i2].0].clone(),
                );

                network1.crossover(network2, 0.99 * weights[i1] / (weights[i1] + weights[i2]))
            } else {
                let mut network = self.networks[ranking[i1].0].clone();
                network.mutate(0.1);
                network
            };

            let l = l as f32;
            let k = ((i1 + i2) as f32) / 2. + 1.;
            new_network.mutate(
                mutation_rate
                    // mutate good networks less than bad networks:
                    // first has a mutation rate of 0.5 and last of 1.
                    * (2. * k + l)
                    / (3. * l),
                // reduce mutation rate when reaching higher generation counts
                // / ((self.generation as f32) / (1000. * k)).exp(),
            );

            new_networks.push(new_network);
        }

        self.networks = new_networks;

        self.generation += 1;
    }

    pub fn save(&self) {
        let encoded = to_allocvec::<Batch>(self).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(format!("./batches/batch_gen_{}", self.generation))
            .unwrap();
        file.write_all(&encoded).unwrap();
    }
}
