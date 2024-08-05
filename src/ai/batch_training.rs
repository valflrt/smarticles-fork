use rand::{distributions::WeightedIndex, prelude::Distribution};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::net::Network;

#[derive(Debug)]
pub struct Batch {
    pub networks: Vec<Network>,
    pub generation_count: usize,
}

impl Batch {
    pub fn new<N>(networks: N) -> Self
    where
        N: Into<Vec<Network>>,
    {
        Self {
            networks: networks.into(),
            generation_count: 0,
        }
    }

    pub fn rank<F>(&self, evaluation_fn: F) -> Vec<(usize, f32)>
    where
        F: Fn(Network) -> f32 + Send + Sync,
    {
        let mut ranking = (0..self.networks.len())
            .into_par_iter()
            .map(|i| (i, evaluation_fn(self.networks[i].to_owned())))
            .collect::<Vec<_>>();
        ranking.sort_by(|(_, score_1), (_, score_2)| score_2.total_cmp(score_1));
        ranking
    }

    /// Evaluates the networks of the batch based on the
    /// `evaluation_fn` which returns a float used to rank networks.
    /// After evaluation, the top 10% is then mutated to evolve
    /// the batch.
    pub fn evolve(&mut self, mutation_rate: f32, ranking: Vec<(usize, f32)>) {
        assert!(
            (0. ..=1.).contains(&mutation_rate),
            "mutation rate must be between 0 and 1"
        );

        let mut new_networks = Vec::with_capacity(self.networks.len());

        let mut rng = rand::thread_rng();
        let weights: Vec<_> = (0..ranking.len())
            .map(|i| {
                let l = ranking.len();
                if i < l / 2 {
                    l - i
                } else if i < 3 * l / 4 {
                    1
                } else {
                    0
                }
            })
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();

        while new_networks.len() < self.networks.len() {
            let i = ranking[dist.sample(&mut rng)].0;
            let mut network = self.networks[i].to_owned();
            network.mutate(mutation_rate);
            new_networks.push(network);
        }

        self.networks = new_networks;

        self.generation_count += 1;
    }
}
