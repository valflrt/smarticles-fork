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

    /// Calcul des scores des réseaux et tri par score décroissant
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

        // Tri du classement par score décroissant
        ranking.sort_by(|(_, score_1), (_, score_2)| score_2.total_cmp(score_1));
        ranking
    }

    /// Évolution des réseau à partir d'un classement: selection
    /// et mutation
    pub fn evolve(&mut self, ranking: Vec<(usize, f32)>, mutation_rate: f32) {
        let mut new_networks = Vec::with_capacity(self.networks.len());

        let mut rng = rand::thread_rng();

        let l = self.networks.len(); // Nombre de réseaux

        let scores = ranking.iter().map(|(_, score)| score).copied();

        // Calcul du score minimal et maximal
        let min = scores.clone().fold(f32::INFINITY, f32::min);
        let max = scores.fold(f32::NEG_INFINITY, f32::max);

        let weights = (0..l)
            .map(|i| {
                if i < l / 2 {
                    let (_, score) = ranking[i];
                    // première moitié: les poids sont obtenus à partir
                    // des scores des réseaux, en les ramenant entre 0 et 1.
                    (score - min) / (max - min)
                } else if i < 3 * l / 4 {
                    let (_, score) = ranking[l / 2 - 1];
                    // troisième quart: même chose que précédemment mais le score
                    // est cette fois ci divisé par 2.
                    0.5 * (score - min) / (max - min)
                } else {
                    // les réseaux du dernier quart ne peuvent pas être choisis.
                    0.
                }
            })
            .collect::<Vec<_>>();
        let dist = WeightedIndex::new(weights.to_owned()).unwrap();

        // Remplissage de la batch en choisissant successivement des
        // réseaux aléatoirement à partir des poids établis ci-dessus.
        while new_networks.len() < l {
            // On choisit deux réseaux au hasard
            let (i1, i2) = (dist.sample(&mut rng), dist.sample(&mut rng));

            let mut new_network = if i1 != i2 {
                let (network1, network2) = (
                    self.networks[ranking[i1].0].clone(),
                    self.networks[ranking[i2].0].clone(),
                );

                // Si ce ne sont pas les mêmes réseaux, on les croise pour en
                // produire un nouveau
                network1.crossover(network2, 0.99 * weights[i1] / (weights[i1] + weights[i2]))
            } else {
                let mut network = self.networks[ranking[i1].0].clone();
                // Si les deux réseaux choisis sont les mêmes, on réalise une
                // mutation de ce réseau
                network.mutate(0.5 * mutation_rate);
                network
            };

            // Le nouveau réseau subit dans tous les cas une mutation
            new_network.mutate(
                // les meilleurs réseaux sont moins mutés que les moins bons
                mutation_rate * (i1 + i2 + l) as f32 / (3 * l) as f32,
            );

            // Ajout du nouveau réseau à la batch
            new_networks.push(new_network);
        }

        self.networks = new_networks;

        self.generation += 1;
    }

    /// Enregistre une batch sous forme d'un fichier pour en
    /// garder une trace
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
