use rand::{distributions::Uniform, Rng};
use serde::{Deserialize, Serialize};

use crate::mat::Mat2D;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// Crée un nouveau réseau à partir de couches
    pub fn new<L>(layers: L) -> Self
    where
        L: Into<Vec<Layer>>,
    {
        let layers: Vec<Layer> = layers.into();

        // Vérification de la validité des tailles des couches
        for window in layers.windows(2) {
            let l1 = &window[0];
            let l2 = &window[1];
            assert_eq!(
                l1.output_size, l2.input_size,
                "input size and output size of consecutive layers must match"
            );
        }

        Self { layers }
    }

    /// Calcule l'image d'une valeur d'entrée par le réseau
    pub fn infer(&self, input: Vec<f32>) -> Vec<f32> {
        let input_size = input.len();
        assert_eq!(
            input_size, self.layers[0].input_size,
            "incorrect input size"
        );

        let input = Mat2D::from_rows(input, input_size, 1);
        (0..self.layers.len())
            .fold(input, |layer_input, i| {
                (&self.layers[i].weights * layer_input + &self.layers[i].biases)
                    .map(|x| self.layers[i].activation_fn.apply(x))
            })
            .vec()
    }

    /// Mute ce réseau en modifiant aléatoirement les poids et
    /// les biais
    pub fn mutate(&mut self, mutation_rate: f32) {
        assert!(
            (0. ..=1.).contains(&mutation_rate),
            "mutation rate must be between 0 and 1"
        );

        for i in 0..self.layers.len() {
            let max = mutation_rate / (self.layers[i].input_size as f32).sqrt();
            self.layers[i].weights = &self.layers[i].weights
                + Mat2D::random(
                    Uniform::new(-max, max),
                    self.layers[i].output_size,
                    self.layers[i].input_size,
                );
            self.layers[i].biases = &self.layers[i].biases
                + Mat2D::random(
                    Uniform::new(-mutation_rate * 0.2, mutation_rate * 0.2),
                    self.layers[i].output_size,
                    1,
                );
        }
    }

    /// Calcule le réseau dont les poids et les biais sont la
    /// moyenne des poids et des biais deux ce réseau et d'un
    /// autre
    pub fn average(self, other: Self) -> Self {
        let mut out = self.to_owned();
        for i in 0..self.layers.len() {
            out.layers[i].weights =
                (&self.layers[i].weights + &other.layers[i].weights).map(|x| x / 2.);
            out.layers[i].biases =
                (&self.layers[i].biases + &other.layers[i].biases).map(|x| x / 2.);
        }
        out
    }

    /// Combine the current network with another by randomly
    /// choosing single biases and weights from their two parents.
    ///
    /// `selection_probability` is between 0 and 1 and represents
    /// the probability of the weight/bias the current network being
    /// chosen.
    pub fn crossover(self, other: Self, selection_probability: f32) -> Self {
        let selection_probability = selection_probability as f64;
        let mut rng = rand::thread_rng();

        let mut out = self.to_owned();
        for i in 0..self.layers.len() {
            for (row, column) in self.layers[i].weights.enumerate() {
                out.layers[i].weights[(row, column)] = if rng.gen_bool(selection_probability) {
                    self.layers[i].weights[(row, column)]
                } else {
                    other.layers[i].weights[(row, column)]
                };
                out.layers[i].biases[(row, 0)] = if rng.gen_bool(selection_probability) {
                    self.layers[i].biases[(row, 0)]
                } else {
                    other.layers[i].biases[(row, 0)]
                };
            }
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Mat2D<f32>,
    pub biases: Mat2D<f32>,
    pub activation_fn: ActivationFn,
}

impl Layer {
    pub fn random(input_size: usize, output_size: usize, activation_fn: ActivationFn) -> Self {
        let max = 1. / (input_size as f32).sqrt();
        let weights = Mat2D::random(Uniform::new(-max, max), output_size, input_size);
        let biases = Mat2D::random(Uniform::new(-0.2, 0.2), output_size, 1);
        Layer {
            input_size,
            output_size,
            weights,
            biases,
            activation_fn,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFn {
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
}

impl ActivationFn {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFn::Relu => x.max(0.),
            ActivationFn::LeakyRelu => {
                if x < 0. {
                    0.01 * x
                } else {
                    x
                }
            }
            ActivationFn::Sigmoid => 1. / (1. + (-x).exp()),
            ActivationFn::Tanh => x.tanh(),
        }
    }
}
