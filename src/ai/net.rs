use rand::distributions::Uniform;

use crate::mat::Mat2D;

#[derive(Debug, Clone)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new<L>(layers: L) -> Self
    where
        L: Into<Vec<Layer>>,
    {
        Self {
            layers: layers.into(),
        }
    }

    pub fn infer(&self, input: Mat2D<f32>) -> Mat2D<f32> {
        assert_eq!(
            input.num_rows(),
            self.layers[0].input_size,
            "incorrect input size"
        );

        (0..self.layers.len()).fold(input, |layer_input, i| {
            (&self.layers[i].weights * layer_input + &self.layers[i].biases)
                .map(|x| self.layers[i].activation_fn.apply(x))
        })
    }

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
                    Uniform::new(-mutation_rate, mutation_rate),
                    self.layers[i].output_size,
                    1,
                );
        }
    }
}

#[derive(Debug, Clone)]
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
        let biases = Mat2D::filled_with(0.01, output_size, 1);
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
#[derive(Debug, Clone, Copy)]
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
