use std::usize;

use crate::mat::Mat2D;

fn leakyrelu(x: f64) -> f64 {
    if x < 0. {
        0.01 * x
    } else {
        x
    }
}
fn dleakyrelu(x: f64) -> f64 {
    if x < 0. {
        0.01
    } else {
        1.
    }
}

pub struct Network {
    layers: [Layer; 3],
}

impl Network {
    pub fn random(layer_sizes: [usize; 4]) -> Self {
        Network {
            layers: [
                Layer::random(layer_sizes[0], layer_sizes[1]),
                Layer::random(layer_sizes[1], layer_sizes[2]),
                Layer::random(layer_sizes[2], layer_sizes[3]),
            ],
        }
    }

    fn calc_activations_and_preactivations(&self, input: Mat2D<f64>) -> [Mat2D<f64>; 6] {
        assert_eq!(
            input.num_rows(),
            self.layers[0].input_size,
            "incorrect input size"
        );

        let z0 = &self.layers[0].weights * &input + &self.layers[0].biases;
        let a0 = z0.map(leakyrelu);
        let z1 = &self.layers[1].weights * &a0 + &self.layers[1].biases;
        let a1 = z1.map(leakyrelu);
        let z2 = &self.layers[2].weights * &a1 + &self.layers[2].biases;
        let a2 = z2.map(leakyrelu);

        [z0, a0, z1, a1, z2, a2]
    }

    fn calc_activations(&self, input: Mat2D<f64>) -> [Mat2D<f64>; 3] {
        let [_, a0, _, a1, _, a2] = self.calc_activations_and_preactivations(input);
        [a0, a1, a2]
    }

    pub fn infer(&self, input: Mat2D<f64>) -> Mat2D<f64> {
        let [_, _, output] = self.calc_activations(input);
        output
    }

    pub fn calc_gradients(
        &mut self,
        input: Mat2D<f64>,
        expected_output: Mat2D<f64>,
    ) -> [[Mat2D<f64>; 3]; 2] {
        assert_eq!(
            input.num_rows(),
            self.layers[0].input_size,
            "incorrect input size"
        );
        assert_eq!(
            expected_output.num_rows(),
            self.layers[2].output_size,
            "incorrect output size"
        );

        let [z0, a0, z1, a1, z2, a2] = self.calc_activations_and_preactivations(input.to_owned());

        let cost = (&a2 - &expected_output)
            .map(|x| x * x / 2.)
            .vec()
            .iter()
            .fold(0., |acc, x| acc + x);

        println!("cost: {}", cost);

        let mut weights_gradients = [
            self.layers[0].weights.map(|_| 0.),
            self.layers[1].weights.map(|_| 0.),
            self.layers[2].weights.map(|_| 0.),
        ];
        let mut biases_gradients = [
            self.layers[0].biases.map(|_| 0.),
            self.layers[1].biases.map(|_| 0.),
            self.layers[2].biases.map(|_| 0.),
        ];

        // output layer (3rd layer)

        for i in 0..self.layers[2].output_size {
            for j in 0..self.layers[2].input_size {
                weights_gradients[2][(i, j)] +=
                    a1[(j, 0)] * dleakyrelu(z2[(i, 0)]) * (a2[(i, 0)] - expected_output[(i, 0)]);
            }
        }
        for i in 0..self.layers[2].output_size {
            biases_gradients[2][(i, 0)] +=
                dleakyrelu(z2[(i, 0)]) * (a2[(i, 0)] - expected_output[(i, 0)]);
        }

        // 2nd layer

        for i in 0..self.layers[1].output_size {
            for j in 0..self.layers[1].input_size {
                weights_gradients[1][(i, j)] +=
                    a1[(j, 0)] * dleakyrelu(z2[(i, 0)]) * (a2[(i, 0)] - expected_output[(i, 0)]);
            }
        }
        for i in 0..self.layers[1].output_size {
            biases_gradients[1][(i, 0)] +=
                dleakyrelu(z2[(i, 0)]) * (a2[(i, 0)] - expected_output[(i, 0)]);
        }

        [weights_gradients, biases_gradients]
    }

    pub fn apply_gradients(&mut self, gradients: [[Mat2D<f64>; 3]; 2]) {
        const LEARNING_RATE: f64 = 0.01;
        let [weights_gradients, biases_gradients] = gradients;

        self.layers[0].weights =
            &self.layers[0].weights - weights_gradients[0].map(|x| x * LEARNING_RATE);
        self.layers[1].weights =
            &self.layers[1].weights - weights_gradients[1].map(|x| x * LEARNING_RATE);
        self.layers[2].weights =
            &self.layers[2].weights - weights_gradients[2].map(|x| x * LEARNING_RATE);

        self.layers[0].biases =
            &self.layers[0].biases - biases_gradients[0].map(|x| x * LEARNING_RATE);
        self.layers[1].biases =
            &self.layers[1].biases - biases_gradients[1].map(|x| x * LEARNING_RATE);
        self.layers[2].biases =
            &self.layers[2].biases - biases_gradients[2].map(|x| x * LEARNING_RATE);
    }
}

#[derive(Debug)]
struct Layer {
    input_size: usize,
    output_size: usize,
    weights: Mat2D<f64>,
    biases: Mat2D<f64>,
}

impl Layer {
    pub fn random(input_size: usize, output_size: usize) -> Self {
        let max = 1. / (input_size as f64).sqrt();
        let weights = Mat2D::random((-max, max), output_size, input_size);
        let biases = Mat2D::filled_with(0.01, output_size, 1);
        Layer {
            input_size,
            output_size,
            weights,
            biases,
        }
    }
}
