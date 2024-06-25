use std::usize;

use rand::{distributions::Open01, Rng};

use crate::mat::{Mat1D, Mat2D};

fn relu(x: f64) -> f64 {
    x.max(0.)
}
fn drelu(x: f64) -> f64 {
    if x < 0. {
        0.
    } else {
        1.
    }
}

pub fn reshape_mat2d_into_mat1d(mat2d: Mat2D<f64>) -> Mat1D<f64> {
    Mat1D::from_vec(mat2d.vec())
}

pub fn reshape_mat1d_into_mat2d(mat1d: Mat1D<f64>, num_rows: usize) -> Mat2D<f64> {
    Mat2D::from_rows(mat1d.vec(), num_rows, mat1d.size() / num_rows)
}

pub struct Network {
    first_hidden_layer: Layer,
    second_hidden_layer: Layer,
    output_layer: Layer,
}

impl Network {
    pub fn random(
        input_size: usize,
        first_hidden_layer_output_size: usize,
        second_hidden_layer_output_size: usize,
        output_layer_output_size: usize,
    ) -> Self {
        Network {
            first_hidden_layer: Layer::random(input_size, first_hidden_layer_output_size),
            second_hidden_layer: Layer::random(
                first_hidden_layer_output_size,
                second_hidden_layer_output_size,
            ),
            output_layer: Layer::random(second_hidden_layer_output_size, output_layer_output_size),
        }
    }

    fn calc_activations(&self, input: Mat1D<f64>) -> [Mat1D<f64>; 3] {
        assert_eq!(input.size(), self.input_size(), "incorrect input size");

        let first_hidden_layer_activations = (&self.first_hidden_layer.weights * &input
            + &self.first_hidden_layer.biases)
            .map(|x| x.max(0.));

        let second_hidden_layer_activations = (&self.second_hidden_layer.weights
            * &first_hidden_layer_activations
            + &self.second_hidden_layer.biases)
            .map(|x| x.max(0.));

        let output_layer_activations = (&self.output_layer.weights
            * &second_hidden_layer_activations
            + &self.output_layer.biases)
            .map(|x| x.max(0.));

        [
            first_hidden_layer_activations,
            second_hidden_layer_activations,
            output_layer_activations,
        ]
    }

    pub fn infer(&self, input: Mat1D<f64>) -> Mat1D<f64> {
        let [_, _, output] = self.calc_activations(input);
        output
    }

    pub fn calc_gradients(&mut self, input: Mat1D<f64>, expected_output: Mat1D<f64>) {
        assert_eq!(input.size(), self.input_size(), "incorrect input size");
        assert_eq!(
            expected_output.size(),
            self.output_size(),
            "incorrect output size"
        );

        for _ in 0..500 {
            let [a1, a2, a3] = self.calc_activations(input.to_owned());

            let c = (&a3 - &expected_output).map(|v| v * v)[0];

            println!("cost: {}", c);

            let dc_da3 = (&a3 - &expected_output).map(|v| 2. * v);

            let mut z = a3.to_owned();
            for i in 0..a3.size() {
                z[i] = drelu((&self.output_layer.weights * &a2 + &self.output_layer.biases)[i])
            }

            let mut da3_dw3 = self.output_layer.weights.to_owned();
            for i in 0..da3_dw3.num_rows() {
                for j in 0..da3_dw3.num_columns() {
                    da3_dw3[(i, j)] = a2[j] * z[i];
                }
            }
            let da3_db3 = z;

            let dc_db3 = dc_da3.mat2d().transpose() * da3_db3;
            let dc_dw3 = dc_da3.mat2d().transpose() * da3_dw3;

            // println!("w3:\n{}", self.output_layer.weights);
            // println!("dc_dw3:\n{}", dc_dw3);
            // println!("b3:\n{}", self.output_layer.biases);
            // println!("dc_db3:\n{}", dc_db3);

            self.output_layer.weights = &self.output_layer.weights - dc_dw3.map(|v| 0.01 * v);
            self.output_layer.biases = &self.output_layer.biases - dc_db3.map(|v| 0.01 * v);
        }
    }

    fn input_size(&self) -> usize {
        self.first_hidden_layer.input_size
    }
    fn hidden_layer_size(&self) -> usize {
        self.first_hidden_layer.output_size
    }
    fn output_size(&self) -> usize {
        self.output_layer.output_size
    }
}

struct Layer {
    input_size: usize,
    output_size: usize,
    weights: Mat2D<f64>,
    biases: Mat1D<f64>,
}

impl Layer {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Layer {
            input_size,
            output_size,
            weights: Mat2D::filled_with(0., input_size, output_size),
            biases: Mat1D::filled_with(0., input_size),
        }
    }

    pub fn random(input_size: usize, output_size: usize) -> Self {
        let max = 1. / (input_size as f64).sqrt();
        let mut rng = rand::thread_rng();

        let mut weights_array = vec![0.; input_size * output_size];
        weights_array.fill_with(|| max * (2. * rng.sample::<f64, _>(Open01) - 1.));

        Layer {
            input_size,
            output_size,
            weights: Mat2D::from_rows(weights_array, output_size, input_size),
            biases: Mat1D::filled_with(0.01, output_size),
        }
    }
}
