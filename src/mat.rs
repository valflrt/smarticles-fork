use std::{
    fmt::Display,
    ops::{Add, Index, IndexMut, Mul, Sub},
};

use rand::{distributions::Uniform, Rng};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Mat2D<T> {
    num_rows: usize,
    num_columns: usize,
    vec: Vec<T>,
    transpose: bool,
}

impl<T> Mat2D<T>
where
    T: Clone,
{
    pub fn from_rows(vec: Vec<T>, num_rows: usize, num_columns: usize) -> Self {
        assert_eq!(
            vec.len(),
            num_rows * num_columns,
            "vec does not match the number of rows and columns"
        );
        Self {
            num_rows,
            num_columns,
            vec,
            transpose: false,
        }
    }

    pub fn filled_with(value: T, num_rows: usize, num_columns: usize) -> Self {
        Self {
            num_rows,
            num_columns,
            vec: vec![value; num_rows * num_columns],
            transpose: false,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut out = self.to_owned();
        out.transpose ^= true;
        out
    }
    pub fn transpose_in_place(&mut self) {
        self.transpose ^= true;
    }

    pub fn vec(&self) -> Vec<T> {
        self.vec.to_owned()
    }

    pub fn num_rows(&self) -> usize {
        if !self.transpose {
            self.num_rows
        } else {
            self.num_columns
        }
    }
    pub fn num_columns(&self) -> usize {
        if !self.transpose {
            self.num_columns
        } else {
            self.num_rows
        }
    }
}

impl Mat2D<f64> {
    pub fn random(range: (f64, f64), num_rows: usize, num_columns: usize) -> Self {
        let mut rng = rand::thread_rng();

        let mut vec = vec![0.; num_rows * num_columns];
        vec.fill_with(|| rng.sample::<f64, _>(Uniform::new(range.0, range.1)));

        Self {
            num_rows,
            num_columns,
            vec,
            transpose: false,
        }
    }

    pub fn map<T>(&self, function: T) -> Self
    where
        T: (Fn(f64) -> f64) + Sync + Send,
    {
        Mat2D::from_rows(
            self.vec().par_iter().map(move |x| function(*x)).collect(),
            self.num_rows(),
            self.num_columns(),
        )
    }

    pub fn elementwise_product(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.num_rows(),
            rhs.num_rows(),
            "num rows must match when performing element-wise product"
        );
        assert_eq!(
            self.num_columns(),
            rhs.num_columns(),
            "num columns must match when performing element-wise product"
        );

        Mat2D::from_rows(
            (0..self.num_rows())
                .into_par_iter()
                .flat_map(move |i| {
                    (0..self.num_columns())
                        .into_par_iter()
                        .map(move |j| self[(i, j)] * rhs[(i, j)])
                })
                .collect(),
            self.num_rows(),
            self.num_columns(),
        )
    }
}

impl<T> Index<(usize, usize)> for Mat2D<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = if !self.transpose {
            index
        } else {
            (index.1, index.0)
        };

        assert!(i < self.num_rows, "row index out of bounds");
        assert!(j < self.num_columns, "column index out of bounds");
        &self.vec[j + i * self.num_columns]
    }
}

impl<T> IndexMut<(usize, usize)> for Mat2D<T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = if !self.transpose {
            index
        } else {
            (index.1, index.0)
        };

        assert!(i < self.num_rows, "row index out of bounds");
        assert!(j < self.num_columns, "column index out of bounds");
        &mut self.vec[j + i * self.num_columns]
    }
}

impl Display for Mat2D<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.num_columns() < 12 {
            for i in 0..self.num_rows() {
                for j in 0..self.num_columns() {
                    let s = if self[(i, j)].is_sign_negative() {
                        format!("{:.3}", self[(i, j)])
                    } else {
                        format!("{:.4}", self[(i, j)])
                    };

                    write!(
                        f,
                        "{}{}{}",
                        if j == 0 {
                            if i == 0 {
                                "[ "
                            } else {
                                "  "
                            }
                        } else {
                            " "
                        },
                        s,
                        if j == self.num_columns() - 1 {
                            if i == self.num_rows() - 1 {
                                " ]"
                            } else {
                                "\n"
                            }
                        } else {
                            " "
                        }
                    )?;
                }
            }
        } else {
            writeln!(f, "mat is too big to print")?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Mat1D<T> {
    mat2d: Mat2D<T>,
}

impl<T> Mat1D<T>
where
    T: Clone,
{
    pub fn from_vec(vec: Vec<T>) -> Self {
        let size = vec.len();
        Self {
            mat2d: Mat2D::from_rows(vec, size, 1),
        }
    }

    pub fn filled_with(value: T, size: usize) -> Self {
        Self {
            mat2d: Mat2D::filled_with(value, size, 1),
        }
    }

    pub fn mat2d(&self) -> Mat2D<T> {
        self.mat2d.to_owned()
    }

    pub fn vec(&self) -> Vec<T> {
        self.mat2d.vec()
    }

    pub fn size(&self) -> usize {
        self.mat2d.num_rows()
    }
}

impl<T> Mat1D<T>
where
    T: Copy,
{
    pub fn replication(&self, num_rows: usize) -> Mat2D<T> {
        Mat2D::from_rows(self.mat2d.vec().repeat(num_rows), num_rows, self.size())
    }
}

impl Mat1D<f64> {
    pub fn random(range: (f64, f64), size: usize) -> Self {
        Mat1D {
            mat2d: Mat2D::random(range, size, 1),
        }
    }

    pub fn map<T>(&self, function: T) -> Self
    where
        T: Fn(f64) -> f64 + Sync + Send,
    {
        Mat1D {
            mat2d: self.mat2d.map(function),
        }
    }

    pub fn elementwise_product(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.size(),
            rhs.size(),
            "sizes must match when performing element-wise product"
        );

        Mat1D::from_vec(
            (0..self.size())
                .into_par_iter()
                .map(move |i| self[i] * rhs[i])
                .collect(),
        )
    }
}

impl<T> Index<usize> for Mat1D<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.mat2d[(index, 0)]
    }
}

impl<T> IndexMut<usize> for Mat1D<T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.mat2d[(index, 0)]
    }
}

impl Display for Mat1D<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.size() {
            let s = if self[i].is_sign_negative() {
                format!("{:.3}", self[i])
            } else {
                format!("{:.4}", self[i])
            };

            write!(
                f,
                "{}{}{}",
                if i == 0 { "[ " } else { "  " },
                s,
                if i == self.size() - 1 { " ]" } else { "\n" }
            )?;
        }
        Ok(())
    }
}

impl Add for &Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.num_rows(),
            rhs.num_rows(),
            "num rows must match when performing addition"
        );
        assert_eq!(
            self.num_columns(),
            rhs.num_columns(),
            "num columns must match when performing addition"
        );

        Mat2D::from_rows(
            (0..self.num_rows())
                .into_par_iter()
                .flat_map(move |i| {
                    (0..self.num_columns())
                        .into_par_iter()
                        .map(move |j| self[(i, j)] + rhs[(i, j)])
                })
                .collect(),
            self.num_rows(),
            self.num_columns(),
        )
    }
}
impl Add for Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}
impl Add<&Mat2D<f64>> for Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn add(self, rhs: &Mat2D<f64>) -> Self::Output {
        &self + rhs
    }
}
impl Add<Mat2D<f64>> for &Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn add(self, rhs: Mat2D<f64>) -> Self::Output {
        self + &rhs
    }
}

impl Sub for &Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.num_rows(),
            rhs.num_rows(),
            "num rows must match when performing addition"
        );
        assert_eq!(
            self.num_columns(),
            rhs.num_columns(),
            "num columns must match when performing addition"
        );

        Mat2D::from_rows(
            (0..self.num_rows())
                .into_par_iter()
                .flat_map(move |i| {
                    (0..self.num_columns())
                        .into_par_iter()
                        .map(move |j| self[(i, j)] - rhs[(i, j)])
                })
                .collect(),
            self.num_rows(),
            self.num_columns(),
        )
    }
}
impl Sub for Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}
impl Sub<&Mat2D<f64>> for Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn sub(self, rhs: &Mat2D<f64>) -> Self::Output {
        &self - rhs
    }
}
impl Sub<Mat2D<f64>> for &Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn sub(self, rhs: Mat2D<f64>) -> Self::Output {
        self - &rhs
    }
}

impl Add for &Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size(),
            rhs.size(),
            "sizes must match when performing addition"
        );

        Mat1D::from_vec(
            (0..self.size())
                .into_par_iter()
                .map(move |i| self[i] + rhs[i])
                .collect(),
        )
    }
}
impl Add for Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}
impl Add<&Mat1D<f64>> for Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn add(self, rhs: &Mat1D<f64>) -> Self::Output {
        &self + rhs
    }
}
impl Add<Mat1D<f64>> for &Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn add(self, rhs: Mat1D<f64>) -> Self::Output {
        self + &rhs
    }
}

impl Sub for &Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size(),
            rhs.size(),
            "sizes must match when performing addition"
        );

        Mat1D::from_vec(
            (0..self.size())
                .into_par_iter()
                .map(move |i| self[i] - rhs[i])
                .collect(),
        )
    }
}
impl Sub for Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}
impl Sub<&Mat1D<f64>> for Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn sub(self, rhs: &Mat1D<f64>) -> Self::Output {
        &self - rhs
    }
}
impl Sub<Mat1D<f64>> for &Mat1D<f64> {
    type Output = Mat1D<f64>;

    fn sub(self, rhs: Mat1D<f64>) -> Self::Output {
        self - &rhs
    }
}

impl Mul for &Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.num_columns(), rhs.num_rows(),
            "rows of the first mat must match columns of the second mat when performing multiplication"
        );

        Mat2D::from_rows(
            (0..self.num_rows())
                .into_par_iter()
                .flat_map(move |i| {
                    (0..rhs.num_columns()).into_par_iter().map(move |j| {
                        (0..self.num_columns())
                            .map(|k| self[(i, k)] * rhs[(k, j)])
                            .sum::<f64>()
                    })
                })
                .collect(),
            self.num_rows(),
            rhs.num_columns(),
        )
    }
}
impl Mul for Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
impl Mul<&Mat2D<f64>> for Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn mul(self, rhs: &Mat2D<f64>) -> Self::Output {
        &self * rhs
    }
}
impl Mul<Mat2D<f64>> for &Mat2D<f64> {
    type Output = Mat2D<f64>;

    fn mul(self, rhs: Mat2D<f64>) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Mat1D<f64>> for &Mat2D<f64> {
    type Output = Mat1D<f64>;

    fn mul(self, rhs: &Mat1D<f64>) -> Self::Output {
        assert_eq!(
            self.num_columns(),
            rhs.size(),
            "columns of mat must match size of vec when performing multiplication"
        );

        Mat1D::from_vec(
            (0..self.num_rows())
                .into_par_iter()
                .map(move |i| {
                    (0..self.num_columns())
                        .map(|k| self[(i, k)] * rhs[k])
                        .sum::<f64>()
                })
                .collect(),
        )
    }
}
impl Mul<Mat1D<f64>> for Mat2D<f64> {
    type Output = Mat1D<f64>;

    fn mul(self, rhs: Mat1D<f64>) -> Self::Output {
        &self * &rhs
    }
}
impl Mul<&Mat1D<f64>> for Mat2D<f64> {
    type Output = Mat1D<f64>;

    fn mul(self, rhs: &Mat1D<f64>) -> Self::Output {
        &self * rhs
    }
}
impl Mul<Mat1D<f64>> for &Mat2D<f64> {
    type Output = Mat1D<f64>;

    fn mul(self, rhs: Mat1D<f64>) -> Self::Output {
        self * &rhs
    }
}

#[cfg(test)]
mod test {
    use crate::mat::Mat1D;

    use super::Mat2D;

    #[test]
    fn mat2d_filled_with() {
        let m = Mat2D::filled_with(0, 2, 3);

        assert_eq!(
            m,
            Mat2D {
                num_rows: 2,
                num_columns: 3,
                vec: vec![0, 0, 0, 0, 0, 0],
                transpose: false
            }
        );
    }

    #[test]
    fn mat2d_from_rows() {
        let v = vec![1, 2, 3, 4, 5, 6];

        let m = Mat2D::from_rows(v.clone(), 2, 3);

        assert_eq!(
            m,
            Mat2D {
                num_rows: 2,
                num_columns: 3,
                vec: v,
                transpose: false
            }
        );
    }

    #[test]
    #[should_panic]
    fn mat2d_from_rows_panic_cases() {
        let _ = Mat2D::from_rows(vec![1, 2, 3, 4], 2, 3);
        let _ = Mat2D::from_rows(vec![1, 2, 3, 4, 5, 6, 7], 2, 3);
    }

    #[test]
    fn mat2d_index() {
        let m = Mat2D::from_rows(vec![1, 2, 3, 4, 5, 6], 2, 3);

        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 2)], 3);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    fn mat2d_transpose() {
        let mut m1 = Mat2D::from_rows(vec![1, 2, 3, 4, 5, 6], 2, 3);
        let m2 = m1.transpose();
        m1.transpose_in_place();

        assert_eq!(m1[(0, 0)], 1);
        assert_eq!(m1[(2, 0)], 3);
        assert_eq!(m1[(2, 1)], 6);

        assert_eq!(m2[(0, 0)], 1);
        assert_eq!(m2[(2, 0)], 3);
        assert_eq!(m2[(2, 1)], 6);
    }

    #[test]
    fn mat2d_add() {
        let m1 = Mat2D::from_rows(vec![1., 2., 3., 4., 5., 6.], 2, 3);
        let m2 = Mat2D::from_rows(vec![6., 5., 4., 3., 2., 1.], 2, 3);

        assert_eq!(
            m1 + m2,
            Mat2D::from_rows(vec![7., 7., 7., 7., 7., 7.], 2, 3)
        );
    }

    #[test]
    #[should_panic]
    fn mat2d_add_panic_cases() {
        let m1 = Mat2D::from_rows(vec![1., 2., 3., 4., 5., 6.], 2, 3);
        let m2 = Mat2D::from_rows(vec![6., 5., 4., 3.], 2, 2);

        let _ = m1 + m2;
    }

    #[test]
    fn mat2d_element_wise_product() {
        let m1 = Mat2D::from_rows(vec![1., 2., 3., 4., 5., 6.], 2, 3);
        let m2 = Mat2D::from_rows(vec![6., 5., 4., 3., 2., 1.], 2, 3);

        assert_eq!(
            m1 + m2,
            Mat2D::from_rows(vec![7., 7., 7., 7., 7., 7.], 2, 3)
        );
    }

    #[test]
    fn mat2d_mul() {
        let m1 = Mat2D::from_rows([1., 2., 3., 4., 5., 6.].to_vec(), 2, 3);
        let m2 = Mat2D::from_rows([4., 3., 2.].to_vec(), 3, 1);

        assert_eq!(m1 * m2, Mat2D::from_rows([16., 43.].to_vec(), 2, 1));
    }

    #[test]
    fn mat2d_mat1d_mul() {
        let m = Mat2D::from_rows([1., 2., 3., 4., 5., 6.].to_vec(), 2, 3);
        let x = Mat1D::from_vec([4., 3., 2.].to_vec());

        assert_eq!(m * x, Mat1D::from_vec([16., 43.].to_vec()));
    }

    #[test]
    fn mat1d_replication() {
        let x = Mat1D::from_vec(vec![1., 2., 3.]);

        println!("{}", x.replication(4));
    }
}
