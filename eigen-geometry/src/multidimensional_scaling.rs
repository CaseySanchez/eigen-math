use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimDiff, DimSub, OMatrix, RealField, SymmetricEigen, U1};

#[derive(Debug)]
pub enum MultidimensionalScalingError {
    InvalidCount,
}

pub fn multidimensional_scaling<Scalar, const FEATURE_DIM: usize, const EMBEDDING_DIM: usize>(
    feature_matrix: &OMatrix<Scalar, Const<FEATURE_DIM>, Const<FEATURE_DIM>>,
    feature_count: usize,
) -> Result<OMatrix<Scalar, Const<EMBEDDING_DIM>, Const<FEATURE_DIM>>, MultidimensionalScalingError>
where
    Scalar: RealField + Copy,
    Const<FEATURE_DIM>: DimSub<U1>,
    DefaultAllocator: Allocator<Const<FEATURE_DIM>, Const<FEATURE_DIM>>
        + Allocator<Const<EMBEDDING_DIM>, Const<EMBEDDING_DIM>>
        + Allocator<Const<FEATURE_DIM>, Const<EMBEDDING_DIM>>
        + Allocator<Const<EMBEDDING_DIM>, Const<FEATURE_DIM>>
        + Allocator<Const<FEATURE_DIM>>
        + Allocator<DimDiff<Const<FEATURE_DIM>, U1>>,
{
    if feature_count < 1 || feature_count > FEATURE_DIM {
        return Err(MultidimensionalScalingError::InvalidCount);
    }

    let inverse_feature_count = Scalar::one() / Scalar::from_usize(feature_count).unwrap();

    let center_matrix =
        OMatrix::<Scalar, Const<FEATURE_DIM>, Const<FEATURE_DIM>>::from_fn(|row, column| {
            if row >= feature_count || column >= feature_count {
                Scalar::zero()
            } else if row == column {
                Scalar::one() - inverse_feature_count
            } else {
                Scalar::zero() - inverse_feature_count
            }
        });

    let gram_matrix =
        (&center_matrix * feature_matrix.map(|feature| feature * feature) * &center_matrix)
            * Scalar::from_f64(-0.5).unwrap();

    let mut eigen_solver = SymmetricEigen::new(gram_matrix);

    for index_1 in 0..FEATURE_DIM {
        let mut max_index = index_1;

        for index_2 in (index_1 + 1)..FEATURE_DIM {
            if eigen_solver.eigenvalues[index_2] > eigen_solver.eigenvalues[max_index] {
                max_index = index_2;
            }
        }

        if max_index != index_1 {
            eigen_solver
                .eigenvalues
                .as_mut_slice()
                .swap(max_index, index_1);
            eigen_solver.eigenvectors.swap_columns(max_index, index_1);
        }
    }

    let eigenvalues_diagonal_matrix =
        OMatrix::<Scalar, Const<EMBEDDING_DIM>, Const<EMBEDDING_DIM>>::from_fn(|row, column| {
            if row == column && row < feature_count {
                eigen_solver.eigenvalues[row].max(Scalar::zero()).sqrt()
            } else {
                Scalar::zero()
            }
        });

    let eigenvectors_embedding_matrix =
        OMatrix::<Scalar, Const<FEATURE_DIM>, Const<EMBEDDING_DIM>>::from_fn(|row, column| {
            if row < feature_count && column < feature_count {
                eigen_solver.eigenvectors[(row, column)]
            } else {
                Scalar::zero()
            }
        });

    Ok((eigenvectors_embedding_matrix * eigenvalues_diagonal_matrix).transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, OMatrix};

    #[test]
    fn invalid_feature_count_returns_error() {
        let matrix = OMatrix::<f64, Const<3>, Const<3>>::zeros();

        assert!(matches!(
            multidimensional_scaling::<f64, 3, 1>(&matrix, 0),
            Err(MultidimensionalScalingError::InvalidCount)
        ));

        assert!(matches!(
            multidimensional_scaling::<f64, 3, 1>(&matrix, 4),
            Err(MultidimensionalScalingError::InvalidCount)
        ));
    }

    #[test]
    fn collinear_points_embed_with_unit_spacing() {
        let distance_matrix = OMatrix::<f64, Const<3>, Const<3>>::from_row_slice(&[
            0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
        ]);

        let result = multidimensional_scaling::<f64, 3, 1>(&distance_matrix, 3).unwrap();

        let mut coords = [result[(0, 0)], result[(0, 1)], result[(0, 2)]];

        coords.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((coords[1] - coords[0] - 1.0).abs() < 1e-6);
        assert!((coords[2] - coords[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn equilateral_triangle_preserves_pairwise_distances() {
        let distance_matrix = OMatrix::<f64, Const<3>, Const<3>>::from_row_slice(&[
            0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        ]);

        let result = multidimensional_scaling::<f64, 3, 2>(&distance_matrix, 3).unwrap();

        let d01 = (result.column(0) - result.column(1)).norm();
        let d02 = (result.column(0) - result.column(2)).norm();
        let d12 = (result.column(1) - result.column(2)).norm();

        assert!((d01 - 1.0).abs() < 1e-6);
        assert!((d02 - 1.0).abs() < 1e-6);
        assert!((d12 - 1.0).abs() < 1e-6);
    }
}
