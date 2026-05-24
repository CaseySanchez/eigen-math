use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::ToTypenum;
use nalgebra::{Const, DefaultAllocator, DimDiff, DimMin, DimSub, OMatrix, OVector, RealField, U1};

#[derive(Debug)]
pub enum UmeyamaError {
    InvalidSvd,
}

pub fn umeyama<Scalar, const FEATURE_DIM: usize, const POINT_COUNT: usize>(
    source_points: &OMatrix<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>,
    destination_points: &OMatrix<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>,
) -> Result<
    (
        OMatrix<Scalar, Const<FEATURE_DIM>, Const<FEATURE_DIM>>,
        OVector<Scalar, Const<FEATURE_DIM>>,
    ),
    UmeyamaError,
>
where
    Scalar: RealField + Copy,
    Const<FEATURE_DIM>:
        ToTypenum + DimMin<Const<FEATURE_DIM>, Output = Const<FEATURE_DIM>> + DimSub<U1>,
    DefaultAllocator: Allocator<Const<FEATURE_DIM>, Const<POINT_COUNT>>
        + Allocator<Const<POINT_COUNT>, Const<FEATURE_DIM>>
        + Allocator<Const<FEATURE_DIM>, Const<FEATURE_DIM>>
        + Allocator<Const<FEATURE_DIM>>
        + Allocator<DimDiff<Const<FEATURE_DIM>, U1>>,
{
    let inverse_point_count = Scalar::one() / Scalar::from_usize(POINT_COUNT).unwrap();

    let source_centroid = source_points.column_sum() * inverse_point_count;
    let destination_centroid = destination_points.column_sum() * inverse_point_count;

    let centered_source =
        OMatrix::<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>::from_fn(|row, col| {
            source_points[(row, col)] - source_centroid[row]
        });

    let centered_destination =
        OMatrix::<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>::from_fn(|row, col| {
            destination_points[(row, col)] - destination_centroid[row]
        });

    let covariance_matrix =
        centered_destination * centered_source.transpose() * inverse_point_count;

    let svd = covariance_matrix.svd(true, true);

    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return Err(UmeyamaError::InvalidSvd),
    };

    let determinant = (&u * &v_t).determinant();

    let mut reflection_correction =
        OMatrix::<Scalar, Const<FEATURE_DIM>, Const<FEATURE_DIM>>::identity();
    reflection_correction[(FEATURE_DIM - 1, FEATURE_DIM - 1)] = determinant.signum();

    let rotation_matrix = u * reflection_correction * v_t;
    let translation_vector = destination_centroid - &rotation_matrix * source_centroid;

    Ok((rotation_matrix, translation_vector))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, OMatrix};

    #[test]
    fn identity_transform() {
        let points = OMatrix::<f64, Const<3>, Const<4>>::from_row_slice(&[
            1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        let (rotation, translation) = umeyama(&points, &points).unwrap();

        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (rotation[(row, col)] - expected).abs() < 1e-10,
                    "rotation[{row},{col}] = {}",
                    rotation[(row, col)]
                );
            }

            assert!(
                translation[row].abs() < 1e-10,
                "translation[{row}] = {}",
                translation[row]
            );
        }
    }
}
