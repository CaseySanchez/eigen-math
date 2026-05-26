use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::ToTypenum;
use nalgebra::{
    Const, DefaultAllocator, DimAdd, DimDiff, DimMin, DimName, DimSub, DimSum, OMatrix, OVector,
    RealField, U1,
};

#[derive(Debug)]
pub enum UmeyamaError {
    InvalidPointCount,
    InvalidSvd,
}

pub fn umeyama<Scalar, const FEATURE_DIM: usize, const POINT_COUNT: usize>(
    source_points: &OMatrix<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>,
    destination_points: &OMatrix<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>,
    point_count: usize,
) -> Result<
    OMatrix<Scalar, DimSum<Const<FEATURE_DIM>, U1>, DimSum<Const<FEATURE_DIM>, U1>>,
    UmeyamaError,
>
where
    Scalar: RealField + Copy,
    Const<FEATURE_DIM>: ToTypenum
        + DimMin<Const<FEATURE_DIM>, Output = Const<FEATURE_DIM>>
        + DimSub<U1>
        + DimAdd<U1>,
    DimSum<Const<FEATURE_DIM>, U1>: DimName,
    DefaultAllocator: Allocator<Const<FEATURE_DIM>, Const<POINT_COUNT>>
        + Allocator<Const<POINT_COUNT>, Const<FEATURE_DIM>>
        + Allocator<Const<FEATURE_DIM>, Const<FEATURE_DIM>>
        + Allocator<Const<FEATURE_DIM>>
        + Allocator<DimDiff<Const<FEATURE_DIM>, U1>>
        + Allocator<DimSum<Const<FEATURE_DIM>, U1>, DimSum<Const<FEATURE_DIM>, U1>>,
{
    if point_count < 1 || point_count > POINT_COUNT {
        return Err(UmeyamaError::InvalidPointCount);
    }

    let inverse_point_count = Scalar::one() / Scalar::from_usize(point_count).unwrap();

    let mut source_centroid = OVector::<Scalar, Const<FEATURE_DIM>>::zeros();
    let mut destination_centroid = OVector::<Scalar, Const<FEATURE_DIM>>::zeros();

    for col in 0..point_count {
        source_centroid += source_points.column(col);
        destination_centroid += destination_points.column(col);
    }

    source_centroid *= inverse_point_count;
    destination_centroid *= inverse_point_count;

    let centered_source =
        OMatrix::<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>::from_fn(|row, col| {
            if col < point_count {
                source_points[(row, col)] - source_centroid[row]
            } else {
                Scalar::zero()
            }
        });

    let centered_destination =
        OMatrix::<Scalar, Const<FEATURE_DIM>, Const<POINT_COUNT>>::from_fn(|row, col| {
            if col < point_count {
                destination_points[(row, col)] - destination_centroid[row]
            } else {
                Scalar::zero()
            }
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
    let translation_vector: OVector<Scalar, Const<FEATURE_DIM>> =
        destination_centroid - &rotation_matrix * source_centroid;

    let mut transform =
        OMatrix::<Scalar, DimSum<Const<FEATURE_DIM>, U1>, DimSum<Const<FEATURE_DIM>, U1>>::zeros();

    for row in 0..FEATURE_DIM {
        for col in 0..FEATURE_DIM {
            transform[(row, col)] = rotation_matrix[(row, col)];
        }

        transform[(row, FEATURE_DIM)] = translation_vector[row];
    }

    transform[(FEATURE_DIM, FEATURE_DIM)] = Scalar::one();

    Ok(transform)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, OMatrix};

    #[test]
    fn invalid_point_count_returns_error() {
        let points = OMatrix::<f64, Const<3>, Const<4>>::zeros();

        assert!(matches!(
            umeyama(&points, &points, 0),
            Err(UmeyamaError::InvalidPointCount)
        ));

        assert!(matches!(
            umeyama(&points, &points, 5),
            Err(UmeyamaError::InvalidPointCount)
        ));
    }

    #[test]
    fn identity_transform() {
        let points = OMatrix::<f64, Const<3>, Const<4>>::from_row_slice(&[
            1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        let transform = umeyama(&points, &points, 4).unwrap();

        for row in 0..4 {
            for col in 0..4 {
                let expected = if row == col { 1.0 } else { 0.0 };

                assert!(
                    (transform[(row, col)] - expected).abs() < 1e-10,
                    "transform[{row},{col}] = {}",
                    transform[(row, col)]
                );
            }
        }
    }
}
