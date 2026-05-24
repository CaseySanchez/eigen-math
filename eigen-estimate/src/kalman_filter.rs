use nalgebra::{
    Const,
    OMatrix,
    OVector,
};

pub trait StateTransitionFunctor<Scalar, const DIM: usize>
where
    Scalar: nalgebra::RealField + nalgebra::Scalar,
{
    fn evaluate(&self, delta_time: &Scalar) -> OMatrix<Scalar, Const<DIM>, Const<DIM>>;
}

impl<Scalar, const DIM: usize, F> StateTransitionFunctor<Scalar, DIM> for F
where
    Scalar: nalgebra::RealField + nalgebra::Scalar,
    F: Fn(&Scalar) -> OMatrix<Scalar, Const<DIM>, Const<DIM>>,
{
    fn evaluate(&self, delta_time: &Scalar) -> OMatrix<Scalar, Const<DIM>, Const<DIM>> {
        self(delta_time)
    }
}

pub trait ProcessCovarianceFunctor<Scalar, const DIM: usize>
where
    Scalar: nalgebra::RealField + nalgebra::Scalar,
{
    fn evaluate(&self, delta_time: &Scalar) -> OMatrix<Scalar, Const<DIM>, Const<DIM>>;
}

impl<Scalar, const DIM: usize, F> ProcessCovarianceFunctor<Scalar, DIM> for F
where
    Scalar: nalgebra::RealField + nalgebra::Scalar,
    F: Fn(&Scalar) -> OMatrix<Scalar, Const<DIM>, Const<DIM>>,
{
    fn evaluate(&self, delta_time: &Scalar) -> OMatrix<Scalar, Const<DIM>, Const<DIM>> {
        self(delta_time)
    }
}

pub struct KalmanFilter<
    Scalar,
    const DIM: usize,
    StateTransition,
    ProcessCovariance,
>
where
    Scalar: nalgebra::RealField + nalgebra::Scalar,
    StateTransition: StateTransitionFunctor<Scalar, DIM>,
    ProcessCovariance: ProcessCovarianceFunctor<Scalar, DIM>,
{
    state_transition: StateTransition,
    process_covariance: ProcessCovariance,
    pub estimate_state: OVector<Scalar, Const<DIM>>,
    pub estimate_covariance: OMatrix<Scalar, Const<DIM>, Const<DIM>>,
}

impl<Scalar, const DIM: usize, StateTransition, ProcessCovariance> KalmanFilter<Scalar, DIM, StateTransition, ProcessCovariance>
where
    Scalar: nalgebra::RealField,
    StateTransition: StateTransitionFunctor<Scalar, DIM>,
    ProcessCovariance: ProcessCovarianceFunctor<Scalar, DIM>,
{
    pub fn new(state_transition: StateTransition, process_covariance: ProcessCovariance) -> Self {
        Self {
            state_transition,
            process_covariance,
            estimate_state: OVector::<Scalar, Const<DIM>>::zeros(),
            estimate_covariance: OMatrix::<Scalar, Const<DIM>, Const<DIM>>::identity(),
        }
    }

    pub fn predict(&mut self, delta_time: &Scalar) -> &mut Self {
        let state_transition = self.state_transition.evaluate(delta_time);
        let process_covariance = self.process_covariance.evaluate(delta_time);

        let predicted_state = &state_transition * &self.estimate_state;
        let predicted_covariance = &state_transition * &self.estimate_covariance * state_transition.transpose() + process_covariance;

        self.estimate_state = predicted_state;
        self.estimate_covariance = predicted_covariance;

        self
    }

    pub fn update<const OBSERVATION_DIM: usize>(
        &mut self,
        observation_state: &OVector<Scalar, Const<OBSERVATION_DIM>>,
        observation_covariance: &OMatrix<Scalar, Const<OBSERVATION_DIM>, Const<OBSERVATION_DIM>>,
        observation_matrix: &OMatrix<Scalar, Const<OBSERVATION_DIM>, Const<DIM>>,
    ) -> &mut Self
    {
        let innovation_state = observation_state - observation_matrix * &self.estimate_state;
        let innovation_covariance = observation_matrix * &self.estimate_covariance * observation_matrix.transpose() + observation_covariance;

        let kalman_gain = &self.estimate_covariance * observation_matrix.transpose() * innovation_covariance.try_inverse().unwrap();
        let state_update_jacobian = &OMatrix::<Scalar, Const<DIM>, Const<DIM>>::identity() - &kalman_gain * observation_matrix;

        self.estimate_state += &kalman_gain * innovation_state;
        self.estimate_covariance = &state_update_jacobian * &self.estimate_covariance * state_update_jacobian.transpose() + &kalman_gain * observation_covariance * kalman_gain.transpose();

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, OMatrix, OVector};

    #[test]
    fn predict_advances_state_with_constant_velocity() {
        let mut kalman_filter = KalmanFilter::<f64, 2, _, _>::new(
            |dt: &f64| OMatrix::<f64, Const<2>, Const<2>>::from_row_slice(&[1.0, *dt, 0.0, 1.0]),
            |_: &f64| OMatrix::<f64, Const<2>, Const<2>>::zeros(),
        );

        kalman_filter.estimate_state[1] = 1.0;

        kalman_filter.predict(&1.0);

        assert!((kalman_filter.estimate_state[0] - 1.0).abs() < 1e-10);
        assert!((kalman_filter.estimate_state[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn update_with_equal_covariances_gives_midpoint() {
        let mut kalman_filter = KalmanFilter::<f64, 1, _, _>::new(
            |_: &f64| OMatrix::<f64, Const<1>, Const<1>>::identity(),
            |_: &f64| OMatrix::<f64, Const<1>, Const<1>>::zeros(),
        );

        kalman_filter.update(
            &OVector::<f64, Const<1>>::from_element(5.0),
            &OMatrix::<f64, Const<1>, Const<1>>::identity(),
            &OMatrix::<f64, Const<1>, Const<1>>::identity(),
        );

        assert!((kalman_filter.estimate_state[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn update_with_low_observation_noise_converges_to_measurement() {
        let mut kalman_filter = KalmanFilter::<f64, 1, _, _>::new(
            |_: &f64| OMatrix::<f64, Const<1>, Const<1>>::identity(),
            |_: &f64| OMatrix::<f64, Const<1>, Const<1>>::zeros(),
        );

        kalman_filter.update(
            &OVector::<f64, Const<1>>::from_element(10.0),
            &OMatrix::<f64, Const<1>, Const<1>>::from_element(1e-10),
            &OMatrix::<f64, Const<1>, Const<1>>::identity(),
        );

        assert!((kalman_filter.estimate_state[0] - 10.0).abs() < 1e-6);
    }
}
