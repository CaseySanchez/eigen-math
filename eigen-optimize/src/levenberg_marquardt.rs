use nalgebra::{Const, OMatrix, OVector, RealField};

pub trait ResidualsFunctor<Scalar, const PARAMETER_DIM: usize, const RESIDUAL_DIM: usize>
where
    Scalar: RealField + Copy,
{
    fn evaluate(
        &self,
        parameters: &OVector<Scalar, Const<PARAMETER_DIM>>,
    ) -> OVector<Scalar, Const<RESIDUAL_DIM>>;
}

impl<Scalar, const PARAMETER_DIM: usize, const RESIDUAL_DIM: usize, F>
    ResidualsFunctor<Scalar, PARAMETER_DIM, RESIDUAL_DIM> for F
where
    Scalar: RealField + Copy,
    F: Fn(&OVector<Scalar, Const<PARAMETER_DIM>>) -> OVector<Scalar, Const<RESIDUAL_DIM>>,
{
    fn evaluate(
        &self,
        parameters: &OVector<Scalar, Const<PARAMETER_DIM>>,
    ) -> OVector<Scalar, Const<RESIDUAL_DIM>> {
        self(parameters)
    }
}

pub trait PostStepFunctor<Scalar, const PARAMETER_DIM: usize>
where
    Scalar: RealField + Copy,
{
    fn evaluate(&mut self, parameters: &mut OVector<Scalar, Const<PARAMETER_DIM>>);
}

impl<Scalar, const PARAMETER_DIM: usize, F> PostStepFunctor<Scalar, PARAMETER_DIM> for F
where
    Scalar: RealField + Copy,
    F: FnMut(&mut OVector<Scalar, Const<PARAMETER_DIM>>),
{
    fn evaluate(&mut self, parameters: &mut OVector<Scalar, Const<PARAMETER_DIM>>) {
        self(parameters)
    }
}

#[derive(Clone, Debug)]
pub struct LevenbergMarquardtConfig<Scalar> {
    pub max_iterations: usize,
    pub lambda_initial: Scalar,
    pub lambda_factor: Scalar,
    pub convergence_tol: Scalar,
    pub finite_difference_epsilon: Scalar,
}

impl<Scalar: RealField + Copy> Default for LevenbergMarquardtConfig<Scalar> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            lambda_initial: Scalar::from_f64(1e-3).unwrap(),
            lambda_factor: Scalar::from_f64(10.0).unwrap(),
            convergence_tol: Scalar::from_f64(1e-8).unwrap(),
            finite_difference_epsilon: Scalar::from_f64(1e-6).unwrap(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LevenbergMarquardtResult<Scalar> {
    pub converged: bool,
    pub cost: Scalar,
    pub iterations: usize,
}

pub struct LevenbergMarquardt<
    Scalar,
    const PARAMETER_DIM: usize,
    const RESIDUAL_DIM: usize,
    Residuals,
    PostStep,
> where
    Scalar: RealField + Copy,
    Residuals: ResidualsFunctor<Scalar, PARAMETER_DIM, RESIDUAL_DIM>,
    PostStep: PostStepFunctor<Scalar, PARAMETER_DIM>,
{
    residuals: Residuals,
    post_step: PostStep,
    pub config: LevenbergMarquardtConfig<Scalar>,
    pub parameters: OVector<Scalar, Const<PARAMETER_DIM>>,
}

impl<Scalar, const PARAMETER_DIM: usize, const RESIDUAL_DIM: usize, Residuals, PostStep>
    LevenbergMarquardt<Scalar, PARAMETER_DIM, RESIDUAL_DIM, Residuals, PostStep>
where
    Scalar: RealField + Copy,
    Residuals: ResidualsFunctor<Scalar, PARAMETER_DIM, RESIDUAL_DIM>,
    PostStep: PostStepFunctor<Scalar, PARAMETER_DIM>,
{
    pub fn new(
        residuals: Residuals,
        post_step: PostStep,
        config: LevenbergMarquardtConfig<Scalar>,
    ) -> Self {
        Self {
            residuals,
            post_step,
            config,
            parameters: OVector::<Scalar, Const<PARAMETER_DIM>>::zeros(),
        }
    }

    pub fn solve(&mut self) -> LevenbergMarquardtResult<Scalar> {
        let mut lambda = self.config.lambda_initial;
        let mut current_residuals = self.residuals.evaluate(&self.parameters);
        let mut current_cost =
            Scalar::from_f64(0.5).unwrap() * current_residuals.dot(&current_residuals);

        for iteration in 0..self.config.max_iterations {
            let jacobian_matrix = compute_numerical_jacobian(
                &self.residuals,
                &self.parameters,
                &current_residuals,
                self.config.finite_difference_epsilon,
            );

            let jacobian_transposed = jacobian_matrix.transpose();
            let normal_matrix = jacobian_transposed * jacobian_matrix;
            let gradient = jacobian_transposed * current_residuals;

            let floor = Scalar::from_f64(1e-15).unwrap();
            let mut augmented_matrix = normal_matrix;

            for i in 0..PARAMETER_DIM {
                augmented_matrix[(i, i)] += lambda * normal_matrix[(i, i)].max(floor);
            }

            let Some(cholesky) = augmented_matrix.cholesky() else {
                lambda *= self.config.lambda_factor;
                continue;
            };

            let delta_parameters = -cholesky.solve(&gradient);

            let mut trial_parameters = self.parameters + delta_parameters;

            self.post_step.evaluate(&mut trial_parameters);

            let trial_residuals = self.residuals.evaluate(&trial_parameters);
            let trial_cost = Scalar::from_f64(0.5).unwrap() * trial_residuals.dot(&trial_residuals);

            if trial_cost < current_cost {
                self.parameters = trial_parameters;
                current_residuals = trial_residuals;
                current_cost = trial_cost;
                lambda /= self.config.lambda_factor;

                if delta_parameters.norm() < self.config.convergence_tol {
                    return LevenbergMarquardtResult {
                        converged: true,
                        cost: current_cost,
                        iterations: iteration + 1,
                    };
                }
            } else {
                lambda *= self.config.lambda_factor;
            }
        }

        LevenbergMarquardtResult {
            converged: false,
            cost: current_cost,
            iterations: self.config.max_iterations,
        }
    }
}

fn compute_numerical_jacobian<
    Scalar,
    Residuals,
    const PARAMETER_DIM: usize,
    const RESIDUAL_DIM: usize,
>(
    residuals_functor: &Residuals,
    parameters: &OVector<Scalar, Const<PARAMETER_DIM>>,
    base_residuals: &OVector<Scalar, Const<RESIDUAL_DIM>>,
    epsilon: Scalar,
) -> OMatrix<Scalar, Const<RESIDUAL_DIM>, Const<PARAMETER_DIM>>
where
    Scalar: RealField + Copy,
    Residuals: ResidualsFunctor<Scalar, PARAMETER_DIM, RESIDUAL_DIM>,
{
    let mut jacobian_matrix = OMatrix::<Scalar, Const<RESIDUAL_DIM>, Const<PARAMETER_DIM>>::zeros();
    let mut perturbed_parameters = *parameters;

    for parameter_index in 0..PARAMETER_DIM {
        perturbed_parameters[parameter_index] += epsilon;

        let perturbed_residuals = residuals_functor.evaluate(&perturbed_parameters);

        perturbed_parameters[parameter_index] -= epsilon;

        for residual_index in 0..RESIDUAL_DIM {
            jacobian_matrix[(residual_index, parameter_index)] =
                (perturbed_residuals[residual_index] - base_residuals[residual_index]) / epsilon;
        }
    }

    jacobian_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, OVector};

    #[test]
    fn simple_quadratic() {
        let target = OVector::<f64, Const<2>>::from_row_slice(&[3.0, 4.0]);

        let mut levenberg_marquardt = LevenbergMarquardt::new(
            |p: &OVector<f64, Const<2>>| p - &target,
            |_: &mut OVector<f64, Const<2>>| {},
            LevenbergMarquardtConfig::default(),
        );

        let result = levenberg_marquardt.solve();

        assert!(result.converged);
        assert!((levenberg_marquardt.parameters[0] - 3.0).abs() < 1e-6);
        assert!((levenberg_marquardt.parameters[1] - 4.0).abs() < 1e-6);
    }
}
