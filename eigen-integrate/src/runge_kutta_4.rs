use nalgebra::{Const, OVector, RealField};

pub trait DerivativeFunctor<Scalar, const DIM: usize>
where
    Scalar: RealField + Copy,
{
    fn evaluate(
        &self,
        state: &OVector<Scalar, Const<DIM>>,
        time: Scalar,
    ) -> OVector<Scalar, Const<DIM>>;
}

impl<Scalar, const DIM: usize, F> DerivativeFunctor<Scalar, DIM> for F
where
    Scalar: RealField + Copy,
    F: Fn(&OVector<Scalar, Const<DIM>>, Scalar) -> OVector<Scalar, Const<DIM>>,
{
    fn evaluate(
        &self,
        state: &OVector<Scalar, Const<DIM>>,
        time: Scalar,
    ) -> OVector<Scalar, Const<DIM>> {
        self(state, time)
    }
}

pub struct RungeKutta4<Scalar, const DIM: usize, Derivative>
where
    Scalar: RealField + Copy,
    Derivative: DerivativeFunctor<Scalar, DIM>,
{
    derivative: Derivative,
    pub time: Scalar,
    pub state: OVector<Scalar, Const<DIM>>,
}

impl<Scalar, const DIM: usize, Derivative> RungeKutta4<Scalar, DIM, Derivative>
where
    Scalar: RealField + Copy,
    Derivative: DerivativeFunctor<Scalar, DIM>,
{
    pub fn new(derivative: Derivative) -> Self {
        Self {
            derivative,
            time: Scalar::zero(),
            state: OVector::<Scalar, Const<DIM>>::zeros(),
        }
    }

    pub fn compute(&mut self, delta_time: Scalar) {
        let half = Scalar::from_f64(0.5).unwrap();
        let two = Scalar::from_f64(2.0).unwrap();
        let six = Scalar::from_f64(6.0).unwrap();

        let t1 = self.time;
        let s1 = self.state;
        let k1 = self.derivative.evaluate(&s1, t1);

        let t2 = t1 + half * delta_time;
        let s2 = s1 + k1 * (half * delta_time);
        let k2 = self.derivative.evaluate(&s2, t2);

        let s3 = s1 + k2 * (half * delta_time);
        let k3 = self.derivative.evaluate(&s3, t2);

        let s4 = s1 + k3 * delta_time;
        let k4 = self.derivative.evaluate(&s4, t1 + delta_time);

        self.time += delta_time;
        self.state += (k1 + k2 * two + k3 * two + k4) * (delta_time / six);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, OVector};

    #[test]
    fn exponential_decay() {
        let mut runge_kutta_4 =
            RungeKutta4::<f64, 1, _>::new(|state: &OVector<f64, Const<1>>, _time: f64| -state);

        runge_kutta_4.state[0] = 1.0;

        for _ in 0..1000 {
            runge_kutta_4.compute(0.001);
        }

        assert!((runge_kutta_4.state[0] - (-1.0_f64).exp()).abs() < 1e-6);
    }
}
