pub type Float = f64;

#[macro_export]
macro_rules! forw_calc {
    ($activ_f:expr, $weighted_sum:expr, $activ_transform:expr) => (
        (($activ_f)($weighted_sum+$activ_transform))
    );
}

#[macro_export]
macro_rules! sigmoid {
    ($a:expr) => (
        1.0/(1.0+Float::powf(std::f32::consts::E as Float,-($a/200.0)))
    );
}

#[macro_export]
macro_rules! sigmoid_derivative {
    ($a:expr) => (
        $a * (1.0 - $a)
    );
}
