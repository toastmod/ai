
pub type Float = f32;

#[macro_export]
macro_rules! forw_calc {
    ($activ_f:expr, $weighted_sum:expr, $activ_transform:expr, $activ_v:expr) => (
        (($activ_f)($weighted_sum)+$activ_transform)*$activ_v
    );
}

#[macro_export]
macro_rules! sigmoid {
    ($a:expr) => (
        1.0/1.0+Float::powf(std::f32::consts::E,-$a)
    );
}