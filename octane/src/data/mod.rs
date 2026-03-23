//! Data structures and loading for episodes.

pub mod behaviors;
pub mod episode;
pub mod es_parquet;
pub mod jsonla;
pub mod trek;

pub use episode::{Episode, Frame};
pub use jsonla::{ActionDistribution, FrameState};
pub use trek::{Trek, TrekEntry, ParquetSource, discover_treks, discover_parquet_sources};

// VehicleState is part of the public API, used by tests and examples.
// The binary doesn't use it directly, hence the allow(unused_imports).
#[allow(unused_imports)]
pub use jsonla::VehicleState;

#[allow(unused_imports)]
pub use episode::EpisodeMeta;
#[allow(unused_imports)]
pub use es_parquet::EsParquetIndex;
#[allow(unused_imports)]
pub use trek::Epoch;
