//! Parquet reader for es parquet files (sample_es.parquet and breakdown es.parquet).
//!
//! Each row group corresponds to one epoch's worth of episode data.
//! Columns use the unified schema: epoch, e, t (float), state.*, reward, p.*, etc.

use anyhow::{Context, Result};
use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::info;

use super::jsonla::{ActionDistribution, EsFrame, FrameState};
use crate::envs::EnvType;

/// Episode location within a parquet file.
#[derive(Debug, Clone)]
pub struct ParquetEpisodeLocation {
    /// Row group indices that contain this episode's data.
    pub row_groups: Vec<usize>,
    /// Number of frames in this episode.
    pub frame_count: usize,
    /// Number of policy-step frames (integer t values only).
    pub n_policy_frames: usize,
    /// Number of alive (not crashed) policy frames.
    pub n_alive_policy_frames: usize,
    /// Total reward for this episode.
    pub total_reward: Option<f64>,
    /// Normalized discounted return: (1-gamma) * sum(gamma^t * reward_t).
    pub nz_return: Option<f64>,
}

/// Default discount factor for normalized return computation.
const DEFAULT_DISCOUNT: f64 = 0.95;

/// Index for efficient access to es parquet episodes.
#[derive(Debug)]
pub struct EsParquetIndex {
    path: PathBuf,
    /// Maps (epoch, episode) to location info.
    index: HashMap<(i64, i64), ParquetEpisodeLocation>,
    /// Column names in the parquet file.
    columns: Vec<String>,
}

/// Shared (env-agnostic) columns extracted from an arrow record batch.
/// Env-specific columns are handled by `EnvParquetColumns` in envs/.
struct SharedColumns {
    epoch: Vec<i64>,
    episode: Vec<i64>,
    /// Fractional timestep (f64). Policy step = floor(t).
    t: Vec<f64>,
    reward: Option<Vec<Option<f64>>>,
    action: Option<Vec<Option<String>>>,
    action_name: Option<Vec<Option<String>>>,
    crashed: Vec<bool>,
    /// Dynamic action probability columns: Vec of (deed_name, values).
    /// Discovered from parquet schema columns matching `p.*`.
    action_probs: Vec<(String, Vec<f64>)>,
    /// Value function estimate V(s) per row.
    v: Option<Vec<Option<f64>>>,
    /// Discounted return from `return` column.
    return_value: Option<Vec<Option<f64>>>,
    /// GAE advantage from `advantage` column.
    advantage: Option<Vec<Option<f64>>>,
    /// Normalized advantage from `nz_advantage` column.
    nz_advantage: Option<Vec<Option<f64>>>,
    /// Log-probability of chosen action from `tendency` column.
    tendency: Option<Vec<Option<f64>>>,
    /// Crash reward from `crash_reward` column.
    crash_reward: Option<Vec<Option<f64>>>,
    /// Crash score from `crash_score` column.
    crash_score: Option<Vec<Option<f64>>>,
    n_rows: usize,
}

// ── Column extraction helpers ────────────────────────────────────────────────

pub(crate) fn get_i64(batch: &arrow::record_batch::RecordBatch, name: &str) -> Vec<i64> {
    batch.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .map(|a| (0..a.len()).map(|i| a.value(i)).collect())
        .unwrap_or_else(|| vec![0; batch.num_rows()])
}

pub(crate) fn get_f64(batch: &arrow::record_batch::RecordBatch, name: &str) -> Vec<f64> {
    if let Some(col) = batch.column_by_name(name) {
        if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
            return (0..a.len()).map(|i| a.value(i)).collect();
        }
        // Fallback: Int64 → f64 (legacy parquet files)
        if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
            return (0..a.len()).map(|i| a.value(i) as f64).collect();
        }
    }
    vec![0.0; batch.num_rows()]
}

pub(crate) fn get_f64_opt(batch: &arrow::record_batch::RecordBatch, name: &str) -> Option<Vec<f64>> {
    batch.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .map(|a| (0..a.len()).map(|i| {
            if a.is_null(i) { 0.0 } else { a.value(i) }
        }).collect())
}

fn get_f64_nullable(
    batch: &arrow::record_batch::RecordBatch, name: &str,
) -> Option<Vec<Option<f64>>> {
    batch.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .map(|a| (0..a.len()).map(|i| {
            if a.is_null(i) { None } else { Some(a.value(i)) }
        }).collect())
}

fn get_bool(batch: &arrow::record_batch::RecordBatch, name: &str) -> Vec<bool> {
    batch.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .map(|a| (0..a.len()).map(|i| a.value(i)).collect())
        .unwrap_or_else(|| vec![false; batch.num_rows()])
}

pub(crate) fn get_string_opt(
    batch: &arrow::record_batch::RecordBatch, name: &str,
) -> Option<Vec<Option<String>>> {
    batch.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .map(|a| (0..a.len()).map(|i| {
            if a.is_null(i) { None } else { Some(a.value(i).to_string()) }
        }).collect())
}

// ── EsParquetIndex ───────────────────────────────────────────────────────────

impl EsParquetIndex {
    /// Build index from an es parquet file.
    pub fn build<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("Failed to open es parquet: {:?}", path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let metadata = builder.metadata().clone();
        let schema = builder.schema().clone();

        let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        info!(
            "Indexing es parquet: {} row groups, {} columns, {} total rows",
            metadata.num_row_groups(),
            columns.len(),
            metadata.file_metadata().num_rows(),
        );

        // Find column indices for metadata columns
        let epoch_idx = columns.iter().position(|c| c == "epoch")
            .context("es parquet missing 'epoch' column")?;
        let episode_idx = columns.iter().position(|c| c == "e")
            .context("es parquet missing 'e' column")?;
        let t_idx = columns.iter().position(|c| c == "t")
            .context("es parquet missing 't' column")?;
        let reward_idx = columns.iter().position(|c| c == "reward");
        let crashed_idx = columns.iter().position(|c| c == "state.crashed");

        // Read only the metadata columns to build the index
        let mut projection_indices: Vec<usize> = vec![epoch_idx, episode_idx, t_idx];
        if let Some(idx) = reward_idx { projection_indices.push(idx); }
        if let Some(idx) = crashed_idx { projection_indices.push(idx); }
        projection_indices.sort();
        projection_indices.dedup();

        let file = File::open(&path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_projection(parquet::arrow::ProjectionMask::leaves(
                builder.metadata().file_metadata().schema_descr(),
                projection_indices.clone(),
            ));
        let reader = builder.build()?;

        let has_crashed_col = crashed_idx.is_some();

        // Accumulate per-episode stats:
        // (frame_count, n_policy_frames, n_alive_policy_frames, total_reward, row_groups, discounted_sum)
        let mut accum: HashMap<(i64, i64), (usize, usize, usize, f64, Vec<usize>, f64)> = HashMap::new();
        let mut current_row_group = 0usize;
        let mut rows_remaining_in_rg = if metadata.num_row_groups() > 0 {
            metadata.row_group(0).num_rows() as usize
        } else {
            0
        };

        for batch_result in reader {
            let batch = batch_result?;
            let n = batch.num_rows();

            let epoch_col = batch.column_by_name("epoch")
                .context("Missing epoch column")?;
            let episode_col = batch.column_by_name("e")
                .context("Missing e column")?;
            let t_col = batch.column_by_name("t")
                .context("Missing t column")?;

            let epochs = epoch_col.as_any().downcast_ref::<Int64Array>()
                .context("epoch column not Int64")?;
            let episodes = episode_col.as_any().downcast_ref::<Int64Array>()
                .context("e column not Int64")?;
            // Accept t as either Float64 (new format) or Int64 (legacy)
            let ts_f64: Option<&Float64Array> = t_col.as_any().downcast_ref::<Float64Array>();
            let ts_i64: Option<&Int64Array> = t_col.as_any().downcast_ref::<Int64Array>();
            if ts_f64.is_none() && ts_i64.is_none() {
                anyhow::bail!("t column is neither Float64 nor Int64");
            }

            let rewards: Option<&Float64Array> = batch.column_by_name("reward")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let crashed_col: Option<&BooleanArray> = batch.column_by_name("state.crashed")
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

            for i in 0..n {
                let epoch = epochs.value(i);
                let episode = episodes.value(i);
                let t = if let Some(arr) = ts_f64 {
                    arr.value(i)
                } else {
                    ts_i64.unwrap().value(i) as f64
                };
                let reward = rewards
                    .and_then(|r| if r.is_null(i) { None } else { Some(r.value(i)) })
                    .unwrap_or(0.0);
                let is_crashed = crashed_col
                    .map(|c| c.value(i))
                    .unwrap_or(false);

                let is_policy_frame = (t - t.floor()).abs() < 1e-6;

                let entry = accum.entry((epoch, episode))
                    .or_insert((0, 0, 0, 0.0, Vec::new(), 0.0));
                entry.0 += 1; // frame_count
                if is_policy_frame {
                    let policy_t = entry.1;
                    entry.5 += DEFAULT_DISCOUNT.powi(policy_t as i32) * reward;
                    entry.1 += 1; // n_policy_frames
                    if !is_crashed {
                        entry.2 += 1; // n_alive_policy_frames
                    }
                }
                entry.3 += reward;
                if !entry.4.contains(&current_row_group) {
                    entry.4.push(current_row_group);
                }

                // Track row group boundaries
                rows_remaining_in_rg -= 1;
                if rows_remaining_in_rg == 0 {
                    current_row_group += 1;
                    if current_row_group < metadata.num_row_groups() {
                        rows_remaining_in_rg = metadata.row_group(current_row_group).num_rows() as usize;
                    }
                }
            }
        }

        let index: HashMap<(i64, i64), ParquetEpisodeLocation> = accum.into_iter()
            .map(|(key, (frame_count, n_policy_frames, n_alive_policy_frames, total_reward, row_groups, discounted_sum))| {
                // If state.crashed column was missing, assume all policy frames alive
                let alive = if has_crashed_col { n_alive_policy_frames } else { n_policy_frames };
                let nz_return = (1.0 - DEFAULT_DISCOUNT) * discounted_sum;
                (key, ParquetEpisodeLocation {
                    row_groups,
                    frame_count,
                    n_policy_frames,
                    n_alive_policy_frames: alive,
                    total_reward: Some(total_reward),
                    nz_return: Some(nz_return),
                })
            })
            .collect();

        info!(
            "Parquet index: {} episodes, {} total frames",
            index.len(),
            index.values().map(|l| l.frame_count).sum::<usize>(),
        );

        Ok(Self { path, index, columns })
    }

    /// Get all unique (epoch, episode) keys, sorted.
    pub fn keys(&self) -> Vec<(i64, i64)> {
        let mut keys: Vec<_> = self.index.keys().copied().collect();
        keys.sort();
        keys
    }

    /// Get episode location info.
    pub fn get(&self, epoch: i64, episode: i64) -> Option<&ParquetEpisodeLocation> {
        self.index.get(&(epoch, episode))
    }

    /// Number of indexed episodes.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Total frame count across all episodes.
    pub fn total_frames(&self) -> usize {
        self.index.values().map(|l| l.frame_count).sum()
    }

    /// Discover action probability deed names from parquet columns.
    /// Looks for columns matching `p.*` and extracts the deed name.
    fn discover_action_prob_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.columns.iter()
            .filter_map(|c| {
                if c.starts_with("p.") {
                    let inner = &c[2..]; // strip "p."
                    if !inner.is_empty() {
                        return Some(inner.to_string());
                    }
                }
                None
            })
            .collect();
        names.sort();
        names
    }

    /// Load frames for a specific episode.
    pub fn load_episode(&self, epoch: i64, episode: i64, env_type: EnvType) -> Result<Vec<EsFrame>> {
        let t_total = std::time::Instant::now();

        let location = self.index.get(&(epoch, episode))
            .with_context(|| format!("Episode ({}, {}) not in parquet index", epoch, episode))?;

        info!(
            "load_episode({}, {}): row_groups={:?}, frame_count={}",
            epoch, episode, location.row_groups, location.frame_count
        );

        // Read only the row groups that contain this episode
        let t_open = std::time::Instant::now();
        let file = File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_row_groups(location.row_groups.clone());
        let reader = builder.build()?;
        info!("  file open + reader build: {:?}", t_open.elapsed());

        let mut frames = Vec::with_capacity(location.frame_count);

        let action_prob_names = self.discover_action_prob_names();

        let t_read = std::time::Instant::now();
        let mut total_rows_read = 0usize;
        for batch_result in reader {
            let batch = batch_result?;
            total_rows_read += batch.num_rows();
            let shared = Self::extract_shared_columns(&batch, &action_prob_names)?;
            let env_cols = env_type.extract_parquet_columns(&batch, &self.columns);

            for i in 0..shared.n_rows {
                let row_epoch = shared.epoch[i];
                let row_episode = shared.episode[i];
                if row_epoch != epoch || row_episode != episode {
                    continue;
                }

                let (ego, npcs) = env_cols.build_frame_state(i);

                let action_distribution = if !shared.action_probs.is_empty() {
                    let probs: Vec<(String, f64)> = shared.action_probs.iter()
                        .map(|(name, vals)| (name.clone(), vals[i]))
                        .collect();
                    Some(ActionDistribution { probs })
                } else {
                    None
                };

                let crash_reward = shared.crash_reward.as_ref().and_then(|v| v[i]);
                let crash_score = shared.crash_score.as_ref().and_then(|v| v[i]);

                frames.push(EsFrame {
                    epoch: row_epoch,
                    episode: row_episode,
                    t: shared.t[i],
                    reward: shared.reward.as_ref().and_then(|v| v[i]),
                    crash_reward,
                    crash_score,
                    action: shared.action.as_ref().and_then(|v| v[i].clone()),
                    action_name: shared.action_name.as_ref().and_then(|v| v[i].clone()),
                    state: FrameState {
                        crashed: shared.crashed[i],
                        ego,
                        npcs,
                        action_distribution,
                        chosen_action: None,
                        old_action_distribution: None,
                    },
                    v: shared.v.as_ref().and_then(|v| v[i]),
                    return_value: shared.return_value.as_ref().and_then(|v| v[i]),
                    tendency: shared.tendency.as_ref().and_then(|v| v[i]),
                    advantage: shared.advantage.as_ref().and_then(|v| v[i]),
                    nz_advantage: shared.nz_advantage.as_ref().and_then(|v| v[i]),
                });
            }
        }

        info!(
            "  parquet read: {:?} ({} rows read, {} frames kept)",
            t_read.elapsed(), total_rows_read, frames.len()
        );

        // Sort by fractional t
        frames.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        // Merge microbe data (tendency, advantage) from sibling microbes parquet if present.
        // Only fills in values not already provided by es parquet columns.
        let microbes_path = self.path.with_file_name("microbes.parquet");
        let microbes_path = if microbes_path.exists() {
            microbes_path
        } else {
            // Backward compat: old treks may use .pq extension
            self.path.with_file_name("microbes.pq")
        };
        if microbes_path.exists() {
            let t_microbes = std::time::Instant::now();
            if let Ok(microbe_data) = load_microbe_lookup(&microbes_path) {
                info!("  microbes load: {:?} ({} entries)", t_microbes.elapsed(), microbe_data.len());
                for frame in &mut frames {
                    if let Some(md) = microbe_data.get(&(frame.epoch, frame.t as i64, frame.episode)) {
                        if frame.tendency.is_none() {
                            frame.tendency = md.tendency;
                        }
                        if let Some(va) = md.vanilla_advantage {
                            frame.advantage = Some(va);
                        }
                        if frame.nz_advantage.is_none() {
                            frame.nz_advantage = md.nz_advantage;
                        }
                    }
                }
            }
        }

        info!("  load_episode total: {:?}", t_total.elapsed());
        Ok(frames)
    }

    /// Extract shared (env-agnostic) columns from a record batch.
    fn extract_shared_columns(
        batch: &arrow::record_batch::RecordBatch,
        action_prob_names: &[String],
    ) -> Result<SharedColumns> {
        let n = batch.num_rows();

        let action_probs: Vec<(String, Vec<f64>)> = action_prob_names.iter()
            .filter_map(|name| {
                let col_name = format!("p.{}", name);
                get_f64_opt(batch, &col_name).map(|vals| (name.clone(), vals))
            })
            .collect();

        Ok(SharedColumns {
            epoch: get_i64(batch, "epoch"),
            episode: get_i64(batch, "e"),
            t: get_f64(batch, "t"),
            reward: get_f64_nullable(batch, "reward"),
            action: get_string_opt(batch, "action"),
            action_name: get_string_opt(batch, "action_name"),
            crashed: get_bool(batch, "state.crashed"),
            action_probs,
            v: get_f64_nullable(batch, "v"),
            return_value: get_f64_nullable(batch, "return"),
            advantage: get_f64_nullable(batch, "advantage"),
            nz_advantage: get_f64_nullable(batch, "nz_advantage"),
            tendency: get_f64_nullable(batch, "tendency"),
            crash_reward: get_f64_nullable(batch, "crash_reward"),
            crash_score: get_f64_nullable(batch, "crash_score"),
            n_rows: n,
        })
    }
}

/// Microbe data loaded from a sibling microbes parquet file.
struct MicrobeData {
    tendency: Option<f64>,
    vanilla_advantage: Option<f64>,
    nz_advantage: Option<f64>,
}

/// Load microbe data from a microbes parquet file into a lookup table.
/// Key: (epoch, t, episode).
fn load_microbe_lookup(path: &Path) -> Result<HashMap<(i64, i64, i64), MicrobeData>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut lookup = HashMap::new();

    for batch_result in reader {
        let batch = batch_result?;
        let n = batch.num_rows();

        let epochs: Vec<i64> = batch.column_by_name("epoch")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .map(|a| (0..a.len()).map(|i| a.value(i)).collect())
            .unwrap_or_default();
        let ts: Vec<i64> = batch.column_by_name("t")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .map(|a| (0..a.len()).map(|i| a.value(i)).collect())
            .unwrap_or_default();
        let es: Vec<i64> = batch.column_by_name("e")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .map(|a| (0..a.len()).map(|i| a.value(i)).collect())
            .unwrap_or_default();

        let tendency_col: Option<Vec<f64>> = batch.column_by_name("tendency")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .map(|a| (0..a.len()).map(|i| a.value(i)).collect());
        let vanilla_adv_col: Option<Vec<f64>> = batch.column_by_name("vanilla_advantage")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .map(|a| (0..a.len()).map(|i| a.value(i)).collect());
        let norm_adv_col: Option<Vec<f64>> = batch.column_by_name("nz_advantage")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .map(|a| (0..a.len()).map(|i| a.value(i)).collect());

        for i in 0..n {
            if i < epochs.len() && i < ts.len() && i < es.len() {
                lookup.insert(
                    (epochs[i], ts[i], es[i]),
                    MicrobeData {
                        tendency: tendency_col.as_ref().map(|v| v[i]),
                        vanilla_advantage: vanilla_adv_col.as_ref().map(|v| v[i]),
                        nz_advantage: norm_adv_col.as_ref().map(|v| v[i]),
                    },
                );
            }
        }
    }

    Ok(lookup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_action_prob_names() {
        let index = EsParquetIndex {
            path: PathBuf::from("/tmp/test.parquet"),
            index: HashMap::new(),
            columns: vec![
                "epoch".into(), "e".into(), "t".into(),
                "p.faster".into(), "p.idle".into(), "p.left".into(),
                "p.right".into(), "p.slower".into(),
                "reward".into(), "state.crashed".into(),
            ],
        };
        let names = index.discover_action_prob_names();
        assert_eq!(names, vec!["faster", "idle", "left", "right", "slower"]);
    }

    #[test]
    fn test_discover_action_prob_names_empty() {
        let index = EsParquetIndex {
            path: PathBuf::from("/tmp/test.parquet"),
            index: HashMap::new(),
            columns: vec!["epoch".into(), "e".into(), "t".into()],
        };
        assert!(index.discover_action_prob_names().is_empty());
    }

    #[test]
    fn test_count_npcs() {
        use crate::envs::highway::HighwayParquetColumns;
        let columns: Vec<String> = vec![
            "state.npc0_x".into(), "state.npc0_y".into(),
            "state.npc1_x".into(), "state.npc1_y".into(),
            "state.npc2_x".into(), "state.npc2_y".into(),
        ];
        assert_eq!(HighwayParquetColumns::count_npcs(&columns), 3);
    }

    #[test]
    fn test_count_npcs_zero() {
        use crate::envs::highway::HighwayParquetColumns;
        let columns: Vec<String> = vec!["epoch".into(), "state.ego_x".into()];
        assert_eq!(HighwayParquetColumns::count_npcs(&columns), 0);
    }
}
