//! Trek directory discovery and parsing.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use super::episode::EpisodeMeta;
use super::es_parquet::EsParquetIndex;
use crate::util::{find_parquet_file, posh_path};

/// Parsed fields from meta.yaml.
struct ParsedMeta {
    ego_speed_range: Option<(f64, f64)>,
    seconds_per_t: f64,
    seconds_per_sub_t: f64,
    n_lanes: Option<usize>,
    env_type: Option<crate::envs::EnvType>,
}

impl Default for ParsedMeta {
    fn default() -> Self {
        Self { ego_speed_range: None, seconds_per_t: 0.1, seconds_per_sub_t: 0.1, n_lanes: None, env_type: None }
    }
}

/// Build `Epoch` structs from a parquet index.
pub fn epochs_from_parquet_index(
    index: &EsParquetIndex,
    parquet_path: &Path,
    trek_path: &Path,
) -> Vec<Epoch> {
    let mut epoch_map: HashMap<i64, Vec<(i64, usize, usize, usize, Option<f64>, Option<f64>)>> =
        HashMap::new();
    for (epoch, episode) in index.keys() {
        if let Some(loc) = index.get(epoch, episode) {
            epoch_map
                .entry(epoch)
                .or_default()
                .push((episode, loc.frame_count, loc.n_policy_frames,
                       loc.n_alive_policy_frames,
                       loc.total_reward, loc.nz_return));
        }
    }

    for episodes in epoch_map.values_mut() {
        episodes.sort_by_key(|(e, _, _, _, _, _)| *e);
    }

    let mut epoch_indices: Vec<i64> = epoch_map.keys().copied().collect();
    epoch_indices.sort();

    epoch_indices
        .iter()
        .enumerate()
        .map(|(idx, &epoch_num)| {
            let episodes_data = epoch_map.get(&epoch_num).unwrap();
            let episodes: Vec<EpisodeMeta> = episodes_data
                .iter()
                .enumerate()
                .map(|(ep_idx, &(orig_episode, frame_count, n_policy_frames, n_alive_policy_frames, total_reward, nz_return))| {
                    EpisodeMeta {
                        index: ep_idx,
                        path: parquet_path.to_path_buf(),
                        n_frames: frame_count,
                        n_policy_frames,
                        n_alive_policy_frames,
                        total_reward,
                        nz_return,
                        es_epoch: Some(epoch_num),
                        es_episode: Some(orig_episode),
                    }
                })
                .collect();

            Epoch::new(idx, epoch_num, trek_path.to_path_buf(), episodes)
        })
        .collect()
}

/// An epoch containing multiple episodes.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Epoch {
    /// Epoch index (0-based position in the epochs vector).
    pub index: usize,
    /// The actual epoch number from the data (e.g. 1 for epoch_001 or epoch=1 in sample_es.parquet).
    pub epoch_number: i64,
    /// Path to the epoch directory.
    pub path: PathBuf,
    /// Metadata for all episodes in this epoch.
    pub episodes: Vec<EpisodeMeta>,
    /// Normalized return from epochia.parquet (across ALL training episodes, not just sampled).
    pub epochia_nz_return: Option<f64>,
    /// Alive fraction from epochia.parquet (across ALL training episodes, not just sampled).
    pub epochia_alive_fraction: Option<f64>,
}

impl Epoch {
    pub fn new(index: usize, epoch_number: i64, path: PathBuf, episodes: Vec<EpisodeMeta>) -> Self {
        Self { index, epoch_number, path, episodes, epochia_nz_return: None, epochia_alive_fraction: None }
    }

    /// Number of episodes in this epoch.
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Mean normalized discounted return across all episodes.
    pub fn mean_nreturn(&self) -> Option<f64> {
        let values: Vec<f64> = self.episodes.iter()
            .filter_map(|ep| ep.nz_return)
            .collect();
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }

}

/// A trek containing multiple epochs.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Trek {
    /// Path to the trek directory.
    pub path: PathBuf,
    /// All epochs in this trek.
    pub epochs: Vec<Epoch>,
    /// Index for efficient random access to sample_es parquet episodes.
    /// Preferred over es_index when both exist.
    pub es_parquet_index: Option<EsParquetIndex>,
    /// NPC speed range from meta.yaml (min, max) in m/s.
    /// For highway envs this comes from npc_speed_min/max.
    pub ego_speed_range: Option<(f64, f64)>,
    /// Policy timestep duration in seconds, from meta.yaml `seconds_per_t`.
    pub seconds_per_t: f64,
    /// Sub-timestep duration in seconds, from meta.yaml `seconds_per_sub_t`.
    /// Equals `seconds_per_t` when no sub-steps exist.
    pub seconds_per_sub_t: f64,
    /// Number of sub-steps per policy timestep (1 when no sub-steps).
    pub n_sub_ts_per_t: usize,
    /// Set of epoch numbers that have saved snapshots in snapshots.sqlite.
    pub snapshot_epochs: std::collections::HashSet<i64>,
    /// Number of lanes from meta.yaml (overrides config default when set).
    pub n_lanes: Option<usize>,
    /// Max policy timesteps per episode (for survival fraction denominator).
    pub n_ts_per_e: usize,
    /// Environment type detected from meta.yaml. None if env not recognized.
    pub env_type: Option<crate::envs::EnvType>,
    /// Error from loading (e.g. corrupt parquet). Shown in the UI.
    pub load_error: Option<String>,
}

impl Trek {
    /// Create an empty Trek with no epochs or data.
    /// Used when a Trek is needed for dispatch but no actual trek is loaded.
    pub fn empty() -> Self {
        Self {
            path: PathBuf::new(),
            epochs: Vec::new(),
            es_parquet_index: None,
            ego_speed_range: None,
            seconds_per_t: 0.1,
            seconds_per_sub_t: 0.1,
            n_sub_ts_per_t: 1,
            snapshot_epochs: std::collections::HashSet::new(),
            n_ts_per_e: 1,
            n_lanes: None,
            env_type: None,
            load_error: None,
        }
    }

    /// Load a trek from a directory path.
    ///
    /// First tries to load from sample_es parquet (highjax trek format, .parquet).
    /// Falls back to discovering epoch_NNN directories.
    pub fn load(path: PathBuf) -> Result<Self> {
        info!("Loading trek from {}", posh_path(&path));

        if !path.exists() {
            anyhow::bail!("Trek path does not exist: {}", posh_path(&path));
        }

        if !path.is_dir() {
            anyhow::bail!("Trek path is not a directory: {}", posh_path(&path));
        }

        // Try to load from sample_es parquet first (preferred format)
        if let Some(es_parquet_path) = find_parquet_file(&path, "sample_es") {
            info!("Found {}, loading parquet format", es_parquet_path.file_name().unwrap().to_string_lossy());
            match Self::load_from_es_parquet(path.clone(), es_parquet_path) {
                Ok(trek) => return Ok(trek),
                Err(e) => {
                    tracing::warn!("Failed to load parquet (corrupt?): {}", e);
                    // Fall through to load a minimal trek with no episodes
                    let meta = Self::parse_meta(&path);
                    let snapshot_epochs = Self::discover_snapshot_epochs(&path);
                    let n_sub_ts_per_t = (meta.seconds_per_t / meta.seconds_per_sub_t).round() as usize;
                    return Ok(Self {
                        path,
                        epochs: Vec::new(),
                        es_parquet_index: None,
                        ego_speed_range: meta.ego_speed_range,
                        seconds_per_t: meta.seconds_per_t,
                        seconds_per_sub_t: meta.seconds_per_sub_t,
                        n_sub_ts_per_t,
                        snapshot_epochs,
                        n_ts_per_e: 1,
                        n_lanes: meta.n_lanes,
                        env_type: meta.env_type,
                        load_error: Some(format!("{}", e)),
                    });
                }
            }
        }

        // Fall back to epoch_NNN directory format
        let mut epochs = Vec::new();

        // Find all epoch_NNN directories
        let mut epoch_dirs: Vec<_> = fs::read_dir(&path)
            .with_context(|| format!("Failed to read trek directory: {:?}", path))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with("epoch_")
            })
            .collect();

        // Sort by epoch number
        epoch_dirs.sort_by_key(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .strip_prefix("epoch_")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
        });

        for (epoch_idx, epoch_entry) in epoch_dirs.into_iter().enumerate() {
            let epoch_path = epoch_entry.path();
            let epoch_number = epoch_entry
                .file_name()
                .to_string_lossy()
                .strip_prefix("epoch_")
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(epoch_idx as i64);
            let episodes = Self::load_epoch_episodes(&epoch_path)?;

            epochs.push(Epoch::new(epoch_idx, epoch_number, epoch_path, episodes));
        }

        info!("Found {} epochs", epochs.len());
        let meta = Self::parse_meta(&path);
        let n_sub_ts_per_t = (meta.seconds_per_t / meta.seconds_per_sub_t).round() as usize;

        let snapshot_epochs = Self::discover_snapshot_epochs(&path);

        let n_ts_per_e = epochs.iter()
            .flat_map(|e| e.episodes.iter())
            .map(|ep| ep.n_policy_frames)
            .max()
            .unwrap_or(1);

        Ok(Self {
            path,
            epochs,
            es_parquet_index: None,
            ego_speed_range: meta.ego_speed_range,
            seconds_per_t: meta.seconds_per_t,
            seconds_per_sub_t: meta.seconds_per_sub_t,
            n_sub_ts_per_t,
            snapshot_epochs,
            n_ts_per_e,
            n_lanes: meta.n_lanes,
            env_type: meta.env_type,
            load_error: None,
        })
    }

    /// Load trek from sample_es parquet file.
    fn load_from_es_parquet(path: PathBuf, es_parquet_path: PathBuf) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let index = EsParquetIndex::build(&es_parquet_path)?;
        info!("Parquet index built in {:.1}s", t0.elapsed().as_secs_f64());

        // Group by epoch
        let mut epochs = epochs_from_parquet_index(&index, &es_parquet_path, &path);

        // Load epochia data and merge into epochs
        Self::load_epochia(&path, &mut epochs);

        info!(
            "Built parquet index for {} epochs with {} total frames",
            epochs.len(),
            index.total_frames(),
        );

        info!("Parsing meta.yaml");
        let meta = Self::parse_meta(&path);
        let n_sub_ts_per_t = (meta.seconds_per_t / meta.seconds_per_sub_t).round() as usize;

        info!("Discovering snapshot epochs");
        let snapshot_epochs = Self::discover_snapshot_epochs(&path);

        let n_ts_per_e = epochs.iter()
            .flat_map(|e| e.episodes.iter())
            .map(|ep| ep.n_policy_frames)
            .max()
            .unwrap_or(1);

        Ok(Self {
            path,
            epochs,
            es_parquet_index: Some(index),
            ego_speed_range: meta.ego_speed_range,
            seconds_per_t: meta.seconds_per_t,
            seconds_per_sub_t: meta.seconds_per_sub_t,
            n_sub_ts_per_t,
            snapshot_epochs,
            n_ts_per_e,
            n_lanes: meta.n_lanes,
            env_type: meta.env_type,
            load_error: None,
        })
    }

    /// Load epochia.parquet and merge nz_return + alive_fraction into epochs.
    fn load_epochia(trek_path: &PathBuf, epochs: &mut Vec<Epoch>) {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use arrow::array::{Float64Array, Int64Array};

        let epochia_path = find_parquet_file(trek_path, "epochia");
        let epochia_path = match epochia_path {
            Some(p) => p,
            None => return,
        };

        let file = match std::fs::File::open(&epochia_path) {
            Ok(f) => f,
            Err(_) => return,
        };
        let reader = match ParquetRecordBatchReaderBuilder::try_new(file) {
            Ok(b) => match b.build() {
                Ok(r) => r,
                Err(_) => return,
            },
            Err(_) => return,
        };

        // Build a map from epoch_number -> (nz_return, alive_fraction)
        let mut epochia_map: std::collections::HashMap<i64, (f64, f64)> = std::collections::HashMap::new();
        for batch in reader {
            let batch = match batch {
                Ok(b) => b,
                Err(_) => continue,
            };
            let epoch_col = match batch.column_by_name("epoch") {
                Some(c) => match c.as_any().downcast_ref::<Int64Array>() {
                    Some(a) => a,
                    None => continue,
                },
                None => continue,
            };
            let nz_return_col = batch.column_by_name("nz_return")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let alive_col = batch.column_by_name("alive_fraction")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            for i in 0..batch.num_rows() {
                let epoch_num = epoch_col.value(i);
                let nz_return = nz_return_col.map(|c| c.value(i)).unwrap_or(0.0);
                let alive_frac = alive_col.map(|c| c.value(i)).unwrap_or(1.0);
                epochia_map.insert(epoch_num, (nz_return, alive_frac));
            }
        }

        // Merge into epochs
        for epoch in epochs.iter_mut() {
            if let Some(&(nz_return, alive_frac)) = epochia_map.get(&epoch.epoch_number) {
                epoch.epochia_nz_return = Some(nz_return);
                epoch.epochia_alive_fraction = Some(alive_frac);
            }
        }
    }

    /// Load episode metadata from an epoch directory.
    fn load_epoch_episodes(epoch_path: &PathBuf) -> Result<Vec<EpisodeMeta>> {
        let mut episodes = Vec::new();

        // Find all episode_NNN.json files
        let mut episode_files: Vec<_> = fs::read_dir(epoch_path)
            .with_context(|| format!("Failed to read epoch directory: {:?}", epoch_path))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("episode_") && name_str.ends_with(".json")
            })
            .collect();

        // Sort by episode number
        episode_files.sort_by_key(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .strip_prefix("episode_")
                .and_then(|s| s.strip_suffix(".json"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
        });

        for (ep_idx, ep_entry) in episode_files.into_iter().enumerate() {
            let ep_path = ep_entry.path();
            match EpisodeMeta::from_path(ep_path.clone(), ep_idx) {
                Ok(meta) => episodes.push(meta),
                Err(e) => {
                    tracing::warn!("Failed to load episode {:?}: {}", ep_path, e);
                }
            }
        }

        Ok(episodes)
    }

    /// Read and parse meta.yaml once, returning the parsed YAML document.
    fn read_meta_yaml(path: &Path) -> Option<serde_yaml::Value> {
        let meta_path = path.join("meta.yaml");
        let content = fs::read_to_string(&meta_path).ok()?;
        serde_yaml::from_str(&content).ok()
    }

    /// Extract a named f64 field from a parsed meta.yaml commands section.
    fn meta_f64(doc: &serde_yaml::Value, field_name: &str) -> Option<f64> {
        let commands = doc.get("commands")?.as_mapping()?;
        for (_key, cmd_value) in commands {
            let cmd = cmd_value.as_mapping()?;
            if let Some(val) = cmd.get(serde_yaml::Value::String(field_name.into())) {
                if let Some(v) = val.as_f64() {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Parse all needed fields from meta.yaml in a single read.
    fn parse_meta(path: &Path) -> ParsedMeta {
        let doc = Self::read_meta_yaml(path);
        match doc {
            Some(ref doc) => {
                let ego_speed_range = Self::meta_f64(doc, "npc_speed_min")
                    .or_else(|| Self::meta_f64(doc, "ego_target_speed_min"))
                    .and_then(|min_v| {
                        Self::meta_f64(doc, "npc_speed_max")
                            .or_else(|| Self::meta_f64(doc, "ego_target_speed_max"))
                            .map(|max_v| (min_v, max_v))
                    });
                if let Some((min_v, max_v)) = ego_speed_range {
                    debug!("Found ego speed range in meta.yaml: {}-{} m/s", min_v, max_v);
                }
                let seconds_per_t = Self::meta_f64(doc, "seconds_per_t").unwrap_or(0.1);
                let seconds_per_sub_t = Self::meta_f64(doc, "seconds_per_sub_t")
                    .unwrap_or(seconds_per_t);
                let n_lanes = Self::meta_f64(doc, "n_lanes").map(|v| v as usize);
                let env_type = crate::envs::EnvType::detect_from_meta_yaml(doc);
                ParsedMeta { ego_speed_range, seconds_per_t, seconds_per_sub_t, n_lanes, env_type }
            }
            None => ParsedMeta::default(),
        }
    }

    /// Discover which epochs have snapshots in snapshots.sqlite.
    fn discover_snapshot_epochs(path: &Path) -> HashSet<i64> {
        let db_path = path.join("snapshots.sqlite");
        if !db_path.exists() {
            return HashSet::new();
        }
        let Ok(conn) = rusqlite::Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ) else {
            return HashSet::new();
        };
        fn table_exists(conn: &rusqlite::Connection, name: &str) -> bool {
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [name], |row| row.get::<_, i64>(0),
            ).map_or(false, |count| count > 0)
        }

        let mut epochs = HashSet::new();
        // Check if genesis table exists and has an entry (epoch 0)
        if table_exists(&conn, "genesis") {
            if let Ok(n) = conn.query_row("SELECT COUNT(*) FROM genesis", [], |row| row.get::<_, i64>(0)) {
                if n > 0 {
                    epochs.insert(0);
                }
            }
        }
        // Get all joint_ascent epochs
        if table_exists(&conn, "joint_ascent") {
            if let Ok(mut stmt) = conn.prepare("SELECT epoch FROM joint_ascent ORDER BY epoch") {
                if let Ok(rows) = stmt.query_map([], |row| row.get::<_, i64>(0)) {
                    for epoch in rows.flatten() {
                        epochs.insert(epoch);
                    }
                }
            }
        }
        if !epochs.is_empty() {
            debug!("Found {} snapshot epochs", epochs.len());
        }
        epochs
    }

    /// Total number of epochs.
    pub fn epoch_count(&self) -> usize {
        self.epochs.len()
    }

    /// Total number of episodes across all epochs.
    pub fn episode_count(&self) -> usize {
        self.epochs.iter().map(|e| e.episode_count()).sum()
    }
}

/// A parquet source discovered in a trek directory.
#[derive(Debug, Clone)]
pub struct ParquetSource {
    /// Full path to the parquet file.
    pub path: PathBuf,
    /// Short display name (e.g. "sample_es" or breakdown run name) for UI.
    pub display: String,
    /// Path relative to trek root (e.g. "sample_es.parquet" or
    /// "breakdown/2026-02-20-13-07-24_a0_e100-100_mm500_jfj/es.parquet").
    /// Used for `-t` CLI matching when target is a parquet file.
    pub relative_path: String,
}

/// Discover all es parquet sources in a trek directory.
/// Returns `sample_es` first (if it exists), then any `breakdown/*/es` files
/// sorted alphabetically by run folder name.
/// Looks for `.parquet` extension (falls back to `.pq` for old treks).
pub fn discover_parquet_sources(trek_path: &Path) -> Vec<ParquetSource> {
    let mut sources = Vec::new();

    // 1. sample_es (.parquet, fallback .pq)
    if let Some(sample_path) = find_parquet_file(trek_path, "sample_es") {
        let filename = sample_path.file_name().unwrap().to_string_lossy().to_string();
        sources.push(ParquetSource {
            path: sample_path,
            display: "sample_es".to_string(),
            relative_path: filename,
        });
    }

    // 2. breakdown/*/es (.parquet, fallback .pq)
    let breakdown_dir = trek_path.join("breakdown");
    if breakdown_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&breakdown_dir) {
            let mut bd_entries: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                        && find_parquet_file(&e.path(), "es").is_some()
                })
                .collect();
            bd_entries.sort_by_key(|e| e.file_name());

            for entry in bd_entries {
                let run_name = entry.file_name().to_string_lossy().to_string();
                if let Some(es_path) = find_parquet_file(&entry.path(), "es") {
                    let es_filename = es_path.file_name().unwrap().to_string_lossy().to_string();
                    let rel = format!("breakdown/{}/{}", run_name, es_filename);
                    sources.push(ParquetSource {
                        path: es_path,
                        display: run_name,
                        relative_path: rel,
                    });
                }
            }
        }
    }

    sources
}

/// An entry in the treks list (for the Treks pane).
#[derive(Debug, Clone)]
pub struct TrekEntry {
    /// Full path to the trek directory.
    pub path: PathBuf,
    /// Poshed display name.
    pub display: String,
}

/// Discover all trek directories in ~/.highjax/t/,
/// sorted by directory name (newest last).
/// Returns entries with batch-poshed display names.
pub fn discover_treks() -> Vec<TrekEntry> {
    let home = match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h),
        Err(_) => return Vec::new(),
    };

    let mut dirs_to_scan = Vec::new();

    // ~/.highjax/t (or $HIGHJAX_HOME/t override)
    if let Ok(highjax_home) = std::env::var("HIGHJAX_HOME") {
        dirs_to_scan.push(PathBuf::from(highjax_home).join("t"));
    } else {
        dirs_to_scan.push(home.join(".highjax").join("t"));
    }

    // Check name first (no I/O) to avoid slow stat calls over network mounts.
    let is_trek_dir = |entry: &fs::DirEntry| -> bool {
        // Use file_type() which avoids extra stat on Linux (uses d_type from readdir)
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            return false;
        }
        // Name starts with digit (timestamp format) — no I/O needed
        if entry.file_name()
            .to_str()
            .map(|s| s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
            .unwrap_or(false)
        {
            return true;
        }
        // Fall back to checking for trek markers (requires stat calls)
        let path = entry.path();
        path.join("meta.yaml").exists() || find_parquet_file(&path, "sample_es").is_some()
    };

    let mut paths: Vec<PathBuf> = Vec::new();
    for dir in &dirs_to_scan {
        if !dir.is_dir() {
            continue;
        }
        paths.extend(
            fs::read_dir(dir)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .filter(&is_trek_dir)
                .map(|e| e.path()),
        );
    }

    info!("Found {} trek directories", paths.len());

    // Sort by directory name (timestamp format).
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    // Batch posh all paths
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
    let displays = crate::util::posh_paths(&path_refs);

    paths
        .into_iter()
        .zip(displays)
        .map(|(path, display)| TrekEntry { path, display })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_epoch_mean_nreturn() {
        let epoch = Epoch::new(0, 0, PathBuf::from("/tmp/epoch_000"), vec![
            EpisodeMeta {
                index: 0,
                path: PathBuf::from("/tmp/ep0.json"),
                n_frames: 10,
                n_policy_frames: 10,
                n_alive_policy_frames: 8,
                total_reward: Some(5.0),
                nz_return: Some(0.3),
                es_epoch: None,
                es_episode: None,
            },
            EpisodeMeta {
                index: 1,
                path: PathBuf::from("/tmp/ep1.json"),
                n_frames: 20,
                n_policy_frames: 20,
                n_alive_policy_frames: 20,
                total_reward: Some(10.0),
                nz_return: Some(0.5),
                es_epoch: None,
                es_episode: None,
            },
        ]);

        assert_eq!(epoch.mean_nreturn(), Some(0.4));
        assert_eq!(epoch.episode_count(), 2);
    }

    #[test]
    fn test_epoch_empty() {
        let epoch = Epoch::new(0, 0, PathBuf::from("/tmp/epoch_000"), vec![]);

        assert_eq!(epoch.mean_nreturn(), None);
        assert_eq!(epoch.episode_count(), 0);
    }

    #[test]
    fn test_trek_nonexistent_path() {
        let result = Trek::load(PathBuf::from("/nonexistent/path/that/does/not/exist"));
        assert!(result.is_err());
    }

    #[test]
    fn test_trek_file_not_directory() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("not_a_dir.txt");
        fs::write(&file_path, "hello").unwrap();

        let result = Trek::load(file_path);
        assert!(result.is_err());
    }

    fn write_test_meta_yaml(dir: &Path) {
        fs::write(
            dir.join("meta.yaml"),
            "commands:\n  2.highway:\n    seconds_per_t: 0.1\n",
        ).unwrap();
    }

    #[test]
    fn test_trek_n_lanes_from_meta() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("meta.yaml"),
            "commands:\n  1.highway:\n    n_lanes: 2\n    seconds_per_t: 0.5\n",
        ).unwrap();
        let trek = Trek::load(dir.path().to_path_buf()).unwrap();
        assert_eq!(trek.n_lanes, Some(2));
    }

    #[test]
    fn test_trek_n_lanes_default_when_missing() {
        let dir = TempDir::new().unwrap();
        write_test_meta_yaml(dir.path());
        let trek = Trek::load(dir.path().to_path_buf()).unwrap();
        assert_eq!(trek.n_lanes, None);
    }

    #[test]
    fn test_trek_empty_directory() {
        let dir = TempDir::new().unwrap();
        write_test_meta_yaml(dir.path());
        let trek = Trek::load(dir.path().to_path_buf()).unwrap();

        assert_eq!(trek.epoch_count(), 0);
        assert_eq!(trek.episode_count(), 0);
    }

    fn create_episode_json(frames: usize, reward_per_frame: f64) -> String {
        let observations: Vec<String> = (0..frames)
            .map(|_| "[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]".to_string())
            .collect();
        let actions: Vec<String> = (0..frames).map(|_| "0".to_string()).collect();
        let rewards: Vec<String> = (0..frames)
            .map(|_| format!("{}", reward_per_frame))
            .collect();
        let dones: Vec<String> = (0..frames)
            .enumerate()
            .map(|(i, _)| {
                if i == frames - 1 {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            })
            .collect();

        format!(
            r#"{{"observations": [{}], "actions": [{}], "rewards": [{}], "dones": [{}]}}"#,
            observations.join(","),
            actions.join(","),
            rewards.join(","),
            dones.join(",")
        )
    }

    #[test]
    fn test_trek_single_epoch_single_episode() {
        let dir = TempDir::new().unwrap();
        write_test_meta_yaml(dir.path());
        let epoch_dir = dir.path().join("epoch_000");
        fs::create_dir(&epoch_dir).unwrap();

        let ep_path = epoch_dir.join("episode_000.json");
        let mut file = fs::File::create(&ep_path).unwrap();
        file.write_all(create_episode_json(5, 1.0).as_bytes())
            .unwrap();

        let trek = Trek::load(dir.path().to_path_buf()).unwrap();

        assert_eq!(trek.epoch_count(), 1);
        assert_eq!(trek.episode_count(), 1);
        assert_eq!(trek.epochs[0].episodes[0].n_frames, 5);
        assert_eq!(trek.epochs[0].episodes[0].total_reward, Some(5.0));
    }

    #[test]
    fn test_trek_multiple_epochs() {
        let dir = TempDir::new().unwrap();
        write_test_meta_yaml(dir.path());

        for epoch_idx in 0..3 {
            let epoch_dir = dir.path().join(format!("epoch_{:03}", epoch_idx));
            fs::create_dir(&epoch_dir).unwrap();

            for ep_idx in 0..2 {
                let ep_path = epoch_dir.join(format!("episode_{:03}.json", ep_idx));
                let mut file = fs::File::create(&ep_path).unwrap();
                file.write_all(create_episode_json(10, 0.5).as_bytes())
                    .unwrap();
            }
        }

        let trek = Trek::load(dir.path().to_path_buf()).unwrap();

        assert_eq!(trek.epoch_count(), 3);
        assert_eq!(trek.episode_count(), 6);

        for epoch in &trek.epochs {
            assert_eq!(epoch.episode_count(), 2);
            assert_eq!(epoch.mean_nreturn(), None); // JSON episodes lack nz_return
        }
    }

    #[test]
    fn test_trek_ignores_non_epoch_dirs() {
        let dir = TempDir::new().unwrap();
        write_test_meta_yaml(dir.path());

        // Create valid epoch
        let epoch_dir = dir.path().join("epoch_000");
        fs::create_dir(&epoch_dir).unwrap();
        let ep_path = epoch_dir.join("episode_000.json");
        fs::write(&ep_path, create_episode_json(3, 1.0)).unwrap();

        // Create non-epoch directories that should be ignored
        fs::create_dir(dir.path().join("other_folder")).unwrap();
        fs::create_dir(dir.path().join("not_an_epoch")).unwrap();
        fs::write(dir.path().join("some_file.txt"), "hello").unwrap();

        let trek = Trek::load(dir.path().to_path_buf()).unwrap();

        assert_eq!(trek.epoch_count(), 1);
    }

    #[test]
    fn test_trek_epoch_ordering() {
        let dir = TempDir::new().unwrap();
        write_test_meta_yaml(dir.path());

        // Create epochs out of order
        for epoch_idx in [5, 2, 8, 0, 3] {
            let epoch_dir = dir.path().join(format!("epoch_{:03}", epoch_idx));
            fs::create_dir(&epoch_dir).unwrap();
            let ep_path = epoch_dir.join("episode_000.json");
            fs::write(&ep_path, create_episode_json(1, epoch_idx as f64)).unwrap();
        }

        let trek = Trek::load(dir.path().to_path_buf()).unwrap();

        assert_eq!(trek.epoch_count(), 5);
        // Epochs should be sorted by index
        assert_eq!(trek.epochs[0].episodes[0].total_reward, Some(0.0));
        assert_eq!(trek.epochs[1].episodes[0].total_reward, Some(2.0));
        assert_eq!(trek.epochs[2].episodes[0].total_reward, Some(3.0));
        assert_eq!(trek.epochs[3].episodes[0].total_reward, Some(5.0));
        assert_eq!(trek.epochs[4].episodes[0].total_reward, Some(8.0));
    }
}
