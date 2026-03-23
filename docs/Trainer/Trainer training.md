# Trainer Training Pipeline

Related: [[Trainer behaviors]] | [[HighJax docs]] | [[Octane docs]]

PPO training for the HighJax highway environment. Multi-agent architecture (1 agent in practice), fully JIT-compiled.

## CLI

```bash
highjax-trainer train --n-epochs 300 --n-es 400 --n-ts 40
```

All CLI flags map directly to `AgentConfig` fields or training loop parameters. Run `highjax-trainer train --help` for the full list.

## Training loop

The training loop follows a Genesis → JointAscent pattern:

1. **Population** — Create a `Population` wrapping one `Brain` per agent (typically 1). Each brain has an actor lobe (attention-based policy network) and a critic lobe (value network).

2. **Genesis** (epoch 0) — Collect an initial rollout with the random policy. The rollout stores per-agent observations, actions, rewards, values, and probabilities, along with agent-agnostic done/crashed flags. With `--replay-buffer-factor N`, the genesis rollout collects `N * n_es` episodes.

3. **JointAscent** (epochs 1+) — Each epoch:
   - Compute per-agent `Ascent` from the current rollout:
     - Run `n_critic_iterations` of critic gradient descent on value loss
     - Recompute value estimates with updated critic, then compute and normalize GAE advantages
     - Flatten all (episode, timestep) data, shuffle, split into minibatches
     - For each of `n_sweeps_per_epoch` sweeps: iterate over minibatches computing the PPO clipped surrogate + entropy objective, apply actor gradient ascent
     - Optionally adjust step size via KLD binary search
   - Wrap results in a `JointAscent` containing old and new brains, loss, KLD, vanilla objective
   - Collect new rollout with updated population; extend replay buffer via `roll_and_extend`
   - `Evaluator` computes per-agent coeval reward from the rollout
   - `EpochReporter` writes enriched metrics to `epochia.pq` and sampled episode data to `sample_es.pq`

4. **Replay buffer** — When `--replay-buffer-factor` > 1, multiple epochs of episodes are retained. Each epoch, new episodes are appended and old ones roll off. The `tail` property filters to the most recent epoch for evaluation/reporting.

Notes on the training loop internals:

- The vital mask is derived from `crashed_by_e_by_t`, not `done_by_e_by_t` — only crashes kill vitality, not timeouts. Episodes that survive to the end correctly bootstrap their terminal value via `epilogue_vital_by_e`.
- `Genesis.create` splits its seed into `(rollout_seed, evaluation_seed, next_seed)` via a 3-split. Then `Genesis.create_first_joint_ascent` splits `self.next_seed` into `(evaluation_seed, next_seed)` via a 2-split before creating the first JointAscent. So the seed chain is: genesis 3-split → extra 2-split → then each epoch does a 3-split of `next_seed`.

## Ego-attention estimator

The default estimator (`AttentionActorEstimator` / `AttentionCriticEstimator` in `highjax_trainer/attention_estimating.py`) uses multi-head attention rather than a flat MLP. It handles variable NPC counts naturally via presence masking and produces interpretable attention weights.

Architecture:

1. **Ego embedding**: MLP maps ego features (row 0 of the observation) to a dense vector. Default: 2 layers of 64 units with ReLU.
2. **Other embedding**: Separate MLP maps each NPC's features to dense vectors. Same architecture as ego embedding.
3. **Attention layers**: Ego embedding as query, all vehicle embeddings (ego + NPCs) as keys/values. Multi-head scaled dot-product attention. Default: 2 layers, 4 heads, feature size 64.
4. **Presence masking**: Vehicles with presence < 0.5 get attention scores of -1e9, effectively zeroing their contribution.
5. **Residual connection**: Attention output is linearly projected and averaged 50/50 with the ego embedding.
6. **Output MLP**: Intermediate ReLU layers (actor default 2x64, critic default 2x192) before the final projection. The actor uses `end_layers` (a tuple of Dense layers, currently 1 element) that project to action logits. The critic uses `end_layer` (a single Dense(1)) for the scalar value estimate.

The `forward_with_attention` method returns both logits and attention weights, used for writing `state.npc{i}_attention` columns to sample_es.

## AgentConfig

All PPO hyperparameters live in `AgentConfig`:

| Field | CLI flag | Default | Description |
|-------|----------|---------|-------------|
| `actor_lr` | `--actor-lr` | 3e-4 | Actor learning rate |
| `critic_lr` | `--critic-lr` | 3e-3 | Critic learning rate |
| `critic_lr_end` | | None | End learning rate for critic linear schedule |
| `optimizer_string` | | 'adam' | Optimizer name with optional kwargs, e.g. 'adam(b1=0.9)' |
| `estimator_string` | | '\<default\>' | Estimator selection string |
| `discount` | `--discount` | 0.95 | Discount factor (gamma) |
| `vanilla_gae_lambda` | | 0.9 | GAE lambda for advantage estimation |
| `critic_lambda` | | 1 | TD-lambda for critic value loss |
| `ppo_clip_epsilon` | `--ppo-clip-epsilon` | 0.2 | PPO clipping |
| `entropy_temperature` | `--entropy-temperature` | 0.05 | Entropy bonus coefficient |
| `n_sweeps_per_epoch` | `--n-sweeps` | 6 | Minibatch sweeps per epoch |
| `n_mts_per_minibatch` | `--minibatch-size` | 64 | Timesteps per minibatch |
| `n_critic_iterations` | `--n-critic-iterations` | 10 | Critic update iterations |
| `max_grad_norm` | | 0.5 | Gradient clipping |
| `actor_logit_clip` | | 5.0 | Logit clipping before softmax |
| `noise` | | 0.0 | Action probability noise |
| `logit_based_entropy` | | False | Use logit variance penalty instead of entropy |
| `logit_variance_threshold` | | 0 | Threshold for logit variance hinge penalty |
| `logit_variance_lambda` | | 1.0 | Weight for logit variance penalty |
| `freeze_from_epoch` | | None | Freeze both lobes from this epoch |
| `freeze_actor_from_epoch` | | None | Freeze actor only |
| `freeze_critic_from_epoch` | | None | Freeze critic only |
| `target_kld` | `--target-kld` | None | Symmetric KLD target (sets both min and max) |
| `target_max_kld` | | None | KLD upper bound (binary search) |
| `target_min_kld` | | None | KLD lower bound (binary search) |
| `kld_method` | | 'binary-search' | KLD adjustment method |

## Trek directory structure

Training with trek enabled (the default) creates a timestamped directory under `~/.highjax/t/` (or `$HIGHJAX_HOME/t/`). Set `$HIGHJAX_HOME` to override the default `~/.highjax/` root. All data (treks, behaviors, config, logs) will use `$HIGHJAX_HOME/` instead.

```
~/.highjax/t/2026-03-15-20-02-25-101327/
  meta.yaml      # Environment config, git info, parameter counts
  sample_es.pq   # Sampled episode trajectories
  epochia.pq     # Epoch-level metrics
  train.log      # Training log output
```

### meta.yaml

Written at trek creation. Contains environment parameters, hostname, git commit/clean status, JAX devices, parameter counts, and estimator types.

### sample_es.pq

One row per sub-timestep per sampled episode. Each epoch writes `n_sample_es` episodes (default 1) evenly spaced across the batch. Columns include:

- `epoch`, `e`, `t` — identifiers (t is float: integer for policy steps, fractional for sub-steps)
- `reward`, `return` — per-step reward and discounted return
- `crash_reward` — reward at crash step (None for non-crash steps)
- `state.crashed` — crash state (pre-action for policy rows)
- `action`, `action_name` — action taken
- `p.{action}` — 5 action probabilities (left, idle, right, faster, slower)
- `v`, `tendency`, `advantage`, `nz_advantage` — value, log-prob, advantage
- `state.ego_x/y/heading/speed` — ego vehicle state
- `state.npc{i}_x/y/heading/speed` — NPC vehicle states
- `state.npc{i}_attention` — attention weights mapped to physical NPC indices

Written with fastparquet append — file is always readable mid-training (valid footer after each epoch).

### epochia.pq

One row per epoch. Key columns:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number |
| `datetime` | Timestamp |
| `wall_seconds` | Seconds since training start |
| `reward.coeval` | Mean coeval reward |
| `alive_count` | Number of alive timesteps |
| `alive_fraction` | Fraction of timesteps alive |
| `loss.v` | Critic value loss |
| `objective.vanilla` | PPO vanilla objective |
| `kld` | KL divergence between old and new policy |
| `v.mean` | Mean value estimate |
| `return.mean` | Mean episode return |
| `nz_speed` | Mean normalized forward speed (0-1 scale) |
| `nz_return` | Normalized discounted return (survival + speed + right_lane) |
| `nz_return.survival` | Survival contribution to nz_return |
| `nz_return.speed` | Speed contribution to nz_return |
| `nz_return.right_lane` | Right-lane contribution to nz_return |
| `behavior.{name}` | Behavior score (e.g. `behavior.collision`) |

Also written with fastparquet append — always readable mid-training.

## Source files

| File | Role |
|------|------|
| `highjax_trainer/training.py` | Training loop (Genesis → JointAscent) |
| `highjax_trainer/genesis.py` | Genesis: epoch 0 initial rollout |
| `highjax_trainer/joint_ascent.py` | JointAscent: per-agent ascent + create_next |
| `highjax_trainer/ascent.py` | Ascent: lightweight result container |
| `highjax_trainer/ascending.py` | PPO gradient computation (Ascender, SweepMaster, Sweeper, Minibatcher) |
| `highjax_trainer/rollout.py` | Rollout with replay buffer (roll_and_extend, tail) |
| `highjax_trainer/populating.py` | Population: multi-agent brain collection |
| `highjax_trainer/stepping.py` | StepCarry, StepReturn, step function |
| `highjax_trainer/brain.py` | Brain: actor + critic lobes |
| `highjax_trainer/lobe.py` | Lobe: network wrapper with optimizer state |
| `highjax_trainer/attention_estimating.py` | Ego-attention actor and critic estimators |
| `highjax_trainer/estimating.py` | Basic ANN estimators (not used by default) |
| `highjax_trainer/config.py` | AgentConfig dataclass |
| `highjax_trainer/defaults.py` | Default hyperparameter values |
| `highjax_trainer/optax_tools.py` | OptimizerRecipe for creating optax optimizers |
| `highjax_trainer/objectives.py` | GAE advantage, returns, and value function objectives |
| `highjax_trainer/jax_utils.py` | JAX utilities (categorical sampling) |
| `highjax_trainer/evaluating.py` | Evaluator: per-agent coeval reward |
| `highjax_trainer/epoch_reporting.py` | EpochReporter: epochia + sample_es writing |
| `highjax_trainer/parquet_writing.py` | ParquetWriter: unified append-mode parquet |
| `highjax_trainer/trekking.py` | Trek class: directory creation, meta.yaml |
| `highjax_trainer/es_writer.py` | Episode writer constants (action names, column definitions) |
| `highjax_trainer/ozette/` | Log capture and processing (config, core, metadata, processing, wrappers) |
| `highjax_trainer/filtros.py` | Stream filtering for suppressing warnings |
| `highjax_trainer/cli.py` | Click CLI entry point |
