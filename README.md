# HighJax: Highway Driving environment for Reinforcement Learning research

<p align="center">
    <img src="https://raw.githubusercontent.com/HumanCompatibleAI/HighJax/master/misc/videos/demo.webp" alt="HighJax PPO training demo"><br/>
    <em>PPO agent learning to drive on a 4-lane highway</em>
</p>

HighJax is an autonomous driving environment for Reinforcement Learning research. It's a JAX implementation of the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv). HighJax provides a fully JIT-compilable and vectorizable highway driving simulation.

Besides being much faster than the original, it provides Octane, a Rust-based TUI for examining your experiment runs. Octane provides an interface for defining behaviors and then measuring how much each policy exhibits them.

HighJax was produced as part of our research project about [BXRL:Behavior-Explainable Reinforcement Learning](https://arxiv.org/abs/XXXX.XXXXX).

## Installation

```bash
pip install highjax-rl # Minimal installation
pip install "highjax-rl[cuda12]" # Including GPU support
pip install "highjax-rl[trainer]" # Including PPO implementation
pip install "highjax-rl[cuda12,trainer]" # Including both
```

## Quick Start

```python
import jax
import highjax

env, params = highjax.make('highjax-v0')
key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)
obs, state, reward, done, info = env.step(key, state, 1, params)  # IDLE
```

## Using with JAX RL Libraries

HighJax follows the [gymnax](https://github.com/RobertTLange/gymnax) API, so it works with JAX RL frameworks that expect gymnax-style environments:

- [PureJaxRL](https://github.com/luchris429/purejaxrl) — drop-in gymnax replacement (no PureJaxRL install needed), see [`examples/use_purejaxrl.py`](examples/use_purejaxrl.py)
- [Stoix](https://github.com/EdanToledo/Stoix) — via `stoa` gymnax adapter, see [`examples/use_stoix.py`](examples/use_stoix.py)
- [Rejax](https://github.com/keraJLi/rejax) — pass env object directly, see [`examples/use_rejax.py`](examples/use_rejax.py)

## Training

Train a PPO agent via the CLI:

```bash
highjax-trainer train
```

Key options:

| Flag                | Default | Description                          |
|---------------------|---------|--------------------------------------|
| `--n-epochs` / `-e` | 300     | Training epochs                      |
| `--n-es`            | 400     | Parallel episodes per epoch          |
| `--n-ts`            | 40      | Timesteps per episode                |
| `--seed` / `-s`     | 0       | Random seed                          |
| `--actor-lr`        | 3e-4    | Actor learning rate                  |
| `--critic-lr`       | 3e-3    | Critic learning rate                 |
| `--n-npcs`          | 50      | NPC vehicles                         |
| `--no-trek`         | —       | Disable trek recording               |
| `--n-sample-es`     | 1       | Episodes to sample per epoch for trek|
| `--trek-path`       | auto    | Custom trek directory path           |
| `--discount`        | 0.95    | Discount factor (gamma)              |
| `--n-lanes`         | 4       | Number of highway lanes              |

Training automatically records episode data to `~/.highjax/t/` for browsing with Octane (the TUI). Use `--no-trek` to disable.

Here's a snazzy one-liner that will let you explore the results of the current experiment run using [VisiData](https://github.com/saulpw/visidata):

```bash
pip install visidata
vd "$(ls -d ~/.highjax/t/2*/ | tail -1)"/epochia.pq
```

Use the following command line to produce similar results as seen in Figure 2 of the paper:

```bash
highjax-trainer train --n-es 128 --n-ts 400 --n-epochs 300 --target-kld 0.0005
```

## Octane (Episode Browser)

This repo also includes Octane, which is a Rust-based TUI for browsing HighJax experiments.

### Installation

```bash
sudo apt-get install build-essential # C toolchain (needed by Rust)
sudo apt-get install ffmpeg # Needed for `octane animate`
git clone https://github.com/HumanCompatibleAI/HighJax # Clone this repo
cd HighJax
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # Install Rust
source "$HOME/.cargo/env"
cd octane && cargo build --release # Build Octane
alias octane="$(readlink -f octane/target/release/octane)"
```

The binary will be at `octane/target/release/octane`.

### Usage

After training, launch Octane to see all the experiments you ran with `highjax-trainer`:

```bash
octane
```

### Figures

Use Octane to make figures for your paper:

```bash
octane draw -t ~/.highjax/t/2026-03-15-20-02-25-101327 --epoch 300 -e 0 --timestep 19 --theme light \
  --zoom 1.8 --png ~/figure.png
```

<p align="center">
    <img src="https://raw.githubusercontent.com/HumanCompatibleAI/HighJax/master/misc/images/figure.png" alt="Octane figure output" width="428"><br/>
</p>

### Behavior crafting

Octane includes a behavior explorer for defining measurable policy properties. While watching an episode, press `b` to capture a scenario — mark which actions you want (positive weight) or don't want (negative weight) at that traffic state. Name it, and Octane saves the behavior to `~/.highjax/behaviors/`. The next time you run `highjax-trainer train`, all discovered behaviors are evaluated every epoch and their scores are recorded as `behavior.{name}` columns in `epochia.parquet`.

<p align="center">
    <img src="https://raw.githubusercontent.com/HumanCompatibleAI/HighJax/master/misc/images/behavior_tui.png" alt="Behavior crafting dialog in Octane" width="364"><br/>
    <em>Defining a behavior scenario in Octane</em>
</p>

Press `B` (Shift-B) to open the full Behavior Explorer tab.

See the [Octane docs](docs/Octane/Octane%20docs.md) for full details.

## Documentation

Full documentation is in the `docs/` folder:

- [HighJax environment docs](docs/HighJax/HighJax%20docs.md) — state, observations, reward, NPCs, physics
- [Octane TUI docs](docs/Octane/Octane%20docs.md) — episode browser, configuration, key bindings
- [Coding conventions](docs/HighJax%20coding%20conventions.md) — naming, array indices, style

## Examples

- `examples/basic_usage.py` — Create env, reset, step, print observations
- `examples/train_ppo.py` — Train a PPO agent and evaluate it
- `examples/use_purejaxrl.py` — PureJaxRL integration (vectorized scan loop)
- `examples/use_stoix.py` — Stoix integration (via stoa gymnax adapter)
- `examples/use_rejax.py` — Rejax integration (JIT-compiled training, vmapped seeds)

## Citation

If you use HighJax in your research, please cite:

```bibtex
@article{rachum2025bxrl,
  title={BXRL: Behavior-Explainable Reinforcement Learning},
  author={Rachum, Ram and Amitai, Yotam and Nakar, Yonatan and Mirsky, Reuth and Allen, Cameron},
  year={2025}
}
```