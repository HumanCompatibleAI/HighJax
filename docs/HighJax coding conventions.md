# HighJax Coding Conventions

Related: [[HighJax docs]]

Naming conventions, style rules, and JAX-specific patterns used throughout the HighJax codebase.

## Array naming: `foo_by_bar_by_baz`

Almost all JAX arrays use the pattern `foo_by_bar_by_baz`. Each `by_something` is an axis (or sometimes multiple axes). Axes go right-to-left: `foo_by_bar_by_baz` is a 2D array where axis 0 is baz, axis 1 is bar, and the value is foo.

Examples:

- `reward_by_e_by_t` -- shape (n_ts, n_es), reward at each (timestep, episode)
- `p_by_action_by_e` -- shape (n_es, *action_shape), action probabilities

Multi-dimensional items like `position` or `action` occupy multiple axes. The number of axes that `action` occupies depends on the environment.

**This applies to ALL arrays** -- including intermediate variables, temporaries, results, diffs, distances. Even a simple subtraction result should be named `diff_by_e_by_t`, not `diff`.

## Dimension size naming: `n_foos_per_bar`

When unpacking array shapes to get dimension sizes, use `n_foos_per_bar`:

- `foos`: plural of what's being counted
- `bar`: the containing unit

Examples:

```python
n_es_per_epoch, n_cells_per_e, n_tokens_per_vocabulary = logit_by_token_by_cell_by_e.shape
n_es_per_epoch, n_cells_per_e = token_by_cell_by_e.shape
```

## Index variables: `i_foo`

When naming a variable that's an index number, use `i_foo`. However, these are exceptions that don't need the `i_` prefix: `epoch`, `t`, `e`.

## Shorthands

Only use shorthands that already exist in the codebase. Don't invent new ones. Established shorthands:

- `p` for probability
- `e` for episode
- `t` for timestep
- `ft` for flat timestep (full flattened pool, in minibatch code)
- `mt` for minibatch timestep (within one minibatch slice)
- `ts` for timesteps (plural, in CLI args like `--n-ts`)
- `es` for episodes (plural, in CLI args like `--n-es`)
- `v` for value estimate
- `vf` for value function
- `kld` for KL divergence (never bare `kl`)
- `obs` for observation (matches gymnax API convention)
- `nz` for normalized (prefix, e.g. `nz_speed`, `nz_return`, `nz_advantage`)
- `theta` for model parameters (neural network param dicts)
- `vital` for alive-mask arrays (not post-crash)
- `tendency` for log-probability of chosen action
- `epilogue` for post-final-step values (e.g. `epilogue_v_by_agent_by_e`)
- `lunge` for action dimension in multi-discrete action spaces
- `deed` for a choice within a lunge
- `mb` for minibatch (in axis names like `_by_mt_by_mb`)
- `sweep` for one pass over all minibatches

Don't shorten `position` to `pos`, etc.

## Code style

- Python 3.12+ required
- Single quotes everywhere, unless there's a quote-in-quote situation
- Maximum line length: 100 characters
- Type annotations using builtins (`list`, `tuple`, `dict`), not `List`, `Tuple`, `Dict`
- `from __future__ import annotations` at the top of every file
- Import order: `__future__` > stdlib > third-party > highjax
- snake_case for variables/functions, PascalCase for classes

## Docstrings and comments

The `highjax` environment package has docstrings on public API functions (reset, step, etc.) since it serves as a library. The `highjax_trainer` package generally avoids function docstrings — the code should be self-explanatory. Add comments sparingly, only when the code is genuinely difficult to understand otherwise.

## JAX JIT

Many functions that process arrays need to be JIT-compiled. This means:

- No Python control flow on array values (use `jnp.where` instead of `if`)
- Be careful with loops (use `jax.lax.scan` or `jax.vmap` instead)
- Use `jax.Array` and pytree-compatible data structures

## Flax dataclasses

HighJax uses `@flax.struct.dataclass` for most data classes. These are JAX-compatible (pytree-registered) frozen dataclasses. Fields can be marked `pytree_node=False` via `flax.struct.field(pytree_node=False)` to exclude from JAX tracing (e.g., config objects).

## Minibatch PPO pipeline

The gradient computation pipeline has a specific data flow with its own naming:

```
Ascender -> SweepMaster (flatten to _by_ft) -> Sweeper (shuffle to _by_mt_by_minibatch) -> Minibatcher (_by_mt per minibatch)
```

The Minibatcher computes the composite actor objective (PPO clipped surrogate + entropy) and produces gradients via `jax.grad`. The critic is updated separately.

## Testing

- **Golden tests** (`test_golden_runs/`): Deterministic training runs with exact expected values. When the training pipeline changes, these need regeneration. Each test defines its own `train()` function; run it, capture the new values, update `golden_data`.
- **Unit tests**: Everything else -- estimators, objectives, masking, freezing, trainer integration, etc.
