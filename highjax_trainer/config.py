'''Agent configuration for PPO training.'''
from __future__ import annotations

from typing import Optional
import dataclasses
import re

import flax.struct

from . import defaults
from .optax_tools import OptimizerRecipe

_inline_argument_pattern = re.compile(r'^(\w+)(?:\(((?:\s*\w+\s*=\s*[^,\s)]+\s*,?\s*)*)\))?$')


def _parse_constructor_string(s: str) -> tuple[str, dict[str, float]]:
    '''Parse "name(k1=v1, k2=v2)" into (name, {k1: v1, k2: v2}).'''
    match = _inline_argument_pattern.fullmatch(s)
    if match is None:
        raise ValueError(f'Invalid constructor string: {s!r}')
    name = match.group(1)
    kwargs_str = match.group(2)
    if not kwargs_str:
        return name, {}
    kwargs = {}
    for part in kwargs_str.split(','):
        part = part.strip()
        if not part:
            continue
        k, v = part.split('=', 1)
        kwargs[k.strip()] = float(v.strip())
    return name, kwargs


@flax.struct.dataclass
class AgentConfig:
    '''PPO agent hyperparameters.'''

    actor_lr: float = defaults.DEFAULT_ACTOR_LR
    critic_lr: float = defaults.DEFAULT_CRITIC_LR
    critic_lr_end: Optional[float] = defaults.DEFAULT_CRITIC_LR_END

    optimizer_string: str = dataclasses.field(default=defaults.DEFAULT_OPTIMIZER_STRING)
    estimator_string: str = dataclasses.field(default=defaults.DEFAULT_ESTIMATOR_STRING)

    vanilla_gae_lambda: float = defaults.DEFAULT_VANILLA_GAE_LAMBDA
    critic_lambda: float = defaults.DEFAULT_CRITIC_LAMBDA
    n_critic_iterations: int = defaults.DEFAULT_N_CRITIC_ITERATIONS

    n_mts_per_minibatch: Optional[int] = defaults.DEFAULT_N_MTS_PER_MINIBATCH
    n_sweeps_per_epoch: int = defaults.DEFAULT_N_SWEEPS_PER_EPOCH
    ppo_clip_epsilon: Optional[float] = defaults.DEFAULT_PPO_CLIP_EPSILON
    max_grad_norm: Optional[float] = defaults.DEFAULT_MAX_GRAD_NORM

    target_kld: Optional[float] = defaults.DEFAULT_TARGET_KLD
    target_max_kld: Optional[float] = defaults.DEFAULT_TARGET_MAX_KLD
    target_min_kld: Optional[float] = defaults.DEFAULT_TARGET_MIN_KLD
    kld_method: str = defaults.DEFAULT_KLD_METHOD

    discount: float = defaults.DEFAULT_DISCOUNT
    noise: float = defaults.DEFAULT_NOISE

    logit_based_entropy: bool = defaults.DEFAULT_LOGIT_BASED_ENTROPY
    logit_variance_threshold: float = defaults.DEFAULT_LOGIT_VARIANCE_THRESHOLD
    logit_variance_lambda: float = defaults.DEFAULT_LOGIT_VARIANCE_LAMBDA
    actor_logit_clip: Optional[float] = defaults.DEFAULT_ACTOR_LOGIT_CLIP

    freeze_from_epoch: Optional[int] = defaults.DEFAULT_FREEZE_FROM_EPOCH
    freeze_actor_from_epoch: Optional[int] = defaults.DEFAULT_FREEZE_ACTOR_FROM_EPOCH
    freeze_critic_from_epoch: Optional[int] = defaults.DEFAULT_FREEZE_CRITIC_FROM_EPOCH

    entropy_temperature: float = defaults.DEFAULT_ENTROPY_TEMPERATURE

    @property
    def effective_target_max_kld(self) -> Optional[float]:
        return self.target_kld if self.target_kld is not None else self.target_max_kld

    @property
    def effective_target_min_kld(self) -> Optional[float]:
        return self.target_kld if self.target_kld is not None else self.target_min_kld

    @property
    def optimizer_kwargs(self) -> dict[str, float]:
        _, kwargs = _parse_constructor_string(self.optimizer_string)
        return kwargs

    @property
    def actor_optimizer_recipe(self) -> OptimizerRecipe:
        optimizer_name = _inline_argument_pattern.fullmatch(self.optimizer_string).group(1)
        return OptimizerRecipe(
            name=optimizer_name,
            learning_rate=self.actor_lr,
            max_grad_norm=self.max_grad_norm,
            optimizer_kwargs=self.optimizer_kwargs,
        )

    @property
    def critic_optimizer_recipe(self) -> OptimizerRecipe:
        optimizer_name = _inline_argument_pattern.fullmatch(self.optimizer_string).group(1)
        if self.critic_lr_end is None:
            return OptimizerRecipe(
                name=optimizer_name,
                learning_rate=self.critic_lr,
                max_grad_norm=self.max_grad_norm,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        else:
            return OptimizerRecipe(
                name=optimizer_name,
                learning_rate=self.critic_lr,
                learning_rate_end=self.critic_lr_end,
                learning_rate_schedule='linear_schedule',
                learning_rate_transition_steps=self.n_critic_iterations,
                max_grad_norm=self.max_grad_norm,
                optimizer_kwargs=self.optimizer_kwargs,
            )
