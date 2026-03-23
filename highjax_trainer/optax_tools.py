'''Pickleable optimizer recipe for creating optax optimizers.'''
from __future__ import annotations

from typing import Any, Optional
import dataclasses

import flax.struct
import optax


@flax.struct.dataclass
class OptimizerRecipe:
    '''A pickleable recipe for creating optax optimizers with optional learning rate schedules.'''

    name: str
    learning_rate: float
    learning_rate_end: Optional[float] = None
    learning_rate_schedule: Optional[str] = None
    learning_rate_transition_steps: Optional[int] = None
    max_grad_norm: Optional[float] = None
    optimizer_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        schedule_fields = [self.learning_rate_end, self.learning_rate_schedule,
                          self.learning_rate_transition_steps]
        none_count = sum(f is None for f in schedule_fields)
        if none_count not in (0, 3):
            raise ValueError(
                'learning_rate_end, learning_rate_schedule, and learning_rate_transition_steps '
                'must all be None or all be set'
            )

    def build_optimizer(self) -> optax.GradientTransformation:
        '''Create the optimizer from the recipe.'''
        if self.learning_rate_schedule is None:
            lr = self.learning_rate
        else:
            schedule_constructor = getattr(optax, self.learning_rate_schedule)
            lr = schedule_constructor(
                init_value=self.learning_rate,
                end_value=self.learning_rate_end,
                transition_steps=self.learning_rate_transition_steps
            )

        optimizer_constructor = getattr(optax, self.name)
        optimizer = optimizer_constructor(learning_rate=lr, **self.optimizer_kwargs)
        if self.max_grad_norm is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optimizer)
        return optimizer
