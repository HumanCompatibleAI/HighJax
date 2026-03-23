from __future__ import annotations

import flax.struct
import jax

from .brain import Brain


@flax.struct.dataclass
class Ascent:
    brain: Brain
    next_brain: Brain
    v_loss: jax.Array
    kld: jax.Array
    vanilla_objective: jax.Array
