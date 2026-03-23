from __future__ import annotations

import jax

import highjax
from highjax_trainer.config import AgentConfig
from highjax_trainer.populating import Population
from highjax_trainer.rollout import collect_rollout
from highjax_trainer.ascending import Ascender

from . import common

golden_data = {
    'epoch': (1, 2, 3, 4),
    'alive_fraction': (0.949999988079071, 0.8999999761581421, 0.949999988079071, 0.949999988079071),
    'kld': (0.000989614869467914, 0.0006557493470609188, 0.0005189530202187598, 0.0005575679242610931),
    'reward_mean': (0.747053325176239, 0.7523806095123291, 0.8788608908653259, 0.7839143872261047),
    'v_loss': (2.622741460800171, 4.067610263824463, 3.28421950340271, 1.5494436025619507),
}


def train():
    env, params = highjax.make('highjax-v0', n_npcs=5)
    config = AgentConfig(n_mts_per_minibatch=None)
    master_seed = jax.random.PRNGKey(1)

    population = Population.create_random(
        (config,), master_seed, env, params, n_es=4,
    )

    rollout_seed, master_seed = jax.random.split(master_seed)
    rollout, _ = collect_rollout(
        env, params, population, rollout_seed, n_es=4, n_ts=10,
    )

    epoch_metrics = []
    for epoch in range(1, 5):
        epoch_seed, master_seed = jax.random.split(master_seed)

        ascender = Ascender(
            rollout=rollout, population=population,
            agent=0, agent_config=config,
        )
        population = population.duplicate(
            {0: ascender.next_brain},
        )
        rollout, _ = collect_rollout(
            env, params, population, epoch_seed, n_es=4, n_ts=10,
        )

        vital = rollout.vital_by_e_by_t
        reward_by_e_by_t = rollout.reward_by_agent_by_e_by_t[
            :, :, 0
        ]
        epoch_metrics.append({
            'reward_mean': float(
                (reward_by_e_by_t * vital).sum() / vital.sum()
            ),
            'v_loss': float(ascender.v_loss),
            'kld': float(ascender.kld),
            'alive_fraction': float(vital.sum() / vital.size),
        })

    return epoch_metrics


def test_golden_hamster():
    epoch_metrics = train()
    if golden_data is None:
        common.print_golden_data(epoch_metrics)
        assert False, 'Golden data not set — copy printed values above'
    common.check_golden_run(epoch_metrics, golden_data)
