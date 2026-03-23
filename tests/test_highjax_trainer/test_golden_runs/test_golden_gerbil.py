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
    'alive_fraction': (0.612500011920929, 0.762499988079071, 0.4000000059604645, 0.38749998807907104),
    'kld': (0.016283435747027397, 0.0029308709781616926, 0.004812698811292648, 0.008435249328613281),
    'reward_mean': (0.7326930165290833, 0.7988559603691101, 0.7413733005523682, 0.678612232208252),
    'v_loss': (2.2858164310455322, 4.17299747467041, 6.555177688598633, 7.591404914855957),
}


def train():
    env, params = highjax.make('highjax-v0', n_npcs=10)
    config = AgentConfig(
        actor_lr=1e-3,
        critic_lr=1e-2,
        entropy_temperature=0.1,
        n_mts_per_minibatch=None,
    )
    master_seed = jax.random.PRNGKey(42)

    population = Population.create_random(
        (config,), master_seed, env, params, n_es=4,
    )

    rollout_seed, master_seed = jax.random.split(master_seed)
    rollout, _ = collect_rollout(
        env, params, population, rollout_seed, n_es=4, n_ts=20,
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
            env, params, population, epoch_seed, n_es=4, n_ts=20,
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


def test_golden_gerbil():
    epoch_metrics = train()
    if golden_data is None:
        common.print_golden_data(epoch_metrics)
        assert False, 'Golden data not set — copy printed values above'
    common.check_golden_run(epoch_metrics, golden_data)
