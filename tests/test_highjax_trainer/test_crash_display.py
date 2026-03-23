from __future__ import annotations

import datetime

import jax
import jax.numpy as jnp
import pandas as pd
import pyarrow.parquet as pq

import highjax
from highjax_trainer.ascending import Ascender
from highjax_trainer.ascent import Ascent
from highjax_trainer.config import AgentConfig
from highjax_trainer.epoch_reporting import EpochReporter
from highjax_trainer.evaluating import Evaluator
from highjax_trainer.joint_ascent import JointAscent
from highjax_trainer.populating import Population
from highjax_trainer.rollout import collect_rollout
from highjax_trainer.trekking import Trek


def _find_crashing_seed(env, params, population, n_es, n_ts,
                        max_attempts=50):
    for seed_num in range(max_attempts):
        seed = jax.random.PRNGKey(seed_num + 100)
        rollout, state_data = collect_rollout(
            env, params, population, seed, n_es, n_ts, epoch=1,
        )
        crashed = state_data['crashed_by_e_by_t']
        if crashed.any():
            return seed, rollout, state_data
    return None, None, None


def _record_epoch(env, params, config, population, rollout,
                  state_data, trek_path, n_sample_es):
    trek = Trek.create(
        env=env, params=params, agent_configs=(config,),
        trek_path=trek_path,
    )
    ascender = Ascender(
        rollout=rollout, population=population,
        agent=0, agent_config=config,
    )
    ascent = Ascent(
        brain=population[0], next_brain=ascender.next_brain,
        v_loss=ascender.v_loss, kld=ascender.kld,
        vanilla_objective=ascender.vanilla_objective,
    )
    joint_ascent = JointAscent(
        ascent_by_agent=(ascent,), rollout=rollout,
        rollout_seed=jax.random.PRNGKey(0),
        evaluation_seed=jax.random.PRNGKey(1),
        next_seed=jax.random.PRNGKey(2),
        replay_buffer_factor=1,
    )
    evaluator = Evaluator(env, params)
    evaluation = evaluator.evaluate(
        jax.random.PRNGKey(3), population, rollout, epoch=1,
    )
    reporter = EpochReporter(
        trek=trek, epoch=1, joint_ascent=joint_ascent,
        evaluation=evaluation,
        startup_time=datetime.datetime.now(),
        env=env, params=params,
    )
    reporter.report(state_data, n_sample_es_per_epoch=n_sample_es)


def _read_episode(trek_path, epoch, e):
    df = pd.read_parquet(trek_path / 'sample_es.pq')
    return df[(df['epoch'] == epoch) & (df['e'] == e)]


def _policy_frames(ep_df):
    return ep_df[
        ep_df['t'].apply(lambda x: abs(x - round(x)) < 1e-6)
    ]


def _sub_frames(ep_df):
    return ep_df[
        ep_df['t'].apply(lambda x: abs(x - round(x)) >= 1e-6)
    ]


class TestCrashDisplay:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=10)
        self.config = AgentConfig(n_mts_per_minibatch=None)
        self.population = Population.create_random(
            (self.config,), jax.random.PRNGKey(0),
            self.env, self.params, n_es=8,
        )

    def test_policy_row_at_crash_step_not_red(self, tmp_path):
        seed, rollout, state_data = _find_crashing_seed(
            self.env, self.params, self.population,
            n_es=8, n_ts=30,
        )
        assert seed is not None, 'Could not find a crashing episode'

        trek_path = tmp_path / 'crash_test'
        _record_epoch(
            self.env, self.params, self.config,
            self.population, rollout, state_data,
            trek_path, n_sample_es=8,
        )

        df = pd.read_parquet(trek_path / 'sample_es.pq')
        crashed_by_e_by_t = state_data['crashed_by_e_by_t']

        for e_idx in range(8):
            ep = df[(df['epoch'] == 1) & (df['e'] == e_idx)]
            if ep.empty:
                continue
            policy = _policy_frames(ep)

            crash_step = None
            for t in range(30):
                if bool(crashed_by_e_by_t[t, e_idx]):
                    crash_step = t
                    break

            if crash_step is None:
                assert not policy['state.crashed'].any(), (
                    f'Episode {e_idx} did not crash but has '
                    f'crashed=True policy rows'
                )
                continue

            crash_policy = policy[policy['t'].apply(
                lambda x: abs(x - crash_step) < 1e-6,
            )]
            if not crash_policy.empty:
                assert not crash_policy.iloc[0]['state.crashed'], (
                    f'Episode {e_idx}: policy row at crash step '
                    f't={crash_step} has state.crashed=True but '
                    f'position is pre-collision'
                )

    def test_exactly_one_crashed_frame(self, tmp_path):
        seed, rollout, state_data = _find_crashing_seed(
            self.env, self.params, self.population,
            n_es=8, n_ts=30,
        )
        assert seed is not None, 'Could not find a crashing episode'

        trek_path = tmp_path / 'crash_one_frame_test'
        _record_epoch(
            self.env, self.params, self.config,
            self.population, rollout, state_data,
            trek_path, n_sample_es=8,
        )

        df = pd.read_parquet(trek_path / 'sample_es.pq')
        crashed_by_e_by_t = state_data['crashed_by_e_by_t']

        found_crash = False
        for e_idx in range(8):
            ep = df[(df['epoch'] == 1) & (df['e'] == e_idx)]
            if ep.empty:
                continue

            crash_step = None
            for t in range(30):
                if bool(crashed_by_e_by_t[t, e_idx]):
                    crash_step = t
                    break
            if crash_step is None:
                continue
            found_crash = True

            crashed_rows = ep[ep['state.crashed']]
            assert len(crashed_rows) == 1, (
                f'Episode {e_idx}: expected 1 crashed frame, '
                f'got {len(crashed_rows)}'
            )

            last_t = ep['t'].max()
            crash_t = crashed_rows.iloc[0]['t']
            assert abs(crash_t - last_t) < 1e-6, (
                f'Episode {e_idx}: crashed frame at '
                f't={crash_t:.4f} is not the last frame'
            )

        assert found_crash, 'No crashing episodes found'

    def test_no_policy_row_after_crash(self, tmp_path):
        seed, rollout, state_data = _find_crashing_seed(
            self.env, self.params, self.population,
            n_es=8, n_ts=30,
        )
        assert seed is not None

        trek_path = tmp_path / 'crash_no_next_test'
        _record_epoch(
            self.env, self.params, self.config,
            self.population, rollout, state_data,
            trek_path, n_sample_es=8,
        )

        df = pd.read_parquet(trek_path / 'sample_es.pq')
        crashed_by_e_by_t = state_data['crashed_by_e_by_t']

        for e_idx in range(8):
            ep = df[(df['epoch'] == 1) & (df['e'] == e_idx)]
            if ep.empty:
                continue
            policy = _policy_frames(ep)

            crash_step = None
            for t in range(30):
                if bool(crashed_by_e_by_t[t, e_idx]):
                    crash_step = t
                    break
            if crash_step is None:
                continue

            next_policy = policy[policy['t'].apply(
                lambda x: abs(x - (crash_step + 1)) < 1e-6,
            )]
            assert next_policy.empty, (
                f'Episode {e_idx}: found policy row at '
                f't={crash_step+1} after crash'
            )

    def test_surviving_episode_no_crashed_flags(self, tmp_path):
        seed = jax.random.PRNGKey(42)
        rollout, state_data = collect_rollout(
            self.env, self.params, self.population,
            seed, n_es=8, n_ts=10, epoch=1,
        )

        trek_path = tmp_path / 'survive_test'
        _record_epoch(
            self.env, self.params, self.config,
            self.population, rollout, state_data,
            trek_path, n_sample_es=8,
        )

        df = pd.read_parquet(trek_path / 'sample_es.pq')
        crashed_by_e_by_t = state_data['crashed_by_e_by_t']

        for e_idx in range(8):
            ep = df[(df['epoch'] == 1) & (df['e'] == e_idx)]
            if ep.empty:
                continue

            survived = not crashed_by_e_by_t[:, e_idx].any()
            if not survived:
                continue

            assert not ep['state.crashed'].any(), (
                f'Surviving episode {e_idx} has crashed=True rows'
            )

    def test_pre_crash_policy_rows_not_crashed(self, tmp_path):
        seed, rollout, state_data = _find_crashing_seed(
            self.env, self.params, self.population,
            n_es=8, n_ts=30,
        )
        assert seed is not None

        trek_path = tmp_path / 'pre_crash_test'
        _record_epoch(
            self.env, self.params, self.config,
            self.population, rollout, state_data,
            trek_path, n_sample_es=8,
        )

        df = pd.read_parquet(trek_path / 'sample_es.pq')
        crashed_by_e_by_t = state_data['crashed_by_e_by_t']

        for e_idx in range(8):
            ep = df[(df['epoch'] == 1) & (df['e'] == e_idx)]
            if ep.empty:
                continue
            policy = _policy_frames(ep)

            crash_step = None
            for t in range(30):
                if bool(crashed_by_e_by_t[t, e_idx]):
                    crash_step = t
                    break
            if crash_step is None:
                continue

            pre_crash = policy[policy['t'] < crash_step - 0.5]
            crashed_pre = pre_crash[pre_crash['state.crashed']]
            assert crashed_pre.empty, (
                f'Episode {e_idx}: pre-crash policy rows have '
                f'state.crashed=True'
            )
