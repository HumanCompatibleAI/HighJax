from __future__ import annotations

import datetime

import pyarrow.parquet as pq
import jax
import yaml

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
from highjax_trainer.training import train


def _make_joint_ascent(rollout, population, config):
    ascender = Ascender(
        rollout=rollout, population=population,
        agent=0, agent_config=config,
    )
    ascent = Ascent(
        brain=population[0],
        next_brain=ascender.next_brain,
        v_loss=ascender.v_loss,
        kld=ascender.kld,
        vanilla_objective=ascender.vanilla_objective,
    )
    return JointAscent(
        ascent_by_agent=(ascent,),
        rollout=rollout,
        rollout_seed=jax.random.PRNGKey(0),
        evaluation_seed=jax.random.PRNGKey(1),
        next_seed=jax.random.PRNGKey(2),
        replay_buffer_factor=1,
    )


class TestEpochReporter:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.config = AgentConfig(n_mts_per_minibatch=None)
        self.population = Population.create_random(
            (self.config,), jax.random.PRNGKey(0),
            self.env, self.params, n_es=4,
        )
        self.startup_time = datetime.datetime.now()
        from highjax.behaviors import discover_behaviors
        self.behaviors = discover_behaviors(self.env, self.params)

    def test_creates_files(self, tmp_path):
        trek = Trek.create(
            env=self.env, params=self.params,
            agent_configs=(self.config,),
            population=self.population,
            trek_path=tmp_path / 'test_trek',
        )
        rollout, state_data = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10, epoch=1,
        )
        joint_ascent = _make_joint_ascent(
            rollout, self.population, self.config,
        )
        evaluator = Evaluator(self.env, self.params)
        evaluation = evaluator.evaluate(
            jax.random.PRNGKey(2), self.population,
            rollout, epoch=1,
        )
        reporter = EpochReporter(
            trek=trek, epoch=1, joint_ascent=joint_ascent,
            evaluation=evaluation,
            startup_time=self.startup_time,
            env=self.env, params=self.params,
            behaviors=self.behaviors,
        )
        reporter.report(state_data, n_sample_es_per_epoch=2)

        assert (tmp_path / 'test_trek' / 'meta.yaml').exists()
        assert (tmp_path / 'test_trek' / 'sample_es.pq').exists()
        assert (tmp_path / 'test_trek' / 'epochia.pq').exists()

    def test_epochia_columns(self, tmp_path):
        trek = Trek.create(
            env=self.env, params=self.params,
            agent_configs=(self.config,),
            trek_path=tmp_path / 'test_trek',
        )
        rollout, state_data = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10, epoch=1,
        )
        joint_ascent = _make_joint_ascent(
            rollout, self.population, self.config,
        )
        evaluator = Evaluator(self.env, self.params)
        evaluation = evaluator.evaluate(
            jax.random.PRNGKey(2), self.population,
            rollout, epoch=1,
        )
        reporter = EpochReporter(
            trek=trek, epoch=1, joint_ascent=joint_ascent,
            evaluation=evaluation,
            startup_time=self.startup_time,
            env=self.env, params=self.params,
            behaviors=self.behaviors,
        )
        reporter.report(state_data, n_sample_es_per_epoch=1)

        table = pq.read_table(tmp_path / 'test_trek' / 'epochia.pq')
        assert table.num_rows == 1
        names = table.column_names
        assert 'epoch' in names
        assert 'wall_seconds' in names
        assert 'alive_count' in names
        assert 'alive_fraction' in names
        assert 'loss.v' in names
        assert 'objective.vanilla' in names
        assert 'kld' in names
        assert 'v.mean' in names
        assert 'return.mean' in names
        assert 'nz_return' in names
        assert 'reward.coeval' in names
        assert 'nz_speed' in names
        assert 'nz_return.survival' in names
        assert 'nz_return.speed' in names
        assert 'nz_return.right_lane' in names
        assert 'behavior.collision' in names
        assert table.column('alive_count').to_pylist()[0] > 0

    def test_sample_es_columns(self, tmp_path):
        trek = Trek.create(
            env=self.env, params=self.params,
            agent_configs=(self.config,),
            trek_path=tmp_path / 'test_trek',
        )
        rollout, state_data = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10, epoch=1,
        )
        joint_ascent = _make_joint_ascent(
            rollout, self.population, self.config,
        )
        evaluator = Evaluator(self.env, self.params)
        evaluation = evaluator.evaluate(
            jax.random.PRNGKey(2), self.population,
            rollout, epoch=1,
        )
        reporter = EpochReporter(
            trek=trek, epoch=1, joint_ascent=joint_ascent,
            evaluation=evaluation,
            startup_time=self.startup_time,
            env=self.env, params=self.params,
            behaviors=self.behaviors,
        )
        reporter.report(state_data, n_sample_es_per_epoch=1)

        table = pq.read_table(
            tmp_path / 'test_trek' / 'sample_es.pq',
        )
        names = table.column_names
        assert 'epoch' in names
        assert 'e' in names
        assert 't' in names
        assert 'reward' in names
        assert 'return' in names
        assert 'crash_reward' in names
        assert 'state.ego_x' in names
        assert 'state.npc0_x' in names
        assert 'p.idle' in names
        assert 'v' in names
        assert 'advantage' in names
        assert 'tendency' in names
        assert 'nz_advantage' in names

    def test_sample_count(self, tmp_path):
        trek = Trek.create(
            env=self.env, params=self.params,
            agent_configs=(self.config,),
            trek_path=tmp_path / 'test_trek',
        )
        rollout, state_data = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10, epoch=1,
        )
        joint_ascent = _make_joint_ascent(
            rollout, self.population, self.config,
        )
        evaluator = Evaluator(self.env, self.params)
        evaluation = evaluator.evaluate(
            jax.random.PRNGKey(2), self.population,
            rollout, epoch=1,
        )
        reporter = EpochReporter(
            trek=trek, epoch=1, joint_ascent=joint_ascent,
            evaluation=evaluation,
            startup_time=self.startup_time,
            env=self.env, params=self.params,
            behaviors=self.behaviors,
        )
        reporter.report(state_data, n_sample_es_per_epoch=2)

        table = pq.read_table(
            tmp_path / 'test_trek' / 'sample_es.pq',
        )
        unique_episodes = set(table.column('e').to_pylist())
        assert len(unique_episodes) == 2


class TestTrek:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.config = AgentConfig(n_mts_per_minibatch=None)
        self.population = Population.create_random(
            (self.config,), jax.random.PRNGKey(0),
            self.env, self.params, n_es=4,
        )

    def test_meta_yaml_content(self, tmp_path):
        trek = Trek.create(
            env=self.env, params=self.params,
            agent_configs=(self.config,),
            population=self.population,
            trek_path=tmp_path / 'test_trek',
        )
        with open(tmp_path / 'test_trek' / 'meta.yaml') as f:
            meta = yaml.safe_load(f)

        assert 'commands' in meta
        keys = list(meta['commands'].keys())
        assert any('highjax' in k for k in keys)
        assert 'hostname' in meta
        assert 'startup_time' in meta
        assert 'parameter_count' in meta
        assert 'actor' in meta['parameter_count']
        assert 'critic' in meta['parameter_count']
        assert 'estimator_type' in meta
        highjax_cmd = meta['commands']['1.highjax']
        assert 'n_lanes' in highjax_cmd
        assert 'n_npcs' in highjax_cmd
        assert 'seconds_per_t' in highjax_cmd


class TestTrainingWithTrek:

    def test_train_with_trek(self, tmp_path):
        trek_path = tmp_path / 'training_trek'
        env, params = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig(n_mts_per_minibatch=None)
        population = train(
            env, params, config,
            n_epochs=2, n_es=4, n_ts=10, verbose=False,
            trek=True, n_sample_es=1, trek_path=trek_path,
        )
        assert population is not None
        assert (trek_path / 'meta.yaml').exists()
        assert (trek_path / 'sample_es.pq').exists()
        assert (trek_path / 'epochia.pq').exists()

        pf = pq.ParquetFile(trek_path / 'sample_es.pq')
        assert pf.metadata.num_row_groups == 2

        et = pq.read_table(trek_path / 'epochia.pq')
        assert et.num_rows == 2
