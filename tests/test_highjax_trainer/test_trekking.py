'''Tests for trek directory creation.'''
from __future__ import annotations

import re

import yaml
import pytest

import highjax
from highjax_trainer.config import AgentConfig
from highjax_trainer.trekking import create_trek


@pytest.fixture
def env_and_params():
    env, params = highjax.make('highjax-v0', n_npcs=5)
    return env, params


@pytest.fixture
def agent_config():
    return AgentConfig()


class TestCreateTrek:

    def test_create_trek_creates_directory(self, tmp_path, env_and_params,
                                           agent_config):
        env, params = env_and_params
        trek_path = create_trek(env, params, agent_config,
                                trek_path=tmp_path / 'my_trek')
        assert trek_path.is_dir()

    def test_create_trek_writes_meta_yaml(self, tmp_path, env_and_params,
                                          agent_config):
        env, params = env_and_params
        trek_path = create_trek(env, params, agent_config,
                                trek_path=tmp_path / 'my_trek')
        meta_path = trek_path / 'meta.yaml'
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert isinstance(meta, dict)

    def test_meta_yaml_has_required_fields(self, tmp_path, env_and_params,
                                           agent_config):
        env, params = env_and_params
        trek_path = create_trek(env, params, agent_config,
                                trek_path=tmp_path / 'my_trek')
        with open(trek_path / 'meta.yaml') as f:
            meta = yaml.safe_load(f)
        command = meta['commands']['1.highjax']
        assert command['n_lanes'] == env.n_lanes
        assert command['n_npcs'] == env.n_npcs
        assert command['seconds_per_t'] == float(params.seconds_per_t)
        assert command['seconds_per_sub_t'] == float(params.seconds_per_sub_t)

    def test_meta_yaml_octane_detection(self, tmp_path, env_and_params,
                                        agent_config):
        env, params = env_and_params
        trek_path = create_trek(env, params, agent_config,
                                trek_path=tmp_path / 'my_trek')
        with open(trek_path / 'meta.yaml') as f:
            meta = yaml.safe_load(f)
        command_keys = list(meta['commands'].keys())
        assert any('highjax' in key for key in command_keys)

    def test_create_trek_custom_path(self, tmp_path, env_and_params,
                                     agent_config):
        env, params = env_and_params
        custom = tmp_path / 'custom' / 'trek'
        trek_path = create_trek(env, params, agent_config, trek_path=custom)
        assert trek_path == custom
        assert trek_path.is_dir()
        assert (trek_path / 'meta.yaml').exists()

    def test_create_trek_timestamp_format(self, tmp_path, env_and_params,
                                          agent_config, monkeypatch):
        env, params = env_and_params
        monkeypatch.setenv('HIGHJAX_HOME', str(tmp_path))
        trek_path = create_trek(env, params, agent_config)
        name = trek_path.name
        pattern = r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}$'
        assert re.match(pattern, name), f'Name {name!r} does not match timestamp pattern'
        assert trek_path.parent == tmp_path / 't'
