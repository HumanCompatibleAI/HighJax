from __future__ import annotations

import datetime
import os
import pathlib
import shlex
import socket
import subprocess
import sys

import flax
import jax
import jaxlib
import optax
import yaml

from .config import AgentConfig
from .parquet_writing import ParquetWriter


def _get_highjax_home() -> pathlib.Path:
    env_value = os.environ.get('HIGHJAX_HOME')
    if env_value:
        return pathlib.Path(env_value)
    return pathlib.Path.home() / '.highjax'


def _get_highjax_treks() -> pathlib.Path:
    return _get_highjax_home() / 't'


def _make_timestamp_name() -> str:
    now = datetime.datetime.now()
    return (
        now.strftime('%Y-%m-%d-%H-%M-%S-') + f'{now.microsecond:06d}'
    )


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_git_is_clean() -> bool | None:
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() == ''
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_cuda_version() -> str | None:
    try:
        platform_version = jax.devices()[0].client.platform_version
        for part in platform_version.split('\n'):
            part = part.strip()
            if part.startswith('cuda'):
                return part
        return platform_version.strip()
    except Exception:
        return None


class Trek:
    def __init__(self, folder: pathlib.Path):
        self.folder = pathlib.Path(folder)
        self.epochia_path = self.folder / 'epochia.pq'
        self.sample_es_path = self.folder / 'sample_es.pq'
        self.epochia_writer = ParquetWriter(self.epochia_path)
        self.sample_es_writer = ParquetWriter(self.sample_es_path)

    def __truediv__(self, path: str | os.PathLike) -> pathlib.Path:
        return self.folder / path

    @staticmethod
    def create(*, env, params, agent_configs, population=None,
               parent_folder: pathlib.Path | None = None,
               trek_path: pathlib.Path | None = None) -> Trek:
        if trek_path is not None:
            folder = trek_path
        elif parent_folder is not None:
            folder = parent_folder / _make_timestamp_name()
        else:
            folder = _get_highjax_treks() / _make_timestamp_name()

        folder.mkdir(parents=True, exist_ok=True)
        trek = Trek(folder)

        startup_time = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S',
        )

        meta = {
            'command_line': ' '.join(
                ['highjax-trainer'] + [shlex.quote(a) for a in sys.argv[1:]]
            ),
            'startup_time': startup_time,
            'hostname': socket.gethostname(),
            'git_commit': _get_git_commit(),
            'git_is_clean': _get_git_is_clean(),
            'jax_devices': repr(jax.devices()),
            'versions': {
                'python': sys.version.split()[0],
                'jax': jax.__version__,
                'jaxlib': jaxlib.__version__,
                'flax': flax.__version__,
                'optax': optax.__version__,
                'cuda': _get_cuda_version(),
            },
            'commands': {
                '1.highjax': {
                    'n_lanes': env.n_lanes,
                    'n_npcs': env.n_npcs,
                    'seconds_per_t': float(params.seconds_per_t),
                    'seconds_per_sub_t': float(
                        params.seconds_per_sub_t,
                    ),
                    'npc_speed_min': float(params.npc_speed_min),
                    'npc_speed_max': float(params.npc_speed_max),
                },
            },
        }

        if population is not None:
            brain = population[0]
            actor_flat, _ = jax.flatten_util.ravel_pytree(
                brain.actor_lobe.theta,
            )
            critic_flat, _ = jax.flatten_util.ravel_pytree(
                brain.critic_lobe.theta,
            )
            meta['parameter_count'] = {
                'actor': int(actor_flat.shape[0]),
                'critic': int(critic_flat.shape[0]),
            }
            meta['estimator_type'] = {
                'actor': brain.actor_lobe.estimator_type.__name__,
                'critic': (
                    brain.critic_lobe.estimator_type.__name__
                ),
            }

        trek.write_meta(meta)
        return trek

    def write_meta(self, data: dict) -> None:
        meta_path = self.folder / 'meta.yaml'
        try:
            with open(meta_path) as f:
                existing = yaml.safe_load(f) or {}
        except FileNotFoundError:
            existing = {}
        existing.update(data)
        with open(meta_path, 'w') as f:
            yaml.dump(
                existing, f, default_flow_style=False,
                sort_keys=False,
            )

    def close(self) -> None:
        pass


# Backward compat: keep create_trek as a convenience
def create_trek(env, params, agent_config: AgentConfig, *,
                trek_path: pathlib.Path | None = None
                ) -> pathlib.Path:
    trek = Trek.create(
        env=env, params=params,
        agent_configs=(agent_config,),
        trek_path=trek_path,
    )
    return trek.folder
