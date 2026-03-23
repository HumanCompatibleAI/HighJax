'''
HighJaxEnv: JAX implementation of highway driving environment.

Follows gymnax conventions:
- EnvState: flax dataclass holding mutable environment state
- EnvParams: flax dataclass holding tunable environment parameters
- HighJaxEnv: env instance with step()/reset() methods, holds structural config
- make() factory returns (env, default_params)

Structural parameters (n_lanes, n_npcs, etc.) live on the HighJaxEnv instance
because they determine array shapes and can't be JIT-traced. Tunable parameters
(duration, speeds, ranges) live in EnvParams and can be traced through JAX.

State Layout
------------
The environment state is an EnvState dataclass containing:
  - ego_state: Vehicle state array (VEHICLE_STATE_SIZE,)
  - npc_states: NPC states array (n_npcs, VEHICLE_STATE_SIZE)
  - time: Time counter (scalar)
  - crashed: Crashed flag (scalar bool)
  - just_crashed: First crash this frame (scalar bool)
  - previous_ego_lane: Lane before action that produced this state (scalar)

Observation
-----------
Kinematic features for ego + nearby vehicles. Shape: (n_observed_vehicles, n_features).
Default: 5x5 matrix with (presence, x, y, vx, vy) for ego + 4 nearest NPCs.

Actions
-------
5 discrete actions: LANE_LEFT=0, IDLE=1, LANE_RIGHT=2, FASTER=3, SLOWER=4
'''
from __future__ import annotations

import functools
import math

import flax.struct
import gymnasium
import jax
import jax.numpy as jnp

from . import kinematics, lanes

# Observation constants
ALL_FEATURES = ('presence', 'x', 'y', 'vx', 'vy', 'heading', 'cos_h', 'sin_h',
                'cos_d', 'sin_d', 'long_off', 'lat_off', 'ang_off', 'lane')
DEFAULT_FEATURES = ('presence', 'x', 'y', 'vx', 'vy')


@flax.struct.dataclass
class EnvState:
    '''Mutable environment state (passed through step/reset).'''
    ego_state: jax.Array  # (VEHICLE_STATE_SIZE,)
    npc_states: jax.Array  # (n_npcs, VEHICLE_STATE_SIZE)
    time: jax.Array  # scalar
    crashed: jax.Array  # scalar bool
    just_crashed: jax.Array  # scalar bool
    previous_ego_lane: jax.Array  # scalar

    @staticmethod
    def from_scenario_dict(env, state_dict: dict) -> EnvState:
        '''Build an EnvState from a scenario state dict (as saved by Octane).

        The dict has keys like ego_x, ego_y, ego_speed, ego_heading, ego_lane,
        npc0_x, npc0_y, npc0_speed, npc0_heading, etc.
        '''
        ego_state = jnp.zeros(kinematics.VEHICLE_STATE_SIZE)
        ego_state = ego_state.at[kinematics.X].set(state_dict.get('ego_x', 0.0))
        ego_state = ego_state.at[kinematics.Y].set(state_dict.get('ego_y', 0.0))
        ego_state = ego_state.at[kinematics.HEADING].set(
            state_dict.get('ego_heading', 0.0))
        ego_state = ego_state.at[kinematics.SPEED].set(
            state_dict.get('ego_speed', 25.0))
        ego_lane = state_dict.get('ego_lane', 0)
        ego_state = ego_state.at[kinematics.TARGET_LANE_IDX].set(float(ego_lane))
        ego_state = ego_state.at[kinematics.SPEED_INDEX].set(1.0)
        ego_state = ego_state.at[kinematics.TARGET_SPEED].set(
            state_dict.get('ego_speed', 25.0))

        npc_states = jnp.zeros((env.n_npcs, kinematics.VEHICLE_STATE_SIZE))
        for i in range(env.n_npcs):
            prefix = f'npc{i}_'
            if f'{prefix}x' not in state_dict:
                break
            npc_states = npc_states.at[i, kinematics.X].set(
                state_dict[f'{prefix}x'])
            npc_states = npc_states.at[i, kinematics.Y].set(
                state_dict[f'{prefix}y'])
            npc_states = npc_states.at[i, kinematics.HEADING].set(
                state_dict.get(f'{prefix}heading', 0.0))
            npc_states = npc_states.at[i, kinematics.SPEED].set(
                state_dict.get(f'{prefix}speed', 22.0))
            npc_lane = state_dict.get(f'{prefix}lane', 0)
            npc_states = npc_states.at[i, kinematics.TARGET_LANE_IDX].set(
                float(npc_lane))
            npc_states = npc_states.at[i, kinematics.TARGET_SPEED].set(
                state_dict.get(f'{prefix}speed', 22.0))

        return EnvState(
            ego_state=ego_state,
            npc_states=npc_states,
            time=jnp.float32(state_dict.get('time', 0.0)),
            crashed=jnp.bool_(state_dict.get('crashed', False)),
            just_crashed=jnp.bool_(False),
            previous_ego_lane=jnp.float32(ego_lane),
        )


@flax.struct.dataclass
class EnvParams:
    '''Tunable environment parameters (can be JIT-traced).

    Structural params (n_lanes, n_npcs, etc.) live on the HighJaxEnv instance
    because they determine array shapes.
    '''
    # Timing
    duration: float = 40.0  # [s] Episode duration
    seconds_per_t: float = 1.0  # [s] per policy timestep
    seconds_per_sub_t: float = 1 / 15  # [s] per servo/NPC sub-timestep

    # Observation normalization
    x_range: float = 200.0  # [m]
    v_range: float = 80.0   # [m/s]

    # Spawn parameters
    ego_initial_speed: float = 25.0  # [m/s]
    ego_initial_lane: int = -1  # Lane index at spawn (-1 for random)
    npc_speed_min: float = 21.0  # [m/s]
    npc_speed_max: float = 24.0  # [m/s]
    vehicles_density: float = 1.0  # NPC density scaling
    ego_spacing: float = 2.0  # ego spacing multiplier


class HighJaxEnv:
    '''Highway driving environment in JAX.

    The ego vehicle must drive fast while avoiding collisions with NPC vehicles.
    NPCs follow IDM for longitudinal control and MOBIL for lane changes.

    Structural parameters are set at construction time and determine array
    shapes (can't change under JIT). Tunable parameters live in EnvParams.

    Follows gymnax-style API:
        obs, state = env.reset(key, params)
        obs, state, reward, done, info = env.step(key, state, action, params)
    '''

    def __init__(self, *,
                 n_lanes: int = 4,
                 n_npcs: int = 50,
                 lane_length: float = 10000.0,
                 lane_width: float = 4.0,
                 n_observed_vehicles: int = 5,
                 features: tuple[str, ...] = DEFAULT_FEATURES,
                 see_behind: bool = False,
                 perception_distance: float = 200.0,
                 enable_npc_lane_change: bool = True,
                 crash_on_predicted: bool = True,
                 simultaneous_update: bool = True,
                 npc_npc_collisions: bool = True,
                 npc_crash_braking: bool = True):
        # Structural params (determine array shapes, can't be traced)
        self.n_lanes = n_lanes
        self.n_npcs = n_npcs
        self.lane_length = lane_length
        self.lane_width = lane_width
        self.n_observed_vehicles = n_observed_vehicles

        # Observation config
        self.features = features
        self.feature_indices = tuple(ALL_FEATURES.index(f) for f in features)
        self.see_behind = see_behind
        self.perception_distance = perception_distance

        # Physics flags (Python bools used in control flow, can't be traced)
        self.enable_npc_lane_change = enable_npc_lane_change
        self.crash_on_predicted = crash_on_predicted
        self.simultaneous_update = simultaneous_update
        self.npc_npc_collisions = npc_npc_collisions
        self.npc_crash_braking = npc_crash_braking

        # Precompute highway lanes (concrete arrays, not traced)
        self.highway_lanes = lanes.make_highway_lanes(
            n_lanes, lane_length, lane_width)

        # Precompute sub-step count
        self.n_sub_ts_per_t = 15  # round(1.0 / (1/15)) = 15

    @property
    def default_params(self) -> EnvParams:
        '''Return default environment parameters.'''
        return EnvParams()

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self.n_observed_vehicles, len(self.features))

    def action_space(self, params: EnvParams = None) -> gymnasium.spaces.Discrete:
        '''Return the action space.'''
        return gymnasium.spaces.Discrete(5)

    def observation_space(self, params: EnvParams = None) -> gymnasium.spaces.Box:
        '''Return the observation space.'''
        return gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=self.observation_shape)

    def reset(self, key: jax.Array, params: EnvParams = None
              ) -> tuple[jax.Array, EnvState]:
        '''Reset environment to initial state.

        Args:
            key: JAX PRNG key
            params: Environment parameters (uses defaults if None)

        Returns:
            (observation, state)
        '''
        if params is None:
            params = self.default_params
        state = self._create_initial_state(key, params)
        obs = self._get_observation(state, params)
        return obs, state

    def step(self, key: jax.Array, state: EnvState, action: int,
             params: EnvParams = None
             ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        '''Step environment forward with auto-reset.

        Args:
            key: JAX PRNG key
            state: Current environment state
            action: Discrete action (0-4)
            params: Environment parameters (uses defaults if None)

        Returns:
            (observation, new_state, reward, done, info)
        '''
        if params is None:
            params = self.default_params

        # Step the environment
        key, step_key, reset_key = jax.random.split(key, 3)
        obs_step, state_step, reward, done, info = self.step_env(
            step_key, state, action, params)

        # Auto-reset: if done, reset and return reset obs
        obs_reset, state_reset = self.reset(reset_key, params)
        state_new = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y),
            state_reset, state_step
        )
        obs_new = jax.lax.select(done, obs_reset, obs_step)

        return obs_new, state_new, reward, done, info

    def step_env(self, key: jax.Array, state: EnvState, action: int,
                 params: EnvParams) -> tuple[jax.Array, EnvState, jax.Array,
                                             jax.Array, dict]:
        '''Step environment forward (no auto-reset).

        Args:
            key: JAX PRNG key
            state: Current environment state
            action: Discrete action (0-4)
            params: Environment parameters

        Returns:
            (observation, new_state, reward, done, info)
        '''
        from .stepper import step_physics

        # Flatten action
        action = jnp.asarray(action).astype(jnp.int32).flatten()[0]

        # Record old lane before action
        old_lane = state.ego_state[kinematics.TARGET_LANE_IDX]

        # Execute action (set targets)
        ego_after_action = kinematics.execute_action(
            state.ego_state, action, self.n_lanes)

        # Run physics sub-steps with collision detection
        new_ego, new_npcs, ego_crashed = step_physics(
            ego_after_action, state.npc_states, state.crashed,
            self.highway_lanes, self, params)

        # Build new state
        new_time = state.time + params.seconds_per_t
        first_crash = ego_crashed & ~state.crashed

        new_state = EnvState(
            ego_state=new_ego,
            npc_states=new_npcs,
            time=new_time,
            crashed=ego_crashed,
            just_crashed=first_crash,
            previous_ego_lane=old_lane,
        )

        # Compute reward and done
        reward = _compute_reward(new_state, self.n_lanes)
        done = _is_done(new_state, params)
        obs = self._get_observation(new_state, params)
        info = {'discount': jnp.where(done, 0.0, 1.0)}

        return obs, new_state, reward, done, info

    def get_sub_states(self, state: EnvState, action: int,
                       params: EnvParams = None):
        '''Run one policy step and return intermediate sub-step states.

        Returns:
            (sub_ego_by_sub_t, sub_npcs_by_sub_t, sub_crashed_by_sub_t) with
            shapes (n_sub_ts, VEHICLE_STATE_SIZE), (n_sub_ts, n_npcs,
            VEHICLE_STATE_SIZE), and (n_sub_ts,).
        '''
        from .stepper import step_physics_with_sub_states
        if params is None:
            params = EnvParams()
        ego_after_action = kinematics.execute_action(
            state.ego_state, action, self.n_lanes)
        _, _, _, sub_ego, sub_npcs, sub_crashed = step_physics_with_sub_states(
            ego_after_action, state.npc_states, state.crashed,
            self.highway_lanes, self, params)
        return sub_ego, sub_npcs, sub_crashed

    def _get_observation(self, state: EnvState, params: EnvParams) -> jax.Array:
        '''Compute observation from state.'''
        from .observations import compute_observation
        return compute_observation(state, self, params)

    def _create_initial_state(self, key: jax.Array,
                              params: EnvParams) -> EnvState:
        '''Create initial state with random NPC placement.'''
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        # Spawn ego vehicle
        ego_lane = jnp.where(
            params.ego_initial_lane == -1,
            jax.random.randint(key5, (), 0, self.n_lanes),
            params.ego_initial_lane,
        )
        ego_y = ego_lane * self.lane_width

        # Sequential spawn: ego at 3 * ego_offset
        exp_factor = jnp.exp(-5.0 / 40.0 * self.n_lanes)
        ego_default_spacing = 12.0 + params.ego_initial_speed
        ego_offset = params.ego_spacing * ego_default_spacing * exp_factor
        ego_x = 3.0 * ego_offset

        ego_state = kinematics.make_vehicle_state(
            x=ego_x, y=ego_y, heading=0.0, speed=params.ego_initial_speed,
            target_lane_idx=ego_lane
        )

        # Spawn NPCs
        npc_lane_by_npc = jax.random.randint(key1, (self.n_npcs,), 0,
                                              self.n_lanes)
        npc_speed_by_npc = jax.random.uniform(
            key3, (self.n_npcs,),
            minval=params.npc_speed_min, maxval=params.npc_speed_max)
        npc_delta_by_npc = jax.random.uniform(
            key4, (self.n_npcs,), minval=3.5, maxval=4.5)

        # Sequential NPC placement
        npc_x_by_npc = _spawn_sequential(
            ego_x, npc_speed_by_npc, self.n_lanes, params, key2)

        lane_width = self.lane_width

        def make_npc(lane_idx, x, speed, delta):
            y = lane_idx.astype(jnp.float32) * lane_width
            lane_change_timer = ((x + y) * jnp.pi) % kinematics.LANE_CHANGE_DELAY
            return jnp.array([x, y, 0.0, speed, lane_idx.astype(jnp.float32),
                              0.0, speed,
                              lane_change_timer, delta, 0.0])

        npc_states = jax.vmap(make_npc)(
            npc_lane_by_npc, npc_x_by_npc, npc_speed_by_npc, npc_delta_by_npc)

        return EnvState(
            ego_state=ego_state,
            npc_states=npc_states,
            time=jnp.array(0.0),
            crashed=jnp.array(False),
            just_crashed=jnp.array(False),
            previous_ego_lane=ego_lane.astype(jnp.float32),
        )


def make(env_id: str = 'highjax-v0', **kwargs) -> tuple[HighJaxEnv, EnvParams]:
    '''Create a HighJax environment.

    Args:
        env_id: Environment identifier (currently only 'highjax-v0')
        **kwargs: Structural parameters passed to HighJaxEnv constructor

    Returns:
        (env, default_params)
    '''
    if env_id != 'highjax-v0':
        raise ValueError(
            f'Unknown env_id: {env_id!r}. Only "highjax-v0" is supported.')
    env = HighJaxEnv(**kwargs)
    return env, env.default_params


def _spawn_sequential(ego_x: float, npc_speed_by_npc: jax.Array,
                      n_lanes: int, params: EnvParams,
                      key: jax.Array) -> jax.Array:
    '''Sequential spawning matching highway-env's create_random.'''
    exp_factor = jnp.exp(-5.0 / 40.0 * n_lanes)
    npc_spacing = 1.0 / params.vehicles_density
    n_npcs = npc_speed_by_npc.shape[0]

    jitter_key_by_npc = jax.random.split(key, n_npcs)

    def place_npc(max_x, inputs):
        speed, jitter_key = inputs
        default_spacing = 12.0 + speed
        offset = npc_spacing * default_spacing * exp_factor
        jitter = jax.random.uniform(jitter_key, minval=0.9, maxval=1.1)
        x = max_x + offset * jitter
        return x, x

    _, npc_x_by_npc = jax.lax.scan(
        place_npc, ego_x, (npc_speed_by_npc, jitter_key_by_npc)
    )
    return npc_x_by_npc


def _compute_reward(state: EnvState, n_lanes: int) -> jax.Array:
    '''Compute highway-v0 reward (matching HighwayEnv's canonical formula).

    r = (-1 * collision + 0.4 * nz_speed + 0.1 * right_lane + 1) / 1.5
    Zero on post-crash timesteps.
    '''
    collision_reward = -1.0
    high_speed_reward = 0.4
    right_lane_reward = 0.1

    collision_component = state.crashed.astype(jnp.float32)

    # Normalized forward speed
    forward_speed = (state.ego_state[kinematics.SPEED] *
                     jnp.cos(state.ego_state[kinematics.HEADING]))
    speed_component = jnp.clip((forward_speed - 20.0) / 10.0, 0, 1)

    right_lane_component = (state.ego_state[kinematics.TARGET_LANE_IDX] /
                            (n_lanes - 1))

    raw_score = (collision_reward * collision_component +
                 high_speed_reward * speed_component +
                 right_lane_reward * right_lane_component)

    reward = ((raw_score - collision_reward) /
              (high_speed_reward + right_lane_reward - collision_reward))

    # Zero on post-crash timesteps (crashed but not just now)
    was_already_crashed = state.crashed & ~state.just_crashed
    reward = jnp.where(was_already_crashed, 0.0, reward)

    return reward


def _is_done(state: EnvState, params: EnvParams) -> jax.Array:
    '''Check if episode is done (crash or timeout).'''
    timed_out = state.time >= params.duration
    return state.crashed | timed_out
