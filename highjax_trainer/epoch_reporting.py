from __future__ import annotations

import datetime

import jax
from jax import numpy as jnp
import numpy as np

from .joint_ascent import JointAscent
from .populating import Population
from .rollout import Rollout
from .trekking import Trek
from .es_writer import ACTION_NAMES
from highjax.kinematics import X, Y, HEADING, SPEED, TARGET_LANE_IDX
from highjax.behaviors import evaluate_behavior


class EpochReporter:
    def __init__(self, trek: Trek, epoch: int,
                 joint_ascent: JointAscent,
                 evaluation: dict,
                 startup_time: datetime.datetime,
                 env, params,
                 behaviors: tuple[dict, ...] = ()) -> None:
        self.trek = trek
        self.epoch = epoch
        self.joint_ascent = joint_ascent
        self.evaluation = evaluation
        self.startup_time = startup_time
        self.env = env
        self.params = params
        self.behaviors = behaviors

    @property
    def rollout(self) -> Rollout:
        return self.joint_ascent.rollout

    @property
    def population(self) -> Population:
        return self.joint_ascent.next_population

    def report(self, state_data: dict, *,
               n_sample_es_per_epoch: int = 1) -> None:
        wall_seconds = (
            datetime.datetime.now() - self.startup_time
        ).total_seconds()

        processed = dict(self.evaluation)
        processed['wall_seconds'] = wall_seconds

        rollout = self.rollout.tail
        vital = rollout.vital_by_e_by_t
        n_alive = int(vital.sum())
        n_total = vital.size
        processed['alive_count'] = n_alive
        processed['alive_fraction'] = n_alive / n_total

        for agent, ascent in enumerate(self.joint_ascent):
            processed['loss.v'] = float(ascent.v_loss)
            processed['objective.vanilla'] = float(
                ascent.vanilla_objective,
            )
            if ascent.kld is not None:
                processed['kld'] = float(ascent.kld)

        for agent in self.joint_ascent.next_population.agents:
            v_by_e_by_t = rollout.v_by_agent_by_e_by_t[:, :, agent]
            processed['v.mean'] = float(
                (v_by_e_by_t * vital).sum() / vital.sum(),
            )
            reward_by_e_by_t = (
                rollout.reward_by_agent_by_e_by_t[:, :, agent]
            )
            return_by_e = (reward_by_e_by_t * vital).sum(axis=0)
            processed['return.mean'] = float(
                return_by_e.mean(),
            )

        # Compute nz_speed and nz_return components from ego states
        ego_by_e_by_t = jnp.array(state_data['ego_state_by_e_by_t'])
        speed_by_e_by_t = ego_by_e_by_t[:, :, SPEED]
        heading_by_e_by_t = ego_by_e_by_t[:, :, HEADING]
        forward_speed_by_e_by_t = (
            speed_by_e_by_t * jnp.cos(heading_by_e_by_t)
        )
        nz_speed_by_e_by_t = jnp.clip(
            (forward_speed_by_e_by_t - 20.0) / 10.0, 0, 1,
        )
        processed['nz_speed'] = float(
            (nz_speed_by_e_by_t * vital).sum() / vital.sum(),
        )

        lane_by_e_by_t = ego_by_e_by_t[:, :, TARGET_LANE_IDX]
        right_lane_by_e_by_t = lane_by_e_by_t / (self.env.n_lanes - 1)
        processed['nz_mean_lane'] = float(
            (right_lane_by_e_by_t * vital).sum() / vital.sum(),
        )

        # nz_return: normalized discounted return decomposed into
        # survival, speed, right_lane components
        score_denom = 1.5  # 0.4 + 0.1 - (-1.0)
        discount = rollout.agent_configs[0].discount
        n_ts_per_e = vital.shape[0]
        t_by_t = jnp.arange(n_ts_per_e)
        discount_by_e_by_t = (discount ** t_by_t)[:, None]

        vital_f = vital.astype(jnp.float32)
        survival_by_e_by_t = vital_f / score_denom
        speed_component_by_e_by_t = (
            0.4 * nz_speed_by_e_by_t * vital_f / score_denom
        )
        right_lane_component_by_e_by_t = (
            0.1 * right_lane_by_e_by_t * vital_f / score_denom
        )
        nz_norm = float(1 - discount)

        nz_survival = float(
            (survival_by_e_by_t * discount_by_e_by_t)
            .sum(axis=0).mean() * nz_norm,
        )
        nz_speed = float(
            (speed_component_by_e_by_t * discount_by_e_by_t)
            .sum(axis=0).mean() * nz_norm,
        )
        nz_right_lane = float(
            (right_lane_component_by_e_by_t * discount_by_e_by_t)
            .sum(axis=0).mean() * nz_norm,
        )
        processed['nz_return'] = nz_survival + nz_speed + nz_right_lane
        processed['nz_return.survival'] = nz_survival
        processed['nz_return.speed'] = nz_speed
        processed['nz_return.right_lane'] = nz_right_lane

        brain = self.population[0]
        for behavior in self.behaviors:
            score = evaluate_behavior(
                brain.get_p_by_action_by_i_observation, behavior,
            )
            processed[f'behavior.{behavior["name"]}'] = score

        self.trek.epochia_writer.write(processed)

        self.report_sample_es(
            state_data, n_sample_es_per_epoch=n_sample_es_per_epoch,
        )

    def report_sample_es(self, state_data: dict, *,
                         n_sample_es_per_epoch: int = 1) -> None:
        if n_sample_es_per_epoch == 0:
            return

        rollout = self.rollout.tail
        ego_np = np.asarray(state_data['ego_state_by_e_by_t'])
        npcs_np = np.asarray(state_data['npc_states_by_e_by_t'])
        crashed_np = np.asarray(state_data['crashed_by_e_by_t'])
        vital_np = np.asarray(rollout.vital_by_e_by_t)

        reward_np = np.asarray(
            rollout.reward_by_agent_by_e_by_t[:, :, 0],
        )
        action_np = np.asarray(
            rollout.action_by_agent_by_e_by_t[:, :, 0],
        )
        p_by_action_np = np.asarray(
            rollout.p_by_action_by_agent_by_e_by_t[:, :, 0],
        )
        v_np = np.asarray(rollout.v_by_agent_by_e_by_t[:, :, 0])
        chosen_p_np = np.asarray(
            rollout.chosen_action_p_by_agent_by_e_by_t[:, :, 0],
        )

        agent_config = rollout.agent_configs[0]
        advantage_by_e_by_t = rollout.calculate_advantage_by_e_by_t(
            0, agent_config,
        )
        adv_np = np.asarray(advantage_by_e_by_t)
        vital_f = vital_np.astype(np.float32)
        mean_adv = float((adv_np * vital_f).sum() / vital_f.sum())
        std_adv = float(np.sqrt(
            ((adv_np - mean_adv) ** 2 * vital_f).sum()
            / vital_f.sum()
        ))

        # Discounted returns
        return_by_e_by_t = rollout.calculate_return_by_e_by_t(
            0, agent_config,
        )
        return_np = np.asarray(return_by_e_by_t)

        # Attention weights
        brain = self.population[0]
        obs_by_e_by_t = rollout.observation_by_agent_by_e_by_t[
            :, :, 0
        ]
        attention_by_e_by_t = brain.get_attention_weights(
            obs_by_e_by_t,
        )
        if attention_by_e_by_t is not None:
            attention_np = np.asarray(
                jax.device_get(attention_by_e_by_t),
            )
        else:
            attention_np = None

        n_es = rollout.n_es
        sample_indices = [
            i * n_es // n_sample_es_per_epoch
            for i in range(n_sample_es_per_epoch)
        ]

        n_sub_ts = self.env.n_sub_ts_per_t
        n_npcs = self.env.n_npcs
        n_ts = rollout.n_ts_per_e

        alive_pairs = []
        for e_idx in sample_indices:
            for t in range(n_ts):
                if not vital_np[t, e_idx]:
                    break
                action_idx = int(action_np[t, e_idx, 0])
                crashed_before = (
                    bool(crashed_np[t - 1, e_idx]) if t > 0
                    else False
                )
                crashed_after = bool(crashed_np[t, e_idx])
                alive_pairs.append(
                    (e_idx, t, action_idx,
                     crashed_before, crashed_after),
                )

        if alive_pairs:
            all_sub_ego, all_sub_npcs, all_sub_crashed = (
                self._batch_get_sub_states(
                    state_data, alive_pairs,
                )
            )

        rows = []
        for b, (e_idx, t, action_idx, crashed_before,
                crashed_after) in enumerate(alive_pairs):
            adv = float(adv_np[t, e_idx])
            nz_adv = (adv - mean_adv) / (std_adv + 1e-8)

            carry = {
                'p.left': float(p_by_action_np[t, e_idx, 0]),
                'p.idle': float(p_by_action_np[t, e_idx, 1]),
                'p.right': float(p_by_action_np[t, e_idx, 2]),
                'p.faster': float(p_by_action_np[t, e_idx, 3]),
                'p.slower': float(p_by_action_np[t, e_idx, 4]),
            }

            # Crash detection: last vital step before termination
            is_last_vital = (
                t + 1 >= n_ts or not vital_np[t + 1, e_idx]
            )
            is_crash_step = (
                is_last_vital and t < n_ts - 1
            )

            row = {
                'epoch': self.epoch,
                'e': int(e_idx),
                't': float(t),
                'reward': float(reward_np[t, e_idx]),
                'state.crashed': crashed_before,
                'action': str(action_idx),
                'action_name': ACTION_NAMES[action_idx],
                **carry,
                'v': float(v_np[t, e_idx]),
                'return': float(return_np[t, e_idx]),
                'tendency': float(
                    np.log(chosen_p_np[t, e_idx] + 1e-8),
                ),
                'advantage': adv,
                'nz_advantage': nz_adv,
                'state.ego_x': float(ego_np[t, e_idx, X]),
                'state.ego_y': float(ego_np[t, e_idx, Y]),
                'state.ego_heading': float(
                    ego_np[t, e_idx, HEADING],
                ),
                'state.ego_speed': float(
                    ego_np[t, e_idx, SPEED],
                ),
            }

            row['crash_reward'] = (
                float(reward_np[t, e_idx]) if is_crash_step
                else None
            )

            # Attention weights mapped to physical NPC indices
            attention_carry = {}
            if attention_np is not None:
                weights = attention_np[t, e_idx]
                ego_x = float(ego_np[t, e_idx, X])
                ego_y = float(ego_np[t, e_idx, Y])
                distances = [
                    (
                        (float(npcs_np[t, e_idx, i, X]) - ego_x)
                        ** 2
                        + (float(npcs_np[t, e_idx, i, Y]) - ego_y)
                        ** 2
                    ) ** 0.5
                    for i in range(n_npcs)
                ]
                sorted_npc_indices = sorted(
                    range(n_npcs), key=lambda i: distances[i],
                )
                # weights[0] = ego self-attention, skip
                n_npc_obs = min(n_npcs, len(weights) - 1)
                observed_set = {}
                for slot in range(n_npc_obs):
                    npc_idx = sorted_npc_indices[slot]
                    observed_set[npc_idx] = float(
                        weights[slot + 1],
                    )
                for i in range(n_npcs):
                    attention_carry[
                        f'state.npc{i}_attention'
                    ] = observed_set.get(i, None)
            row.update(attention_carry)

            for i in range(n_npcs):
                row[f'state.npc{i}_x'] = float(
                    npcs_np[t, e_idx, i, X],
                )
                row[f'state.npc{i}_y'] = float(
                    npcs_np[t, e_idx, i, Y],
                )
                row[f'state.npc{i}_heading'] = float(
                    npcs_np[t, e_idx, i, HEADING],
                )
                row[f'state.npc{i}_speed'] = float(
                    npcs_np[t, e_idx, i, SPEED],
                )
            rows.append(row)

            # Sub-step rows
            sub_ego = all_sub_ego[b]
            sub_npcs = all_sub_npcs[b]
            sub_crashed = all_sub_crashed[b]
            crash_seen = crashed_before
            sub_carry = dict(carry)
            if attention_np is not None:
                sub_carry.update(attention_carry)
            for s in range(n_sub_ts - 1):
                sub_t = float(t) + (s + 1) / n_sub_ts
                crash_seen = crash_seen or bool(sub_crashed[s])
                if s == n_sub_ts - 2:
                    crash_seen = crash_seen or crashed_after
                sub_row = {
                    'epoch': self.epoch,
                    'e': int(e_idx),
                    't': sub_t,
                    'state.crashed': crash_seen,
                    'state.ego_x': float(sub_ego[s, X]),
                    'state.ego_y': float(sub_ego[s, Y]),
                    'state.ego_heading': float(
                        sub_ego[s, HEADING],
                    ),
                    'state.ego_speed': float(
                        sub_ego[s, SPEED],
                    ),
                }
                if not crash_seen:
                    sub_row.update(sub_carry)
                for i in range(n_npcs):
                    sub_row[f'state.npc{i}_x'] = float(
                        sub_npcs[s, i, X],
                    )
                    sub_row[f'state.npc{i}_y'] = float(
                        sub_npcs[s, i, Y],
                    )
                    sub_row[f'state.npc{i}_heading'] = float(
                        sub_npcs[s, i, HEADING],
                    )
                    sub_row[f'state.npc{i}_speed'] = float(
                        sub_npcs[s, i, SPEED],
                    )
                rows.append(sub_row)
                if crash_seen:
                    break

        self.trek.sample_es_writer.write_batch(rows)

    def _batch_get_sub_states(self, state_data, alive_pairs):
        from highjax.stepper import step_physics_with_sub_states
        from highjax import kinematics

        batch_ego = jnp.stack([
            state_data['ego_state_by_e_by_t'][t, e_idx]
            for e_idx, t, _, _, _ in alive_pairs
        ])
        batch_npcs = jnp.stack([
            state_data['npc_states_by_e_by_t'][t, e_idx]
            for e_idx, t, _, _, _ in alive_pairs
        ])
        batch_crashed = jnp.array(
            [cb for _, _, _, cb, _ in alive_pairs],
        )
        batch_actions = jnp.array(
            [a for _, _, a, _, _ in alive_pairs], dtype=jnp.int32,
        )

        def _single(ego, npcs, crashed, action):
            ego_after = kinematics.execute_action(
                ego, action, self.env.n_lanes,
            )
            _, _, _, sub_ego, sub_npcs, sub_crashed = (
                step_physics_with_sub_states(
                    ego_after, npcs, crashed,
                    self.env.highway_lanes, self.env, self.params,
                )
            )
            return sub_ego, sub_npcs, sub_crashed

        all_sub_ego, all_sub_npcs, all_sub_crashed = jax.vmap(
            _single,
        )(batch_ego, batch_npcs, batch_crashed, batch_actions)

        return (
            np.asarray(all_sub_ego),
            np.asarray(all_sub_npcs),
            np.asarray(all_sub_crashed),
        )
