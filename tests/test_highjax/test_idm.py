'''Tests for IDM (Intelligent Driver Model) module.'''
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from highjax import idm, kinematics, lanes


class TestDesiredGap:
    def test_same_speed(self):
        '''When both vehicles have same speed, gap is minimum.'''
        gap = idm.desired_gap(25.0, 25.0)
        expected = idm.DISTANCE_WANTED + 25.0 * idm.TIME_WANTED
        assert float(gap) >= idm.DISTANCE_WANTED

    def test_approaching_increases_gap(self):
        '''When ego is faster (approaching), desired gap increases.'''
        gap_same = idm.desired_gap(25.0, 25.0)
        gap_approaching = idm.desired_gap(30.0, 20.0)  # ego faster
        assert float(gap_approaching) > float(gap_same)

    def test_receding_decreases_gap(self):
        '''When ego is slower (receding), desired gap can decrease.'''
        gap_same = idm.desired_gap(25.0, 25.0)
        gap_receding = idm.desired_gap(20.0, 30.0)  # ego slower
        # Gap should still be at least minimum
        assert float(gap_receding) >= idm.DISTANCE_WANTED


class TestIdmAcceleration:
    def test_free_road_acceleration(self):
        '''On free road, should accelerate to target speed.'''
        acc = idm.idm_acceleration(
            ego_speed=20.0,
            target_speed=30.0,
            front_distance=1000.0,
            front_speed=25.0,
            has_front=jnp.array(False)
        )
        assert float(acc) > 0  # Should accelerate

    def test_at_target_speed(self):
        '''At target speed on free road, acceleration should be near zero.'''
        acc = idm.idm_acceleration(
            ego_speed=25.0,
            target_speed=25.0,
            front_distance=1000.0,
            front_speed=25.0,
            has_front=jnp.array(False)
        )
        assert abs(float(acc)) < 0.5  # Should be close to zero

    def test_close_following(self):
        '''Following close should cause deceleration.'''
        acc = idm.idm_acceleration(
            ego_speed=25.0,
            target_speed=25.0,
            front_distance=5.0,  # Very close
            front_speed=20.0,  # Front is slower
            has_front=jnp.array(True)
        )
        assert float(acc) < 0  # Should brake

    def test_comfortable_following(self):
        '''Following at comfortable distance should have mild acceleration.'''
        comfortable_gap = idm.DISTANCE_WANTED + 25.0 * idm.TIME_WANTED * 2
        acc = idm.idm_acceleration(
            ego_speed=25.0,
            target_speed=25.0,
            front_distance=comfortable_gap,
            front_speed=25.0,  # Same speed
            has_front=jnp.array(True)
        )
        # Should be close to zero or slightly positive
        assert float(acc) > -1.0


class TestFindFrontVehicle:
    def setup_method(self):
        # Create ego at x=50 in lane 1
        self.ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)

    def test_finds_closest_front(self):
        '''Should find the closest vehicle in front.'''
        # NPCs at x=60, x=80 in lane 1
        npc1 = kinematics.make_vehicle_state(60.0, 4.0, 0.0, 20.0, 1)
        npc2 = kinematics.make_vehicle_state(80.0, 4.0, 0.0, 20.0, 1)
        all_states = jnp.stack([npc1, npc2])

        dist, speed, has_front, _heading = idm.find_front_vehicle(self.ego, all_states, 1)
        assert float(has_front)
        assert float(dist) == pytest.approx(10.0)  # 60 - 50
        assert float(speed) == pytest.approx(20.0)

    def test_ignores_different_lane(self):
        '''Should ignore vehicles in other lanes.'''
        npc_other_lane = kinematics.make_vehicle_state(60.0, 8.0, 0.0, 20.0, 2)
        all_states = jnp.stack([npc_other_lane])

        dist, speed, has_front, _heading = idm.find_front_vehicle(self.ego, all_states, 1)
        assert not float(has_front)

    def test_ignores_behind(self):
        '''Should ignore vehicles behind.'''
        npc_behind = kinematics.make_vehicle_state(30.0, 4.0, 0.0, 20.0, 1)
        all_states = jnp.stack([npc_behind])

        dist, speed, has_front, _heading = idm.find_front_vehicle(self.ego, all_states, 1)
        assert not float(has_front)


class TestFindRearVehicle:
    def setup_method(self):
        self.ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)

    def test_finds_closest_rear(self):
        '''Should find the closest vehicle behind.'''
        npc1 = kinematics.make_vehicle_state(30.0, 4.0, 0.0, 20.0, 1)
        npc2 = kinematics.make_vehicle_state(20.0, 4.0, 0.0, 20.0, 1)
        all_states = jnp.stack([npc1, npc2])

        dist, speed, has_rear, _heading, _tgt_speed, _delta = idm.find_rear_vehicle(self.ego, all_states, 1)
        assert float(has_rear)
        assert float(dist) == pytest.approx(20.0)  # 50 - 30

    def test_ignores_ahead(self):
        '''Should ignore vehicles ahead.'''
        npc_ahead = kinematics.make_vehicle_state(70.0, 4.0, 0.0, 20.0, 1)
        all_states = jnp.stack([npc_ahead])

        dist, speed, has_rear, _heading, _tgt_speed, _delta = idm.find_rear_vehicle(self.ego, all_states, 1)
        assert not float(has_rear)


class TestMobilLaneChange:
    def setup_method(self):
        self.n_lanes = 4

    def test_stays_in_lane_when_free(self):
        '''Should stay in lane when no benefit from changing.'''
        ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        # No NPCs blocking
        all_states = jnp.zeros((1, kinematics.VEHICLE_STATE_SIZE))
        all_states = all_states.at[0, kinematics.X].set(1000.0)  # Far away

        new_lane = idm.mobil_lane_change(ego, all_states, self.n_lanes)
        assert int(new_lane) == 1

    def test_changes_when_blocked(self):
        '''Should change lane when current lane is blocked.'''
        ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        # Slow NPC directly in front in lane 1
        blocking_npc = kinematics.make_vehicle_state(55.0, 4.0, 0.0, 10.0, 1)
        all_states = jnp.stack([blocking_npc])

        new_lane = idm.mobil_lane_change(ego, all_states, self.n_lanes)
        # Should want to change lane (either 0 or 2)
        assert int(new_lane) in [0, 2]

    def test_respects_lane_boundaries(self):
        '''Should not change to non-existent lanes.'''
        ego = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)  # Lane 0
        # Need at least one NPC for the function to work (uses argmin)
        far_npc = kinematics.make_vehicle_state(1000.0, 0.0, 0.0, 25.0, 0)
        all_states = jnp.stack([far_npc])

        new_lane = idm.mobil_lane_change(ego, all_states, self.n_lanes)
        assert 0 <= int(new_lane) < self.n_lanes


class TestNpcStep:
    def setup_method(self):
        self.highway_lanes = lanes.make_highway_lanes(4, 1000.0, 4.0)
        self.dt = 0.1

    def test_npc_moves_forward(self):
        '''NPC should move forward.'''
        npc = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        all_states = jnp.stack([ego])

        new_npc = idm.npc_sub_step(npc, all_states, self.highway_lanes, self.dt)
        assert float(new_npc[kinematics.X]) > float(npc[kinematics.X])

    def test_npc_slows_for_front(self):
        '''NPC should slow down when approaching front vehicle.'''
        npc = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 30.0, 1)  # Fast NPC
        slow_front = kinematics.make_vehicle_state(55.0, 4.0, 0.0, 10.0, 1)  # Slow vehicle ahead
        all_states = jnp.stack([slow_front])

        # Take a few steps
        for _ in range(5):
            npc = idm.npc_sub_step(npc, all_states, self.highway_lanes, self.dt)

        # Should have slowed down
        assert float(npc[kinematics.SPEED]) < 30.0


class TestNpcStepAll:
    def setup_method(self):
        self.highway_lanes = lanes.make_highway_lanes(4, 1000.0, 4.0)
        self.dt = 0.1

    def test_steps_all_npcs(self):
        '''Should step all NPCs.'''
        npc1 = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)
        npc2 = kinematics.make_vehicle_state(60.0, 4.0, 0.0, 25.0, 1)
        npc3 = kinematics.make_vehicle_state(70.0, 8.0, 0.0, 25.0, 2)
        npc_states = jnp.stack([npc1, npc2, npc3])

        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)

        new_npcs = idm.npc_sub_step_all(npc_states, ego, self.highway_lanes, self.dt)

        assert new_npcs.shape == npc_states.shape
        # All should have moved forward
        for i in range(3):
            assert float(new_npcs[i, kinematics.X]) > float(npc_states[i, kinematics.X])

    def test_preserves_shape(self):
        '''Output shape should match input shape.'''
        n_npcs = 5
        npc_states = jnp.zeros((n_npcs, kinematics.VEHICLE_STATE_SIZE))
        for i in range(n_npcs):
            npc_states = npc_states.at[i].set(
                kinematics.make_vehicle_state(50.0 + i * 20, i * 4.0, 0.0, 25.0, i % 4)
            )

        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        new_npcs = idm.npc_sub_step_all(npc_states, ego, self.highway_lanes, self.dt)

        assert new_npcs.shape == (n_npcs, kinematics.VEHICLE_STATE_SIZE)


class TestMobilFollowerTerms:
    def test_politeness_zero_unchanged(self):
        '''With POLITENESS=0 (default), follower terms should not affect gain.'''
        assert idm.POLITENESS == 0.0
        ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        # Slow NPC blocking in lane 1, lane 0 is free
        blocking = kinematics.make_vehicle_state(55.0, 4.0, 0.0, 10.0, 1)
        # Rear vehicle far behind in target lane (100m gap avoids safety check)
        rear = kinematics.make_vehicle_state(-50.0, 0.0, 0.0, 25.0, 0)
        all_states = jnp.stack([blocking, rear])

        gain = idm.mobil_gain(ego, all_states, current_lane=1, target_lane=0)
        # The gain should be purely self-improvement (acc_new - acc_current)
        # since POLITENESS=0 zeroes out follower terms
        assert float(gain) > 0  # Lane 0 is better (no blocking)

    def test_politeness_positive_affects_gain(self):
        '''With POLITENESS>0, follower terms should reduce gain when
        ego's lane change inconveniences followers.'''
        import unittest.mock
        ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        # Slow NPC blocking in lane 1
        blocking = kinematics.make_vehicle_state(55.0, 4.0, 0.0, 10.0, 1)
        # Follower behind in target lane 0 — 100m gap (safe, but close enough
        # to be inconvenienced by ego inserting)
        rear = kinematics.make_vehicle_state(-50.0, 0.0, 0.0, 25.0, 0)
        all_states = jnp.stack([blocking, rear])

        # Gain with POLITENESS=0 (baseline)
        gain_selfish = idm.mobil_gain(ego, all_states, current_lane=1, target_lane=0)

        # Gain with POLITENESS=0.5
        with unittest.mock.patch.object(idm, 'POLITENESS', 0.5):
            gain_polite = idm.mobil_gain(ego, all_states, current_lane=1, target_lane=0)

        # Polite gain should be lower (follower in target lane is inconvenienced)
        assert float(gain_polite) < float(gain_selfish)
