'''Tests for leader/follower search using lateral position (divergence #8).

HE uses on_lane(margin=1) -- lateral position check instead of TARGET_LANE_IDX.
A vehicle mid-lane-change should be found by both old and new lane searches.
'''
from __future__ import annotations

import pytest
import jax.numpy as jnp

from highjax import kinematics, idm


LANE_WIDTH = 4.0  # Default


class TestLateralPositionSearch:
    '''find_front/rear should use lateral position, not TARGET_LANE_IDX.'''

    def test_mid_lane_change_found_by_both_lanes(self):
        '''Vehicle physically between lanes should be found by both.'''
        ego = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)

        # NPC at y=2.0 (between lane 0 center=0 and lane 1 center=4)
        # TARGET_LANE_IDX=1 (heading to lane 1), but still close to lane 0
        npc = kinematics.make_vehicle_state(60.0, 2.0, 0.0, 20.0, 1)
        all_states = jnp.stack([npc])

        # Should be found in lane 0 (y=2 is within lane_width/2+margin of center 0)
        dist0, speed0, has0, _ = idm.find_front_vehicle(
            ego, all_states, 0, LANE_WIDTH)
        assert bool(has0), 'Should find NPC in lane 0 by position'

        # Should also be found in lane 1 (y=2 is within lane_width/2+margin of center 4)
        dist1, speed1, has1, _ = idm.find_front_vehicle(
            ego, all_states, 1, LANE_WIDTH)
        assert bool(has1), 'Should find NPC in lane 1 by position'

    def test_vehicle_clearly_in_lane_found(self):
        '''Vehicle clearly centered in a lane should be found.'''
        ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        npc = kinematics.make_vehicle_state(60.0, 4.0, 0.0, 20.0, 1)
        all_states = jnp.stack([npc])

        dist, speed, has_front, _ = idm.find_front_vehicle(
            ego, all_states, 1, LANE_WIDTH)
        assert bool(has_front)
        assert float(dist) == pytest.approx(10.0)

    def test_vehicle_far_from_lane_not_found(self):
        '''Vehicle far from lane center should not be found.'''
        ego = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)
        # NPC at y=8 (lane 2 center), searching lane 0
        npc = kinematics.make_vehicle_state(60.0, 8.0, 0.0, 20.0, 2)
        all_states = jnp.stack([npc])

        _, _, has_front, _ = idm.find_front_vehicle(
            ego, all_states, 0, LANE_WIDTH)
        assert not bool(has_front)

    def test_rear_mid_lane_change_found(self):
        '''Rear vehicle mid-lane-change should be found by both lanes.'''
        ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        # NPC behind, at y=2.0, target_lane=0 but physically between lanes
        npc = kinematics.make_vehicle_state(40.0, 2.0, 0.0, 20.0, 0)
        all_states = jnp.stack([npc])

        # Should be found in lane 1 (y=2 within range of center 4)
        dist1, speed1, has1, _, _, _ = idm.find_rear_vehicle(
            ego, all_states, 1, LANE_WIDTH)
        assert bool(has1), 'Should find rear NPC in lane 1 by position'

    def test_target_lane_idx_ignored(self):
        '''Vehicle with wrong TARGET_LANE_IDX but right position should be found.'''
        ego = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)
        # NPC at y=0 (lane 0 center) but TARGET_LANE_IDX=2
        npc = kinematics.make_vehicle_state(60.0, 0.0, 0.0, 20.0, 2)
        all_states = jnp.stack([npc])

        dist, speed, has_front, _ = idm.find_front_vehicle(
            ego, all_states, 0, LANE_WIDTH)
        assert bool(has_front), 'Should find NPC by position regardless of target_lane'
