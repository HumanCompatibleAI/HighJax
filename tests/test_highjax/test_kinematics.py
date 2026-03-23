'''Tests for kinematics module.'''
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from highjax import kinematics, lanes


class TestMakeVehicleState:
    def test_creates_correct_shape(self):
        state = kinematics.make_vehicle_state(10.0, 5.0, 0.1, 25.0, 1)
        assert state.shape == (kinematics.VEHICLE_STATE_SIZE,)

    def test_position_values(self):
        state = kinematics.make_vehicle_state(10.0, 5.0, 0.1, 25.0, 1)
        assert float(state[kinematics.X]) == 10.0
        assert float(state[kinematics.Y]) == 5.0
        assert float(state[kinematics.HEADING]) == pytest.approx(0.1)
        assert float(state[kinematics.SPEED]) == 25.0

    def test_target_lane(self):
        state = kinematics.make_vehicle_state(10.0, 5.0, 0.1, 25.0, 2)
        assert int(state[kinematics.TARGET_LANE_IDX]) == 2

    def test_speed_index_mapping(self):
        # Speed 25 should map to index 1 (middle of [20, 25, 30])
        state = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        assert int(state[kinematics.SPEED_INDEX]) == 1
        assert float(state[kinematics.TARGET_SPEED]) == 25.0


class TestSpeedToIndex:
    def test_exact_speeds(self):
        assert int(kinematics.speed_to_index(20.0)) == 0
        assert int(kinematics.speed_to_index(25.0)) == 1
        assert int(kinematics.speed_to_index(30.0)) == 2

    def test_intermediate_speeds(self):
        # Should round to nearest
        assert int(kinematics.speed_to_index(22.0)) == 0
        assert int(kinematics.speed_to_index(23.0)) == 1
        assert int(kinematics.speed_to_index(27.0)) == 1
        assert int(kinematics.speed_to_index(28.0)) == 2

    def test_out_of_range(self):
        # Should clip to valid range
        assert int(kinematics.speed_to_index(15.0)) == 0
        assert int(kinematics.speed_to_index(35.0)) == 2


class TestWrapToPi:
    def test_within_range(self):
        assert float(kinematics.wrap_to_pi(jnp.array(0.5))) == pytest.approx(0.5)
        assert float(kinematics.wrap_to_pi(jnp.array(-0.5))) == pytest.approx(-0.5)

    def test_positive_overflow(self):
        assert float(kinematics.wrap_to_pi(jnp.array(jnp.pi + 0.5))) == pytest.approx(-jnp.pi + 0.5)

    def test_negative_overflow(self):
        assert float(kinematics.wrap_to_pi(jnp.array(-jnp.pi - 0.5))) == pytest.approx(jnp.pi - 0.5)

    def test_multiple_rotations(self):
        angle = 5 * jnp.pi + 0.3
        wrapped = kinematics.wrap_to_pi(jnp.array(angle))
        assert -jnp.pi <= float(wrapped) <= jnp.pi


class TestNotZero:
    def test_large_values_unchanged(self):
        assert float(kinematics.not_zero(jnp.array(1.0))) == 1.0
        assert float(kinematics.not_zero(jnp.array(-1.0))) == -1.0

    def test_small_positive_returns_eps(self):
        result = kinematics.not_zero(jnp.array(0.001), eps=0.01)
        assert float(result) == pytest.approx(0.01)

    def test_small_negative_returns_negative_eps(self):
        result = kinematics.not_zero(jnp.array(-0.001), eps=0.01)
        assert float(result) == pytest.approx(-0.01)

    def test_zero_returns_eps(self):
        result = kinematics.not_zero(jnp.array(0.0), eps=0.01)
        assert float(result) == pytest.approx(0.01)


class TestBicycleStep:
    def test_straight_motion(self):
        '''Vehicle moving straight with no steering.'''
        dt = 0.1
        x, y, heading, speed = kinematics.bicycle_forward(
            0.0, 0.0, 0.0, 10.0,  # x, y, heading, speed
            0.0, 0.0, dt  # steering, acceleration, dt
        )
        assert float(x) == pytest.approx(1.0)  # 10 m/s * 0.1s
        assert float(y) == pytest.approx(0.0)
        assert float(heading) == pytest.approx(0.0)
        assert float(speed) == pytest.approx(10.0)

    def test_acceleration(self):
        '''Vehicle accelerating.'''
        dt = 0.1
        x, y, heading, speed = kinematics.bicycle_forward(
            0.0, 0.0, 0.0, 10.0,
            0.0, 5.0, dt  # 5 m/s^2 acceleration
        )
        assert float(speed) == pytest.approx(10.5)  # 10 + 5 * 0.1

    def test_speed_clipping(self):
        '''Speed should be clipped to limits.'''
        dt = 0.1
        # Try to exceed max speed
        _, _, _, speed = kinematics.bicycle_forward(
            0.0, 0.0, 0.0, 39.0,
            0.0, 50.0, dt
        )
        assert float(speed) <= kinematics.MAX_SPEED

        # Try to go below min speed
        _, _, _, speed = kinematics.bicycle_forward(
            0.0, 0.0, 0.0, -39.0,
            0.0, -50.0, dt
        )
        assert float(speed) >= kinematics.MIN_SPEED

    def test_turning(self):
        '''Vehicle should turn when steering is applied.'''
        dt = 0.1
        x, y, heading, speed = kinematics.bicycle_forward(
            0.0, 0.0, 0.0, 10.0,
            0.3, 0.0, dt  # Some steering angle
        )
        # Heading should change
        assert float(heading) != 0.0
        # Y position should change (turning)
        assert float(y) != 0.0


class TestExecuteAction:
    def setup_method(self):
        self.state = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        self.n_lanes = 4

    def test_idle_action(self):
        '''IDLE should not change targets.'''
        new_state = kinematics.execute_action(self.state, kinematics.ACTION_IDLE, self.n_lanes)
        assert int(new_state[kinematics.TARGET_LANE_IDX]) == 1
        assert int(new_state[kinematics.SPEED_INDEX]) == 1

    def test_lane_left(self):
        '''LANE_LEFT should decrease lane index.'''
        new_state = kinematics.execute_action(self.state, kinematics.ACTION_LANE_LEFT, self.n_lanes)
        assert int(new_state[kinematics.TARGET_LANE_IDX]) == 0

    def test_lane_right(self):
        '''LANE_RIGHT should increase lane index.'''
        new_state = kinematics.execute_action(
            self.state, kinematics.ACTION_LANE_RIGHT, self.n_lanes)
        assert int(new_state[kinematics.TARGET_LANE_IDX]) == 2

    def test_lane_left_at_boundary(self):
        '''LANE_LEFT at lane 0 should stay at 0.'''
        state = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)
        new_state = kinematics.execute_action(state, kinematics.ACTION_LANE_LEFT, self.n_lanes)
        assert int(new_state[kinematics.TARGET_LANE_IDX]) == 0

    def test_lane_right_at_boundary(self):
        '''LANE_RIGHT at last lane should stay.'''
        state = kinematics.make_vehicle_state(50.0, 12.0, 0.0, 25.0, 3)
        new_state = kinematics.execute_action(state, kinematics.ACTION_LANE_RIGHT, self.n_lanes)
        assert int(new_state[kinematics.TARGET_LANE_IDX]) == 3

    def test_faster(self):
        '''FASTER should increase speed index.'''
        new_state = kinematics.execute_action(self.state, kinematics.ACTION_FASTER, self.n_lanes)
        assert int(new_state[kinematics.SPEED_INDEX]) == 2
        assert float(new_state[kinematics.TARGET_SPEED]) == 30.0

    def test_slower(self):
        '''SLOWER should decrease speed index.'''
        new_state = kinematics.execute_action(self.state, kinematics.ACTION_SLOWER, self.n_lanes)
        assert int(new_state[kinematics.SPEED_INDEX]) == 0
        assert float(new_state[kinematics.TARGET_SPEED]) == 20.0


class TestVehicleStep:
    def setup_method(self):
        self.highway_lanes = lanes.make_highway_lanes(4, 1000.0, 4.0)
        self.dt = 0.1

    def test_moves_forward(self):
        '''Vehicle should move forward.'''
        state = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        new_state = kinematics.vehicle_step(
            state, kinematics.ACTION_IDLE, self.highway_lanes, self.dt)
        assert float(new_state[kinematics.X]) > 0.0

    def test_lane_change_updates_target(self):
        '''Lane change should update target lane.'''
        state = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)
        new_state = kinematics.vehicle_step(
            state, kinematics.ACTION_LANE_RIGHT,
            self.highway_lanes, self.dt)
        assert int(new_state[kinematics.TARGET_LANE_IDX]) == 1

    def test_multiple_steps(self):
        '''Vehicle should be able to take multiple steps.'''
        state = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        for _ in range(10):
            state = kinematics.vehicle_step(
                state, kinematics.ACTION_IDLE,
                self.highway_lanes, self.dt)
        # Should have moved forward
        assert float(state[kinematics.X]) > 20.0


class TestGetVehicleCorners:
    def test_axis_aligned_corners(self):
        '''Corners should be correct for axis-aligned vehicle (heading=0).'''
        corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        )
        assert corners.shape == (4, 2)
        # Front-left, front-right, rear-right, rear-left
        hl, hw = kinematics.VEHICLE_LENGTH / 2, kinematics.VEHICLE_WIDTH / 2
        expected = jnp.array([
            [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]
        ])
        assert jnp.allclose(corners, expected, atol=1e-5)

    def test_rotated_90_degrees(self):
        '''Corners should rotate correctly for 90 degree heading.'''
        corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(jnp.pi / 2)
        )
        hl, hw = kinematics.VEHICLE_LENGTH / 2, kinematics.VEHICLE_WIDTH / 2
        # After 90 deg rotation: x becomes -y, y becomes x
        expected = jnp.array([
            [-hw, hl], [hw, hl], [hw, -hl], [-hw, -hl]
        ])
        assert jnp.allclose(corners, expected, atol=1e-5)

    def test_translated_corners(self):
        '''Corners should be translated by position.'''
        corners = kinematics.get_vehicle_corners(
            jnp.array(10.0), jnp.array(5.0), jnp.array(0.0)
        )
        hl, hw = kinematics.VEHICLE_LENGTH / 2, kinematics.VEHICLE_WIDTH / 2
        expected = jnp.array([
            [10 + hl, 5 + hw], [10 + hl, 5 - hw],
            [10 - hl, 5 - hw], [10 - hl, 5 + hw]
        ])
        assert jnp.allclose(corners, expected, atol=1e-5)

    def test_jittable(self):
        '''get_vehicle_corners should be jittable.'''
        jit_fn = jax.jit(kinematics.get_vehicle_corners)
        corners = jit_fn(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        assert corners.shape == (4, 2)


class TestRotatedRectanglesIntersect:
    def test_same_position_intersects(self):
        '''Two rectangles at same position should intersect.'''
        corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        )
        assert bool(kinematics.rotated_rectangles_intersect(corners, corners)[0])

    def test_far_apart_no_intersect(self):
        '''Two rectangles far apart should not intersect.'''
        corners_a = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        )
        corners_b = kinematics.get_vehicle_corners(
            jnp.array(100.0), jnp.array(0.0), jnp.array(0.0)
        )
        assert not bool(kinematics.rotated_rectangles_intersect(corners_a, corners_b)[0])

    def test_adjacent_no_intersect(self):
        '''Two rectangles just touching should not intersect.'''
        corners_a = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        )
        # Place second vehicle exactly one vehicle length away
        corners_b = kinematics.get_vehicle_corners(
            jnp.array(kinematics.VEHICLE_LENGTH + 0.1), jnp.array(0.0), jnp.array(0.0)
        )
        assert not bool(kinematics.rotated_rectangles_intersect(corners_a, corners_b)[0])

    def test_overlapping_intersects(self):
        '''Two overlapping rectangles should intersect.'''
        corners_a = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        )
        # Place second vehicle half a length away (overlapping)
        corners_b = kinematics.get_vehicle_corners(
            jnp.array(kinematics.VEHICLE_LENGTH / 2), jnp.array(0.0), jnp.array(0.0)
        )
        assert bool(kinematics.rotated_rectangles_intersect(corners_a, corners_b)[0])

    def test_rotated_overlap(self):
        '''Two rotated overlapping rectangles should intersect.'''
        corners_a = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.3)
        )
        corners_b = kinematics.get_vehicle_corners(
            jnp.array(2.0), jnp.array(1.0), jnp.array(-0.2)
        )
        assert bool(kinematics.rotated_rectangles_intersect(corners_a, corners_b)[0])

    def test_rotated_no_overlap(self):
        '''Two rotated non-overlapping rectangles should not intersect.'''
        corners_a = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.3)
        )
        corners_b = kinematics.get_vehicle_corners(
            jnp.array(10.0), jnp.array(5.0), jnp.array(-0.2)
        )
        assert not bool(kinematics.rotated_rectangles_intersect(corners_a, corners_b)[0])

    def test_jittable(self):
        '''rotated_rectangles_intersect should be jittable.'''
        jit_fn = jax.jit(kinematics.rotated_rectangles_intersect)
        corners_a = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        )
        corners_b = kinematics.get_vehicle_corners(
            jnp.array(100.0), jnp.array(0.0), jnp.array(0.0)
        )
        result = jit_fn(corners_a, corners_b)
        assert not bool(result[0])


class TestCheckVehicleCollision:
    def test_colliding_vehicles(self):
        '''Two vehicles at same position should collide.'''
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        npc = kinematics.make_vehicle_state(2.0, 0.5, 0.0, 20.0, 0)
        assert bool(kinematics.check_vehicle_collision(ego, npc)[0])

    def test_non_colliding_vehicles(self):
        '''Two vehicles far apart should not collide.'''
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        npc = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 20.0, 0)
        assert not bool(kinematics.check_vehicle_collision(ego, npc)[0])

    def test_jittable(self):
        '''check_vehicle_collision should be jittable.'''
        jit_fn = jax.jit(kinematics.check_vehicle_collision)
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        npc = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 20.0, 0)
        result = jit_fn(ego, npc)
        assert not bool(result[0])


class TestCheckVehicleCollisionsAll:
    def test_no_collisions(self):
        '''Should return False when no NPCs collide.'''
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        npcs = jnp.stack([
            kinematics.make_vehicle_state(50.0, 0.0, 0.0, 20.0, 0),
            kinematics.make_vehicle_state(100.0, 4.0, 0.0, 22.0, 1),
            kinematics.make_vehicle_state(150.0, 8.0, 0.0, 18.0, 2),
        ])
        per_npc_colliding, _, _ = kinematics.check_vehicle_collisions_all(ego, npcs)
        assert not bool(jnp.any(per_npc_colliding))

    def test_one_collision(self):
        '''Should return True when one NPC collides.'''
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        npcs = jnp.stack([
            kinematics.make_vehicle_state(50.0, 0.0, 0.0, 20.0, 0),
            kinematics.make_vehicle_state(2.0, 0.5, 0.0, 22.0, 0),  # Colliding
            kinematics.make_vehicle_state(150.0, 8.0, 0.0, 18.0, 2),
        ])
        per_npc_colliding, _, _ = kinematics.check_vehicle_collisions_all(ego, npcs)
        assert bool(jnp.any(per_npc_colliding))

    def test_returns_separate_colliding_and_will_collide(self):
        '''Should return colliding and will_collide as separate arrays.'''
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 20.0, 0)
        # NPC just ahead, not overlapping, but closing due to stationary NPC
        npc = kinematics.make_vehicle_state(7.0, 0.0, 0.0, 0.0, 0)
        npcs = jnp.stack([npc])
        per_npc_colliding, per_npc_will_collide, _ = (
            kinematics.check_vehicle_collisions_all(ego, npcs, dt=0.1))
        assert not bool(per_npc_colliding[0])
        assert bool(per_npc_will_collide[0])

    def test_jittable(self):
        '''check_vehicle_collisions_all should be jittable.'''
        jit_fn = jax.jit(kinematics.check_vehicle_collisions_all)
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        npcs = jnp.stack([
            kinematics.make_vehicle_state(50.0, 0.0, 0.0, 20.0, 0),
            kinematics.make_vehicle_state(100.0, 4.0, 0.0, 22.0, 1),
        ])
        per_npc_colliding, _, _ = jit_fn(ego, npcs)
        assert not bool(jnp.any(per_npc_colliding))

    def test_vmappable(self):
        '''check_vehicle_collisions_all should work with vmap over ego states.'''
        egos = jnp.stack([
            kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0),
            kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0),
        ])
        npcs = jnp.stack([
            kinematics.make_vehicle_state(2.0, 0.5, 0.0, 20.0, 0),  # Collides with first ego
            kinematics.make_vehicle_state(100.0, 4.0, 0.0, 22.0, 1),
        ])
        results = jax.vmap(lambda e: kinematics.check_vehicle_collisions_all(e, npcs))(egos)
        per_npc_colliding_by_ego = results[0]  # (2, 2) — colliding per npc per ego
        assert bool(jnp.any(per_npc_colliding_by_ego[0]))  # First ego collides
        assert not bool(jnp.any(per_npc_colliding_by_ego[1]))  # Second ego doesn't
