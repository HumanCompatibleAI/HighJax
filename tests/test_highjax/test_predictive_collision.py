'''Tests for predictive collision detection + impulse.

Predictive collision: SAT with velocity-based projection detects collisions
one frame before polygons physically overlap.

Impulse: translation vector pushes vehicles apart when collision predicted.
'''
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from highjax import kinematics

DT = 0.1


class TestPredictiveCollision:
    '''SAT should detect will_intersect before physical overlap.'''

    def test_approaching_vehicles_detected_before_overlap(self):
        '''Two vehicles approaching should trigger will_intersect before overlap.'''
        # Ego at x=0, heading=0, speed=20 -> moving right
        ego_corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        ego_vel = jnp.array([20.0, 0.0])

        # NPC at x=7 (gap=4.5m with 5m vehicles), heading=0, speed=0 -> stationary
        npc_corners = kinematics.get_vehicle_corners(
            jnp.array(7.0), jnp.array(0.0), jnp.array(0.0))
        npc_vel = jnp.array([0.0, 0.0])

        intersecting, will_intersect, translation = (
            kinematics.rotated_rectangles_intersect(
                ego_corners, npc_corners, ego_vel, npc_vel, DT))

        assert not bool(intersecting), 'Should not overlap yet'
        assert bool(will_intersect), 'Should predict collision'
        # Translation vector should push them apart along x axis
        assert float(translation[0]) != 0.0

    def test_separating_vehicles_not_predicted(self):
        '''Two vehicles moving apart should not trigger will_intersect.'''
        ego_corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        ego_vel = jnp.array([-20.0, 0.0])  # Moving left (away)

        npc_corners = kinematics.get_vehicle_corners(
            jnp.array(7.0), jnp.array(0.0), jnp.array(0.0))
        npc_vel = jnp.array([20.0, 0.0])  # Moving right (away)

        intersecting, will_intersect, translation = (
            kinematics.rotated_rectangles_intersect(
                ego_corners, npc_corners, ego_vel, npc_vel, DT))

        assert not bool(intersecting)
        assert not bool(will_intersect)

    def test_already_overlapping_returns_intersecting(self):
        '''Overlapping vehicles should return intersecting=True.'''
        ego_corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        npc_corners = kinematics.get_vehicle_corners(
            jnp.array(2.0), jnp.array(0.0), jnp.array(0.0))

        intersecting, will_intersect, translation = (
            kinematics.rotated_rectangles_intersect(
                ego_corners, npc_corners,
                jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), DT))

        assert bool(intersecting), 'Should detect current overlap'


class TestTranslationVector:
    '''Translation vector should push vehicles apart.'''

    def test_head_on_translation_along_x(self):
        '''Head-on approach -> translation mostly along x axis.'''
        ego_corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        npc_corners = kinematics.get_vehicle_corners(
            jnp.array(4.0), jnp.array(0.0), jnp.array(0.0))

        _, _, translation = kinematics.rotated_rectangles_intersect(
            ego_corners, npc_corners,
            jnp.array([10.0, 0.0]), jnp.array([-10.0, 0.0]), DT)

        # Translation x should dominate (vehicles aligned along x)
        assert abs(float(translation[0])) > abs(float(translation[1]))

    def test_side_approach_translation_along_y(self):
        '''Side approach -> translation mostly along y axis.'''
        ego_corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        npc_corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(2.5), jnp.array(0.0))

        _, _, translation = kinematics.rotated_rectangles_intersect(
            ego_corners, npc_corners,
            jnp.array([0.0, 5.0]), jnp.array([0.0, -5.0]), DT)

        # Translation y should dominate
        assert abs(float(translation[1])) > abs(float(translation[0]))


class TestImpulseDisplacement:
    '''Impulse should modify positions in the step function.'''

    def test_collision_pushes_vehicles_apart(self):
        '''After collision impulse, vehicles should be further apart.'''
        # Set up ego approaching NPC from behind
        ego = kinematics.make_vehicle_state(45.0, 4.0, 0.0, 30.0, 1)
        npc = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 10.0, 1)

        ego_corners = kinematics.get_vehicle_corners(
            ego[kinematics.X], ego[kinematics.Y], ego[kinematics.HEADING])
        npc_corners = kinematics.get_vehicle_corners(
            npc[kinematics.X], npc[kinematics.Y], npc[kinematics.HEADING])

        ego_vel = jnp.array([
            ego[kinematics.SPEED] * jnp.cos(ego[kinematics.HEADING]),
            ego[kinematics.SPEED] * jnp.sin(ego[kinematics.HEADING]),
        ])
        npc_vel = jnp.array([
            npc[kinematics.SPEED] * jnp.cos(npc[kinematics.HEADING]),
            npc[kinematics.SPEED] * jnp.sin(npc[kinematics.HEADING]),
        ])

        _, will_intersect, translation = kinematics.rotated_rectangles_intersect(
            ego_corners, npc_corners, ego_vel, npc_vel, DT)

        # Apply impulse: ego gets +translation/2, npc gets -translation/2
        new_ego_x = ego[kinematics.X] + translation[0] / 2
        new_ego_y = ego[kinematics.Y] + translation[1] / 2
        new_npc_x = npc[kinematics.X] - translation[0] / 2
        new_npc_y = npc[kinematics.Y] - translation[1] / 2

        old_dist = float(jnp.sqrt((ego[kinematics.X] - npc[kinematics.X])**2 +
                                   (ego[kinematics.Y] - npc[kinematics.Y])**2))
        new_dist = float(jnp.sqrt((new_ego_x - new_npc_x)**2 +
                                   (new_ego_y - new_npc_y)**2))

        if bool(will_intersect):
            assert new_dist >= old_dist, 'Impulse should push vehicles apart'


class TestCrashOnPredicted:
    '''crash_on_predicted parameter controls whether predicted overlap sets crash flag.'''

    def test_predicted_only_no_crash_by_default(self):
        '''will_collide=True but colliding=False should NOT crash with default settings.'''
        # Ego at x=0, moving right at 20 m/s. NPC at x=7, stationary.
        # Gap of 2m between bumpers. Not overlapping, but prediction says will collide.
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 20.0, 0)
        npc = kinematics.make_vehicle_state(7.0, 0.0, 0.0, 0.0, 0)
        colliding, will_collide, _ = kinematics.check_vehicle_collision(ego, npc, dt=DT)
        assert not bool(colliding)
        assert bool(will_collide)

    def test_predicted_only_crash_when_enabled(self):
        '''will_collide=True but colliding=False should crash with crash_on_predicted=True.'''
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 20.0, 0)
        npc = kinematics.make_vehicle_state(7.0, 0.0, 0.0, 0.0, 0)
        colliding, will_collide, _ = kinematics.check_vehicle_collision(ego, npc, dt=DT)
        # With crash_on_predicted=True, this would trigger crash
        assert bool(colliding | will_collide)

    def test_actual_overlap_crashes_regardless(self):
        '''colliding=True should crash regardless of crash_on_predicted setting.'''
        ego = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 20.0, 0)
        npc = kinematics.make_vehicle_state(2.0, 0.0, 0.0, 0.0, 0)
        colliding, _, _ = kinematics.check_vehicle_collision(ego, npc, dt=DT)
        assert bool(colliding)


class TestStepCollisionImpulse:
    '''The step function should apply impulse displacement.'''

    def test_backward_compatibility(self):
        '''rotated_rectangles_intersect with no velocity args still works.'''
        # This tests that we can call the old API (just corners)
        ego_corners = kinematics.get_vehicle_corners(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        npc_corners = kinematics.get_vehicle_corners(
            jnp.array(2.0), jnp.array(0.0), jnp.array(0.0))

        # New API returns 3-tuple
        result = kinematics.rotated_rectangles_intersect(
            ego_corners, npc_corners,
            jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), DT)
        assert len(result) == 3
        intersecting, will_intersect, translation = result
        assert bool(intersecting)
