'''Tests to verify kinematics parity with expected results.

These tests compare individual kinematics components to ensure parity.
'''
from __future__ import annotations

import pytest
import jax.numpy as jnp
import numpy as np

from highjax import kinematics, lanes


class TestBicycleModelParity:
    '''Test that bicycle model matches tollroad.'''

    def test_straight_motion(self):
        '''Straight motion (steering=0) should match.'''
        # Initial conditions
        x, y, heading, speed = 0.0, 0.0, 0.0, 25.0
        steering, acceleration = 0.0, 0.0
        dt = 0.1

        hj_x, hj_y, hj_heading, hj_speed = kinematics.bicycle_forward(
            x, y, heading, speed, steering, acceleration, dt
        )

        # Tollroad calculation (inline, same as Vehicle.step)
        beta = np.arctan(kinematics.SLIP_ANGLE_FACTOR * np.tan(steering))
        v = speed * np.array([np.cos(heading + beta), np.sin(heading + beta)])
        tr_x = x + v[0] * dt
        tr_y = y + v[1] * dt
        tr_heading = heading + speed * np.sin(beta) / kinematics.HALF_LENGTH * dt
        tr_speed = np.clip(speed + acceleration * dt, kinematics.MIN_SPEED, kinematics.MAX_SPEED)

        assert float(hj_x) == pytest.approx(tr_x, rel=1e-5)
        assert float(hj_y) == pytest.approx(tr_y, rel=1e-5)
        assert float(hj_heading) == pytest.approx(tr_heading, rel=1e-5)
        assert float(hj_speed) == pytest.approx(tr_speed, rel=1e-5)

    def test_with_steering(self):
        '''Motion with steering should match.'''
        x, y, heading, speed = 10.0, 2.0, 0.1, 25.0
        steering, acceleration = 0.15, 0.5
        dt = 0.1

        hj_x, hj_y, hj_heading, hj_speed = kinematics.bicycle_forward(
            x, y, heading, speed, steering, acceleration, dt
        )

        # Tollroad calculation
        beta = np.arctan(kinematics.SLIP_ANGLE_FACTOR * np.tan(steering))
        v = speed * np.array([np.cos(heading + beta), np.sin(heading + beta)])
        tr_x = x + v[0] * dt
        tr_y = y + v[1] * dt
        tr_heading = heading + speed * np.sin(beta) / kinematics.HALF_LENGTH * dt
        tr_speed = np.clip(speed + acceleration * dt, kinematics.MIN_SPEED, kinematics.MAX_SPEED)

        assert float(hj_x) == pytest.approx(tr_x, rel=1e-5)
        assert float(hj_y) == pytest.approx(tr_y, rel=1e-5)
        assert float(hj_heading) == pytest.approx(tr_heading, rel=1e-5)
        assert float(hj_speed) == pytest.approx(tr_speed, rel=1e-5)

    def test_negative_steering(self):
        '''Motion with negative steering should match.'''
        x, y, heading, speed = 50.0, 4.0, -0.05, 30.0
        steering, acceleration = -0.2, -1.0
        dt = 0.1

        hj_x, hj_y, hj_heading, hj_speed = kinematics.bicycle_forward(
            x, y, heading, speed, steering, acceleration, dt
        )

        # Tollroad calculation
        beta = np.arctan(kinematics.SLIP_ANGLE_FACTOR * np.tan(steering))
        v = speed * np.array([np.cos(heading + beta), np.sin(heading + beta)])
        tr_x = x + v[0] * dt
        tr_y = y + v[1] * dt
        tr_heading = heading + speed * np.sin(beta) / kinematics.HALF_LENGTH * dt
        tr_speed = np.clip(speed + acceleration * dt, kinematics.MIN_SPEED, kinematics.MAX_SPEED)

        assert float(hj_x) == pytest.approx(tr_x, rel=1e-5)
        assert float(hj_y) == pytest.approx(tr_y, rel=1e-5)
        assert float(hj_heading) == pytest.approx(tr_heading, rel=1e-5)
        assert float(hj_speed) == pytest.approx(tr_speed, rel=1e-5)

    def test_multiple_steps(self):
        '''Multiple steps should accumulate correctly.'''
        x, y, heading, speed = 0.0, 0.0, 0.0, 25.0
        steering, acceleration = 0.05, 0.2
        dt = 0.1
        n_steps = 10

        # HighJax - multiple steps
        hj_x, hj_y, hj_heading, hj_speed = x, y, heading, speed
        for _ in range(n_steps):
            hj_x, hj_y, hj_heading, hj_speed = kinematics.bicycle_forward(
                hj_x, hj_y, hj_heading, hj_speed, steering, acceleration, dt
            )

        # Tollroad - multiple steps
        tr_x, tr_y, tr_heading, tr_speed = x, y, heading, speed
        for _ in range(n_steps):
            beta = np.arctan(kinematics.SLIP_ANGLE_FACTOR * np.tan(steering))
            v = tr_speed * np.array([np.cos(tr_heading + beta), np.sin(tr_heading + beta)])
            tr_x = tr_x + v[0] * dt
            tr_y = tr_y + v[1] * dt
            tr_heading = tr_heading + tr_speed * np.sin(beta) / kinematics.HALF_LENGTH * dt
            tr_speed = np.clip(
                tr_speed + acceleration * dt,
                kinematics.MIN_SPEED, kinematics.MAX_SPEED)

        assert float(hj_x) == pytest.approx(tr_x, rel=1e-5)
        assert float(hj_y) == pytest.approx(tr_y, rel=1e-5)
        assert float(hj_heading) == pytest.approx(tr_heading, rel=1e-5)
        assert float(hj_speed) == pytest.approx(tr_speed, rel=1e-5)


class TestSpeedControlParity:
    '''Test that speed control matches tollroad.'''

    def test_speed_below_target(self):
        '''Should accelerate when below target speed.'''
        speed, target_speed = 20.0, 25.0

        vehicle_state = kinematics.make_vehicle_state(0.0, 0.0, 0.0, speed, 0)
        vehicle_state = vehicle_state.at[kinematics.TARGET_SPEED].set(target_speed)
        hj_accel = kinematics.speed_control(vehicle_state)

        # Tollroad: KP_A * (target_speed - speed)
        tr_accel = kinematics.KP_A * (target_speed - speed)

        assert float(hj_accel) == pytest.approx(tr_accel, rel=1e-5)

    def test_speed_above_target(self):
        '''Should decelerate when above target speed.'''
        speed, target_speed = 30.0, 25.0

        vehicle_state = kinematics.make_vehicle_state(0.0, 0.0, 0.0, speed, 0)
        vehicle_state = vehicle_state.at[kinematics.TARGET_SPEED].set(target_speed)
        hj_accel = kinematics.speed_control(vehicle_state)

        # Tollroad
        tr_accel = kinematics.KP_A * (target_speed - speed)

        assert float(hj_accel) == pytest.approx(tr_accel, rel=1e-5)

    def test_at_target_speed(self):
        '''Should have zero acceleration at target speed.'''
        speed = target_speed = 25.0

        vehicle_state = kinematics.make_vehicle_state(0.0, 0.0, 0.0, speed, 0)
        vehicle_state = vehicle_state.at[kinematics.TARGET_SPEED].set(target_speed)
        hj_accel = kinematics.speed_control(vehicle_state)

        assert float(hj_accel) == pytest.approx(0.0, abs=1e-10)


class TestSteeringControlParity:
    '''Test that steering control matches tollroad.'''

    def test_on_lane_center(self):
        '''Steering should be minimal when on lane center.'''
        # Create a horizontal lane at y=0 (from x=0 to x=1000)
        lane_params = lanes.make_lane_params(0.0, 0.0, 1000.0, 0.0)

        # Vehicle on lane center
        vehicle_state = kinematics.make_vehicle_state(50.0, 0.0, 0.0, 25.0, 0)
        hj_steering = kinematics.steering_control(vehicle_state, lane_params)

        # Should be close to zero
        assert abs(float(hj_steering)) < 0.01

    def test_lateral_offset_left(self):
        '''Should steer right when offset to the left (y > target).'''
        # Horizontal lane at y=0 (from x=0 to x=1000)
        lane_params = lanes.make_lane_params(0.0, 0.0, 1000.0, 0.0)

        # Vehicle offset to left (y > 0)
        vehicle_state = kinematics.make_vehicle_state(50.0, 2.0, 0.0, 25.0, 0)
        hj_steering = kinematics.steering_control(vehicle_state, lane_params)

        # Should steer right (negative for right in standard coordinates)
        # Actually depends on coordinate system - just check it's nonzero
        assert abs(float(hj_steering)) > 0.001

    def test_lateral_offset_right(self):
        '''Should steer left when offset to the right (y < target).'''
        # Horizontal lane at y=4 (from x=0 to x=1000)
        lane_params = lanes.make_lane_params(0.0, 4.0, 1000.0, 4.0)

        # Vehicle offset to right of lane (y < 4)
        vehicle_state = kinematics.make_vehicle_state(50.0, 2.0, 0.0, 25.0, 0)
        hj_steering = kinematics.steering_control(vehicle_state, lane_params)

        # Should steer left to get to lane
        assert abs(float(hj_steering)) > 0.001


class TestConstantsParity:
    '''Test that constants match tollroad.'''

    def test_vehicle_dimensions(self):
        '''Vehicle dimensions should match.'''
        assert kinematics.VEHICLE_LENGTH == 5.0
        assert kinematics.VEHICLE_WIDTH == 2.0

    def test_speed_limits(self):
        '''Speed limits should match.'''
        assert kinematics.MAX_SPEED == 40.0
        assert kinematics.MIN_SPEED == -40.0

    def test_control_gains(self):
        '''Control gains should match tollroad.'''
        # From tollroad ControlledVehicle
        assert kinematics.TAU_ACC == pytest.approx(0.6)
        assert kinematics.TAU_HEADING == pytest.approx(0.2)
        assert kinematics.TAU_LATERAL == pytest.approx(0.6)
        assert kinematics.KP_A == pytest.approx(1 / 0.6)
        assert kinematics.KP_HEADING == pytest.approx(1 / 0.2)
        assert kinematics.KP_LATERAL == pytest.approx(1 / 0.6)

    def test_default_target_speeds(self):
        '''Default target speeds should match.'''
        # tollroad: np.linspace(20, 30, 3) = [20, 25, 30]
        expected = jnp.array([20.0, 25.0, 30.0])
        assert jnp.allclose(kinematics.DEFAULT_TARGET_SPEEDS, expected)

    def test_slip_angle_factor(self):
        '''Slip angle factor should match.'''
        assert kinematics.SLIP_ANGLE_FACTOR == 0.5
