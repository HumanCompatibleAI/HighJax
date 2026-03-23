'''Tests for lane geometry module.'''
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from highjax import lanes


class TestMakeLaneParams:
    def test_horizontal_lane(self):
        '''Create a lane along the x-axis.'''
        params = lanes.make_lane_params(0.0, 0.0, 100.0, 0.0, width=4.0)
        assert params.shape == (lanes.LANE_PARAMS_SIZE,)
        assert float(params[lanes.START_X]) == 0.0
        assert float(params[lanes.START_Y]) == 0.0
        assert float(params[lanes.END_X]) == 100.0
        assert float(params[lanes.END_Y]) == 0.0
        assert float(params[lanes.WIDTH]) == 4.0
        assert float(params[lanes.LENGTH]) == pytest.approx(100.0)
        assert float(params[lanes.HEADING]) == pytest.approx(0.0)

    def test_vertical_lane(self):
        '''Create a lane along the y-axis.'''
        params = lanes.make_lane_params(0.0, 0.0, 0.0, 100.0, width=4.0)
        assert float(params[lanes.LENGTH]) == pytest.approx(100.0)
        assert float(params[lanes.HEADING]) == pytest.approx(jnp.pi / 2)

    def test_diagonal_lane(self):
        '''Create a diagonal lane.'''
        params = lanes.make_lane_params(0.0, 0.0, 100.0, 100.0, width=4.0)
        assert float(params[lanes.LENGTH]) == pytest.approx(100.0 * jnp.sqrt(2))
        assert float(params[lanes.HEADING]) == pytest.approx(jnp.pi / 4)

    def test_direction_vectors(self):
        '''Direction vectors should be unit vectors.'''
        params = lanes.make_lane_params(0.0, 0.0, 100.0, 50.0, width=4.0)
        dir_x, dir_y = params[lanes.DIR_X], params[lanes.DIR_Y]
        dir_lat_x, dir_lat_y = params[lanes.DIR_LAT_X], params[lanes.DIR_LAT_Y]
        # Check unit length
        assert float(dir_x ** 2 + dir_y ** 2) == pytest.approx(1.0)
        assert float(dir_lat_x ** 2 + dir_lat_y ** 2) == pytest.approx(1.0)
        # Check orthogonality
        assert float(dir_x * dir_lat_x + dir_y * dir_lat_y) == pytest.approx(0.0)


class TestLaneLocalCoordinates:
    def setup_method(self):
        # Horizontal lane from (0,5) to (100,5) with width 4
        self.lane = lanes.make_lane_params(0.0, 5.0, 100.0, 5.0, width=4.0)

    def test_origin(self):
        '''Point at lane start should have (0, 0) local coordinates.'''
        long, lat = lanes.lane_local_coordinates(self.lane, 0.0, 5.0)
        assert float(long) == pytest.approx(0.0)
        assert float(lat) == pytest.approx(0.0)

    def test_longitudinal(self):
        '''Point along lane centerline.'''
        long, lat = lanes.lane_local_coordinates(self.lane, 50.0, 5.0)
        assert float(long) == pytest.approx(50.0)
        assert float(lat) == pytest.approx(0.0)

    def test_lateral_left(self):
        '''Point to the left of centerline (positive lateral).'''
        long, lat = lanes.lane_local_coordinates(self.lane, 50.0, 7.0)
        assert float(long) == pytest.approx(50.0)
        assert float(lat) == pytest.approx(2.0)

    def test_lateral_right(self):
        '''Point to the right of centerline (negative lateral).'''
        long, lat = lanes.lane_local_coordinates(self.lane, 50.0, 3.0)
        assert float(long) == pytest.approx(50.0)
        assert float(lat) == pytest.approx(-2.0)


class TestLanePosition:
    def setup_method(self):
        self.lane = lanes.make_lane_params(0.0, 5.0, 100.0, 5.0, width=4.0)

    def test_origin(self):
        '''Position at (0, 0) local coords.'''
        pos = lanes.lane_position(self.lane, 0.0, 0.0)
        assert float(pos[0]) == pytest.approx(0.0)
        assert float(pos[1]) == pytest.approx(5.0)

    def test_along_centerline(self):
        '''Position along centerline.'''
        pos = lanes.lane_position(self.lane, 50.0, 0.0)
        assert float(pos[0]) == pytest.approx(50.0)
        assert float(pos[1]) == pytest.approx(5.0)

    def test_with_lateral_offset(self):
        '''Position with lateral offset.'''
        pos = lanes.lane_position(self.lane, 50.0, 2.0)
        assert float(pos[0]) == pytest.approx(50.0)
        assert float(pos[1]) == pytest.approx(7.0)

    def test_roundtrip(self):
        '''Converting world->local->world should give original position.'''
        world_x, world_y = 75.0, 8.0
        long, lat = lanes.lane_local_coordinates(self.lane, world_x, world_y)
        pos = lanes.lane_position(self.lane, float(long), float(lat))
        assert float(pos[0]) == pytest.approx(world_x)
        assert float(pos[1]) == pytest.approx(world_y)


class TestLaneOnLane:
    def setup_method(self):
        self.lane = lanes.make_lane_params(0.0, 0.0, 100.0, 0.0, width=4.0)

    def test_on_centerline(self):
        '''Point on centerline should be on lane.'''
        assert lanes.lane_on_lane(self.lane, 50.0, 0.0)

    def test_within_width(self):
        '''Point within lane width should be on lane.'''
        assert lanes.lane_on_lane(self.lane, 50.0, 1.5)
        assert lanes.lane_on_lane(self.lane, 50.0, -1.5)

    def test_outside_width(self):
        '''Point outside lane width should not be on lane.'''
        assert not lanes.lane_on_lane(self.lane, 50.0, 3.0)
        assert not lanes.lane_on_lane(self.lane, 50.0, -3.0)

    def test_with_margin(self):
        '''Point outside width but within margin should be on lane.'''
        assert lanes.lane_on_lane(self.lane, 50.0, 3.0, margin=1.0)


class TestMakeHighwayLanes:
    def test_creates_correct_number_of_lanes(self):
        hw = lanes.make_highway_lanes(4, 100.0, 4.0)
        assert hw.shape == (4, lanes.LANE_PARAMS_SIZE)

    def test_lane_positions(self):
        hw = lanes.make_highway_lanes(4, 100.0, 4.0)
        # Lane 0 at y=0, lane 1 at y=4, lane 2 at y=8, lane 3 at y=12
        for i in range(4):
            assert float(hw[i, lanes.START_Y]) == pytest.approx(i * 4.0)
            assert float(hw[i, lanes.END_Y]) == pytest.approx(i * 4.0)

    def test_lane_lengths(self):
        hw = lanes.make_highway_lanes(4, 100.0, 4.0)
        for i in range(4):
            assert float(hw[i, lanes.LENGTH]) == pytest.approx(100.0)

    def test_all_lanes_horizontal(self):
        hw = lanes.make_highway_lanes(4, 100.0, 4.0)
        for i in range(4):
            assert float(hw[i, lanes.HEADING]) == pytest.approx(0.0)


class TestSideLanes:
    def test_middle_lane(self):
        left, right = lanes.side_lanes(2, 5)
        assert int(left) == 1
        assert int(right) == 3

    def test_first_lane(self):
        left, right = lanes.side_lanes(0, 5)
        assert int(left) == 0  # Clipped
        assert int(right) == 1

    def test_last_lane(self):
        left, right = lanes.side_lanes(4, 5)
        assert int(left) == 3
        assert int(right) == 4  # Clipped
