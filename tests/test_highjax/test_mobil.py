'''Tests for MOBIL overhaul (divergences #4, #11, #12).

#4:  Lane change cooldown timer — NPC shouldn't evaluate MOBIL within 1.0s
#11: Follower target speed — mobil_gain uses rear vehicle's actual target_speed
#12: Abort-if-conflict — lane change aborted when another NPC targets same gap
'''
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from highjax import kinematics, idm, lanes


LANE_WIDTH = 4.0
N_LANES = 4
HIGHWAY_LANES = lanes.make_highway_lanes(N_LANES, 1000.0, LANE_WIDTH)
DT = 0.1


def _make(x, y, speed, target_lane, target_speed=25.0, timer=0.0):
    '''Helper to create a vehicle state with lane_change_timer.'''
    state = kinematics.make_vehicle_state(x, y, 0.0, speed, target_lane)
    state = state.at[kinematics.TARGET_SPEED].set(target_speed)
    state = state.at[kinematics.LANE_CHANGE_TIMER].set(timer)
    return state


class TestLaneChangeCooldown:
    '''#4: NPC shouldn't evaluate MOBIL within LANE_CHANGE_DELAY of last change.'''

    def test_fresh_timer_blocks_lane_change(self):
        '''Timer=0 (just changed) → MOBIL should be blocked.'''
        # NPC in lane 1, slow car ahead → would normally want to change
        npc = _make(50.0, 4.0, 25.0, 1, timer=0.0)
        slow_front = _make(55.0, 4.0, 10.0, 1, timer=5.0)
        all_states = jnp.stack([slow_front])

        new_npc = idm.npc_sub_step(npc, all_states, HIGHWAY_LANES, DT)
        # Should stay in current lane because cooldown isn't met
        assert int(new_npc[kinematics.TARGET_LANE_IDX]) == 1

    def test_cooldown_met_allows_lane_change(self):
        '''Timer >= LANE_CHANGE_DELAY → MOBIL should be evaluated.'''
        npc = _make(50.0, 4.0, 25.0, 1, timer=kinematics.LANE_CHANGE_DELAY)
        slow_front = _make(55.0, 4.0, 10.0, 1, timer=5.0)
        all_states = jnp.stack([slow_front])

        new_npc = idm.npc_sub_step(npc, all_states, HIGHWAY_LANES, DT)
        new_lane = int(new_npc[kinematics.TARGET_LANE_IDX])
        # Should be allowed to change lane
        assert new_lane != 1, 'MOBIL should suggest lane change when cooldown met'

    def test_timer_resets_on_cooldown_evaluation(self):
        '''Timer resets to 0 whenever cooldown fires, even without lane change.'''
        # NPC in lane 0 (leftmost) with no reason to change lanes, but cooldown is met
        npc = _make(50.0, 0.0, 25.0, 0, timer=kinematics.LANE_CHANGE_DELAY)
        far_npc = _make(1000.0, 0.0, 25.0, 0, timer=5.0)
        all_states = jnp.stack([far_npc])

        new_npc = idm.npc_sub_step(npc, all_states, HIGHWAY_LANES, DT)
        # NPC stays in lane (no reason to change)
        assert int(new_npc[kinematics.TARGET_LANE_IDX]) == 0
        # Timer still resets because cooldown_ready fired
        assert float(new_npc[kinematics.LANE_CHANGE_TIMER]) == pytest.approx(0.0)

    def test_timer_resets_on_lane_change(self):
        '''When NPC changes lane via cooldown, timer resets to 0.'''
        npc = _make(50.0, 4.0, 25.0, 1, timer=kinematics.LANE_CHANGE_DELAY)
        slow_front = _make(55.0, 4.0, 10.0, 1, timer=5.0)
        all_states = jnp.stack([slow_front])

        new_npc = idm.npc_sub_step(npc, all_states, HIGHWAY_LANES, DT)
        new_lane = int(new_npc[kinematics.TARGET_LANE_IDX])
        if new_lane != 1:
            assert float(new_npc[kinematics.LANE_CHANGE_TIMER]) == pytest.approx(0.0)

    def test_timer_increments_when_staying(self):
        '''When NPC stays in lane, timer increments by dt.'''
        npc = _make(50.0, 4.0, 25.0, 1, timer=0.5)
        # No reason to change lanes (no blocking vehicle)
        far_npc = _make(1000.0, 4.0, 25.0, 1, timer=5.0)
        all_states = jnp.stack([far_npc])

        new_npc = idm.npc_sub_step(npc, all_states, HIGHWAY_LANES, DT)
        assert float(new_npc[kinematics.LANE_CHANGE_TIMER]) == pytest.approx(0.6)


class TestFollowerTargetSpeed:
    '''#11: mobil_gain uses rear vehicle's actual target_speed.'''

    def test_rear_target_speed_returned(self):
        '''find_rear_vehicle should return the rear vehicle's target_speed.'''
        ego = _make(50.0, 4.0, 25.0, 1)
        rear = _make(30.0, 4.0, 20.0, 1, target_speed=30.0)
        all_states = jnp.stack([rear])

        _, _, has_rear, _, rear_tgt_speed, _ = idm.find_rear_vehicle(
            ego, all_states, 1, LANE_WIDTH)
        assert bool(has_rear)
        assert float(rear_tgt_speed) == pytest.approx(30.0)

    def test_rear_target_speed_affects_safety(self):
        '''Follower's target_speed determines safe/unsafe, proving it's used.

        The follower's target_speed affects the IDM free-road term.
        When follower_speed > target_speed, free_acc goes very negative,
        making the lane change unsafe. This verifies #11: mobil_gain uses
        the rear vehicle's actual target_speed, not a hardcoded constant.
        '''
        ego = _make(50.0, 4.0, 25.0, 1)
        front_blocker = _make(60.0, 4.0, 10.0, 1)

        # Rear at 70m back, speed=20. Distance enough that interaction is mild.
        # target_speed=25 → free_acc positive → safe
        # target_speed=15 → speed>target → free_acc very negative → unsafe
        rear_safe = _make(-20.0, 0.0, 20.0, 0, target_speed=25.0)
        rear_unsafe = _make(-20.0, 0.0, 20.0, 0, target_speed=15.0)

        all_safe = jnp.stack([rear_safe, front_blocker])
        all_unsafe = jnp.stack([rear_unsafe, front_blocker])

        gain_safe = float(idm.mobil_gain(ego, all_safe, 1, 0, LANE_WIDTH))
        gain_unsafe = float(idm.mobil_gain(ego, all_unsafe, 1, 0, LANE_WIDTH))

        assert gain_safe != float('-inf'), 'Should be safe with high target_speed'
        assert gain_unsafe == float('-inf'), 'Should be unsafe with low target_speed'


class TestLaneSelectionRightWins:
    '''M1: Right lane wins when both pass MOBIL threshold.'''

    def test_right_wins_when_both_pass(self):
        '''Both left and right have positive MOBIL gain → right wins.'''
        # NPC in lane 1 with slow blocker → both lane 0 and 2 improve
        npc = _make(50.0, 4.0, 25.0, 1, timer=kinematics.LANE_CHANGE_DELAY)
        blocker = _make(60.0, 4.0, 5.0, 1, timer=5.0)

        all_states = jnp.stack([blocker])
        new_lane = idm.mobil_lane_change(npc, all_states, N_LANES, LANE_WIDTH)
        assert int(new_lane) == 2, \
            'Right should win when both lanes pass MOBIL threshold'

    def test_left_wins_when_right_fails(self):
        '''Only left passes MOBIL → left chosen.'''
        # NPC in lane 1 with slow blocker. Lane 2 also blocked.
        npc = _make(50.0, 4.0, 25.0, 1, timer=kinematics.LANE_CHANGE_DELAY)
        blocker = _make(60.0, 4.0, 5.0, 1, timer=5.0)
        right_blocker = _make(55.0, 8.0, 3.0, 2, timer=5.0)

        all_states = jnp.stack([blocker, right_blocker])
        new_lane = idm.mobil_lane_change(npc, all_states, N_LANES, LANE_WIDTH)
        assert int(new_lane) == 0, \
            'Left should win when right is blocked'


class TestAbortIfConflict:
    '''M2: Abort-if-conflict during ongoing lane changes only.'''

    def test_ongoing_change_aborted_by_conflict(self):
        '''Ongoing lane change aborted when conflict vehicle ahead in target.'''
        # NPC physically between lanes (y=2 → lane 0), target=1 (ongoing)
        npc = _make(50.0, 2.0, 25.0, 1, timer=0.5)
        # Conflict NPC: ahead, at y=8 (lane 2), targeting lane 1
        conflict_npc = _make(60.0, 8.0, 25.0, 1, timer=5.0)
        far_npc = _make(500.0, 12.0, 25.0, 3, timer=5.0)

        all_states = jnp.stack([conflict_npc, far_npc])
        _, _, new_lane = idm.npc_decide(
            npc, all_states, HIGHWAY_LANES, True)

        assert int(new_lane) == 0, \
            'Ongoing lane change should be aborted due to conflict ahead'

    def test_no_abort_at_decision_time(self):
        '''Abort should not fire on new MOBIL decisions (not ongoing change).'''
        # NPC at y=0 (lane 0), target=0, cooldown ready
        npc = _make(50.0, 0.0, 25.0, 0, timer=kinematics.LANE_CHANGE_DELAY)
        blocker = _make(55.0, 0.0, 5.0, 0, timer=5.0)
        # Conflict NPC converging to lane 1 from lane 2
        conflict_npc = _make(60.0, 8.0, 25.0, 1, timer=5.0)

        all_states = jnp.stack([blocker, conflict_npc])
        _, _, new_lane = idm.npc_decide(
            npc, all_states, HIGHWAY_LANES, True)

        assert int(new_lane) == 1, \
            'Abort should not fire at decision time (no ongoing change)'

    def test_no_abort_for_conflict_behind(self):
        '''Conflict vehicle behind (negative signed distance) → no abort.'''
        # NPC mid-lane-change to lane 1
        npc = _make(50.0, 2.0, 25.0, 1, timer=0.5)
        # Conflict NPC behind, targeting lane 1 from lane 2
        conflict_behind = _make(40.0, 8.0, 25.0, 1, timer=5.0)
        far_npc = _make(500.0, 12.0, 25.0, 3, timer=5.0)

        all_states = jnp.stack([conflict_behind, far_npc])
        _, _, new_lane = idm.npc_decide(
            npc, all_states, HIGHWAY_LANES, True)

        assert int(new_lane) == 1, \
            'Conflict behind should not trigger abort (signed distance)'

    def test_no_abort_for_vehicle_already_in_target(self):
        '''Vehicle already in target lane should not trigger abort.'''
        # NPC mid-lane-change to lane 1
        npc = _make(50.0, 2.0, 25.0, 1, timer=0.5)
        # Vehicle ahead, already in lane 1 (y=4), targeting lane 1
        in_target = _make(60.0, 4.0, 25.0, 1, timer=5.0)
        far_npc = _make(500.0, 12.0, 25.0, 3, timer=5.0)

        all_states = jnp.stack([in_target, far_npc])
        _, _, new_lane = idm.npc_decide(
            npc, all_states, HIGHWAY_LANES, True)

        assert int(new_lane) == 1, \
            'Vehicle already in target lane should not trigger abort'

class TestMobilBlockedDuringOngoingChange:
    '''S5 residual: MOBIL must not fire while NPC is mid-lane-change.'''

    def test_mobil_blocked_during_ongoing_change(self):
        '''MOBIL does not fire during ongoing lane change even if cooldown expires.

        NPC physically between lanes (y=2 → physical lane 0) but target=1
        (ongoing change). Cooldown timer is met. Slow blocker in lane 1
        would normally trigger MOBIL to switch to lane 2. But because the
        NPC is mid-change, MOBIL should be blocked and the target stays 1.
        '''
        npc = _make(50.0, 2.0, 25.0, 1, timer=kinematics.LANE_CHANGE_DELAY)
        blocker = _make(55.0, 4.0, 5.0, 1, timer=5.0)
        far_npc = _make(500.0, 12.0, 25.0, 3, timer=5.0)

        all_states = jnp.stack([blocker, far_npc])
        _, _, new_lane = idm.npc_decide(
            npc, all_states, HIGHWAY_LANES, True)

        assert int(new_lane) == 1, \
            'MOBIL should be blocked during ongoing lane change'


class TestIdmClippingMovedOut:
    '''S4: idm_acceleration returns unclipped values, clipping happens in
    npc_decide after min-of-two-lanes. MOBIL sees raw accelerations.'''

    def test_idm_acceleration_can_exceed_acc_max(self):
        '''At very close range, raw IDM acceleration should exceed ACC_MAX.'''
        acc = idm.idm_acceleration(
            ego_speed=jnp.array(30.0),
            target_speed=jnp.array(25.0),
            front_distance=jnp.array(0.5),
            front_speed=jnp.array(0.0),
            has_front=jnp.array(True),
        )
        assert float(acc) < -idm.ACC_MAX, (
            f'Raw IDM should exceed -ACC_MAX at close range, got {float(acc):.1f}')

    def test_mobil_gain_uses_unclipped_values(self):
        '''MOBIL gain should use unclipped IDM values (matching highway-env).

        With a very close front vehicle, the raw IDM acceleration is extreme.
        If MOBIL sees clipped values, the gain is smaller. If unclipped, the
        gain is larger because the current-lane penalty is more severe.
        '''
        # NPC in lane 1 with very close blocker (raw IDM will be extreme)
        npc = _make(50.0, 4.0, 25.0, 1, timer=LANE_WIDTH)
        close_blocker = _make(51.0, 4.0, 5.0, 1, timer=5.0)

        all_states = jnp.stack([close_blocker])
        gain = float(idm.mobil_gain(npc, all_states, 1, 0, LANE_WIDTH))

        # With unclipped IDM, the current-lane acceleration is very negative
        # (e.g. -50), so the gain (improvement from lane change) should be
        # much larger than ACC_MAX
        assert gain > idm.ACC_MAX, (
            f'MOBIL gain should exceed ACC_MAX with unclipped IDM, '
            f'got {gain:.1f}')

    def test_state_expansion_size(self):
        '''Vehicle state should be 10 elements (includes lane_change_timer, idm_delta, crashed).'''
        state = kinematics.make_vehicle_state(0.0, 0.0, 0.0, 25.0, 0)
        assert state.shape == (10,)
        assert kinematics.VEHICLE_STATE_SIZE == 10
        assert kinematics.LANE_CHANGE_TIMER == 7
        assert kinematics.IDM_DELTA == 8
        assert kinematics.CRASHED == 9
