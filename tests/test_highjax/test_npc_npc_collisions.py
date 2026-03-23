'''Tests for NPC-NPC collision impulses.

When npc_npc_collisions=True (default, matching highway-env), overlapping
NPCs get pushed apart via impulse separation. When False, NPCs overlap freely.
'''
from __future__ import annotations

import pytest
import jax.numpy as jnp

from highjax import kinematics

DT = 0.1


class TestNpcNpcCollisions:
    '''NPC-NPC collision detection and impulse.'''

    def test_overlapping_npcs_get_impulse(self):
        '''Two overlapping NPCs should receive nonzero, equal-and-opposite impulse.'''
        # Vehicles 4m apart (length=5m) -> 1m overlap in x, less than 2m width
        # -> SAT pushes along x
        npc_a = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 20.0, 1)
        npc_b = kinematics.make_vehicle_state(54.0, 4.0, 0.0, 20.0, 1)
        npc_states = jnp.stack([npc_a, npc_b])

        impulse_by_npc, colliding_by_npc = kinematics.check_npc_npc_collisions_all(
            npc_states, dt=DT)

        # Nonzero impulse
        assert float(jnp.linalg.norm(impulse_by_npc[0])) > 0
        # Equal and opposite
        assert jnp.allclose(impulse_by_npc[0], -impulse_by_npc[1], atol=1e-5)
        # Both flagged as colliding
        assert bool(colliding_by_npc[0])
        assert bool(colliding_by_npc[1])

    def test_separated_npcs_no_impulse(self):
        '''Well-separated NPCs should receive zero impulse.'''
        npc_a = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 20.0, 1)
        npc_b = kinematics.make_vehicle_state(70.0, 4.0, 0.0, 20.0, 1)
        npc_states = jnp.stack([npc_a, npc_b])

        impulse_by_npc, colliding_by_npc = kinematics.check_npc_npc_collisions_all(
            npc_states, dt=DT)

        assert jnp.allclose(impulse_by_npc, 0.0, atol=1e-6)
        assert not bool(colliding_by_npc[0])
        assert not bool(colliding_by_npc[1])

    def test_three_npcs_pairwise(self):
        '''Three NPCs: A overlaps B, B overlaps C, A doesn't overlap C.'''
        # 4m apart (1m overlap), so A-B and B-C overlap but A-C don't (8m apart)
        npc_a = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 20.0, 1)
        npc_b = kinematics.make_vehicle_state(54.0, 4.0, 0.0, 20.0, 1)
        npc_c = kinematics.make_vehicle_state(58.0, 4.0, 0.0, 20.0, 1)
        npc_states = jnp.stack([npc_a, npc_b, npc_c])

        impulse_by_npc, colliding_by_npc = kinematics.check_npc_npc_collisions_all(
            npc_states, dt=DT)

        # A gets pushed left (away from B), C gets pushed right (away from B)
        assert float(jnp.linalg.norm(impulse_by_npc[0])) > 0, (
            'NPC A should have impulse from B')
        assert float(jnp.linalg.norm(impulse_by_npc[2])) > 0, (
            'NPC C should have impulse from B')
        # B gets symmetric impulses from A and C that cancel
        assert float(jnp.linalg.norm(impulse_by_npc[1])) < 1e-5, (
            'Middle NPC impulses should cancel')
        # All three are involved in collisions
        assert bool(colliding_by_npc[0])
        assert bool(colliding_by_npc[1])
        assert bool(colliding_by_npc[2])

    def test_ego_npc_collision_unchanged(self):
        '''Ego-NPC collision should work the same regardless of npc_npc_collisions.'''
        ego = kinematics.make_vehicle_state(50.0, 4.0, 0.0, 25.0, 1)
        npc = kinematics.make_vehicle_state(52.0, 4.0, 0.0, 20.0, 1)
        npc_states = npc[None, :]

        per_npc_colliding, _, per_npc_translations = (
            kinematics.check_vehicle_collisions_all(ego, npc_states, dt=DT))

        # Ego-NPC collision is independent of NPC-NPC system
        assert bool(per_npc_colliding[0])
        assert float(jnp.linalg.norm(per_npc_translations[0])) > 0
