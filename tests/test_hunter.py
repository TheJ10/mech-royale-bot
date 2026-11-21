"""Behavioral verification tests for Opportunistic Hunter (experiment/opportunistic-hunter-v1)."""
from __future__ import annotations

import math
import unittest

from bot.controller import MechBot
from test_bot import ARENA_DIR, make_hello, make_obs


class TestOpportunisticHunter(unittest.TestCase):
    def setUp(self):
        # Open synthetic world for pure combat-decision & target-prioritization checks
        self.bot = MechBot(make_hello())

    def test_profile_is_opportunistic_hunter(self):
        self.assertEqual(self.bot.profile.name, "opportunistic_hunter")
        self.assertFalse(self.bot.profile.pacifist)
        self.assertEqual(self.bot.name, "hunter-v2")

    def test_engages_vulnerable_wounded_enemy_and_fires(self):
        wounded = {
            "id": 7, "kind": "mech", "pos": [60.0, 0.0, 0.0], "vel": [0, 0, 0],
            "aim": [0.0, 0.0], "hp": 320.0, "max_hp": 1400.0, "cubes": 4, "weapons": ["gun"],
        }
        obs = make_obs(pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0], hp=1200.0, mechs=[wounded])
        act = self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "COMBAT_KITE")
        self.assertTrue(act["fire"][0])

    def test_prioritizes_exposed_rear_armor_and_wounded_over_healthy_front(self):
        healthy_front = {
            "id": 8, "kind": "mech", "pos": [55.0, 20.0, 0.0], "vel": [0, 0, 0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        wounded_rear = {
            "id": 9, "kind": "mech", "pos": [65.0, -10.0, 0.0], "vel": [0, 0, 0],
            "aim": [0.0, 0.0], "hp": 300.0, "max_hp": 1400.0, "cubes": 3, "weapons": ["gun"],
        }
        obs = make_obs(pos=[0.0, 0.0, 0.0], mechs=[healthy_front, wounded_rear])
        self.bot.threats.update(obs)
        chosen = self.bot.combat.select_mech_target(obs, self.bot.threats)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.id, 9)

    def test_retreats_from_poor_expected_value_duel(self):
        boss = {
            "id": 10, "kind": "mech", "pos": [45.0, 0.0, 0.0], "vel": [0, 0, 0],
            "aim": [math.pi, 0.0], "hp": 1600.0, "max_hp": 1600.0, "cubes": 6, "weapons": ["gun", "lightning"],
        }
        obs = make_obs(pos=[0.0, 0.0, 0.0], hp=420.0, mechs=[boss])
        self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "RETREAT_HEAL")


if __name__ == "__main__":
    unittest.main()
