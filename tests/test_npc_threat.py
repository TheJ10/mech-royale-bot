"""Targeted tests for NPC threat tracking, low-HP avoidance, LoS, and healthy NPC farming."""
from __future__ import annotations

import math
import unittest

from bot.controller import MechBot
from bot.world import Prism, WorldModel
from test_bot import make_hello, make_obs


class TestNPCThreat(unittest.TestCase):
    def test_01_healthy_bot_still_farms_isolated_npc_for_2_cubes(self):
        bot = MechBot(make_hello())
        npc = {"id": 101, "kind": "npc", "pos": [30.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [0, 0], "hp": 350, "max_hp": 350}
        obs = make_obs(pos=[0.0, 0.0, 0.0], aim=[0.0, -0.17], hp=1300.0, mechs=[npc])
        act = bot.act(obs)
        self.assertEqual(bot.strategy.mode, "LOOT_NPC")
        self.assertTrue(act["fire"][0])

    def test_02_low_hp_bot_avoids_and_retreats_from_los_npc(self):
        bot = MechBot(make_hello())
        npc = {"id": 102, "kind": "npc", "pos": [30.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 350, "max_hp": 350}
        obs = make_obs(pos=[0.0, 0.0, 0.0], hp=450.0, mechs=[npc])
        act = bot.act(obs)
        self.assertEqual(bot.strategy.mode, "RETREAT_HEAL")
        # Moves away from NPC at +30 X
        self.assertLess(act["move"][0], 0.0)
        # NPC included in danger_at when avoiding NPCs
        self.assertGreater(bot.threats.danger_at(30.0, 0.0, 10.0, include_npcs=True), 1.0)

    def test_03_npc_behind_wall_does_not_trigger_false_retreat(self):
        w = WorldModel()
        wall = Prism("building", -2.0, 25.0, [12, -15, 22, -15, 22, 15, 12, 15], [])
        w.load_synthetic((-100, 100, -100, 100), [wall])
        bot = MechBot(make_hello())
        bot.world = w
        bot.strategy.world = w
        npc = {"id": 103, "kind": "npc", "pos": [32.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 350, "max_hp": 350}
        obs = make_obs(pos=[0.0, 0.0, 0.0], hp=450.0, mechs=[npc])
        bot.act(obs)
        self.assertNotEqual(bot.strategy.mode, "RETREAT_HEAL")

    def test_04_incoming_npc_bullet_triggers_perpendicular_dodge(self):
        bot = MechBot(make_hello())
        obs = make_obs(
            pos=[0.0, 0.0, 0.0],
            projectiles=[{"id": 301, "kind": "npc", "pos": [25.0, 0.0, 4.0], "vel": [-90.0, 0.0, 0.0]}],
        )
        dodge = bot.threats.compute_dodge_vector(obs, (0.0, 0.0))
        self.assertIsNotNone(dodge)
        self.assertGreater(abs(dodge[1]), 0.8)


if __name__ == "__main__":
    unittest.main()
