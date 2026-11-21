"""Targeted tests for active safe-zone resource patrol (experiment/tactical-v2-resource-patrol)."""
from __future__ import annotations

import math
import unittest

from bot.controller import MechBot
from test_bot import ARENA_DIR, make_hello, make_obs


class TestResourcePatrol(unittest.TestCase):
    def test_01_healthy_bot_advances_patrol_sector_after_3s_at_anchor(self):
        bot = MechBot(make_hello(ARENA_DIR))
        # First call picks an anchor
        obs1 = make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], hp=1400.0)
        bot.act(obs1)
        a1 = bot.strategy._anchor
        self.assertIsNotNone(a1)
        # Arrive at anchor a1 and stay 3.5s with <8 cubes -> should advance to a fresh sector >= 40m away!
        obs2 = make_obs(tick=110, t=11.0, pos=[a1[0], a1[1], 0.0], hp=1400.0)
        bot.act(obs2)
        obs3 = make_obs(tick=180, t=14.5, pos=[a1[0], a1[1], 0.0], hp=1400.0)
        bot.act(obs3)
        a2 = bot.strategy._anchor
        self.assertIsNotNone(a2)
        self.assertGreater(math.hypot(a2[0] - a1[0], a2[1] - a1[1]), 40.0)

    def test_02_low_hp_bot_does_not_unnecessarily_patrol(self):
        bot = MechBot(make_hello(ARENA_DIR))
        obs1 = make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], hp=700.0)
        bot.act(obs1)
        a1 = bot.strategy._anchor
        obs2 = make_obs(tick=110, t=11.0, pos=[a1[0], a1[1], 0.0], hp=700.0)
        bot.act(obs2)
        obs3 = make_obs(tick=180, t=14.5, pos=[a1[0], a1[1], 0.0], hp=700.0)
        bot.act(obs3)
        self.assertEqual(bot.strategy._anchor, a1)


if __name__ == "__main__":
    unittest.main()
