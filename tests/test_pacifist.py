"""Behavioral verification tests for the Pure Pacifist strategy (experiment/pure-pacifist-v1)."""
from __future__ import annotations

import gzip
import json
import math
import os
import unittest

from bot.controller import MechBot
from test_bot import ARENA_DIR, make_hello, make_obs


class TestPurePacifist(unittest.TestCase):
    def setUp(self):
        self.bot = MechBot(make_hello(ARENA_DIR))

    def test_profile_is_pure_pacifist(self):
        self.assertEqual(self.bot.profile.name, "pure_pacifist")
        self.assertTrue(self.bot.profile.pacifist)
        self.assertEqual(self.bot.name, "pacifist-v1")

    def test_never_fires_even_when_attacked(self):
        enemy = {
            "id": 5, "kind": "mech", "pos": [20.0, 0.0, 0.0], "vel": [0, 0, 0],
            "aim": [math.pi, 0.0], "hp": 150.0, "max_hp": 1400.0, "cubes": 3, "weapons": ["gun"],
        }
        obs = make_obs(
            pos=[0.0, 0.0, 0.0],
            aim=[0.0, 0.0],
            mechs=[enemy],
            events=[{"type": "damaged", "amount": 90, "weapon": "gun", "bearing": 0.0, "by": 5, "rear": False}],
        )
        act = self.bot.act(obs)
        self.assertEqual(act["fire"], [False, False])
        self.assertEqual(self.bot.strategy.mode, "RETREAT_HEAL")

    def test_silent_walking_when_safe(self):
        obs = make_obs(pos=[0.0, 0.0, 0.0], in_zone=True)
        act = self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "SAFE_ROTATE")
        speed_frac = math.hypot(act["move"][0], act["move"][1])
        # walk_speed is 11.7; silent speed is <= 5.0 m/s -> move magnitude <= 5.0/11.7 ~= 0.43
        self.assertLessEqual(speed_frac * 11.7, 5.0)
        self.assertFalse(act["boost"])

    def test_zero_shots_across_entire_300_obs_recorded_stream(self):
        stream_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "check", "stream.jsonl.gz")
        with gzip.open(stream_path, "rt", encoding="utf-8") as f:
            hello = json.loads(f.readline())
            hello["map"]["dir"] = ARENA_DIR
            bot = MechBot(hello)
            obs_count = 0
            for line in f:
                msg = json.loads(line)
                if msg.get("type") == "obs":
                    obs_count += 1
                    act = bot.act(msg)
                    self.assertEqual(act["fire"], [False, False], f"Pacifist fired on tick {msg['tick']}!")
                    self.assertNotIn("take", act)
            self.assertEqual(obs_count, 300)


if __name__ == "__main__":
    unittest.main()
