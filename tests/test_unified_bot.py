"""Unit and integration tests for Unified Battle Intelligence (UBI-v1) decision engine and telemetry."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

from bot.controller import MechBot
from bot.nav import Navigator
from bot.world import Prism, WorldModel
from test_bot import ARENA_DIR, make_hello, make_obs
from tools.analyze_telemetry import analyze_trace


class TestUnifiedBattleIntelligence(unittest.TestCase):
    def setUp(self):
        self.bot = MechBot(make_hello(ARENA_DIR))

    def test_01_bot_name_and_unified_profile(self):
        self.assertEqual(self.bot.name, "ubi-v1")
        self.assertEqual(self.bot.profile.name, "unified")
        self.assertFalse(self.bot.profile.pacifist)

    def test_02_phase_classification_across_match_lifecycle(self):
        s = self.bot.strategy
        self.assertEqual(s.classify_phase(now=25.0, alive=15, next_radius=380.0), "EARLY_GAME")
        self.assertEqual(s.classify_phase(now=130.0, alive=9, next_radius=140.0), "MID_GAME")
        self.assertEqual(s.classify_phase(now=260.0, alive=4, next_radius=70.0), "LATE_GAME")
        self.assertEqual(s.classify_phase(now=360.0, alive=2, next_radius=25.0), "FINAL_CIRCLE")

    def test_03_central_decision_metadata_and_hysteresis(self):
        obs = make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], hp=1400.0)
        self.bot.act(obs)
        meta = self.bot.strategy.last_decision_meta
        self.assertIn("phase", meta)
        self.assertIn("selected_objective", meta)
        self.assertIn("scores", meta)
        self.assertIn("reason_codes", meta)
        self.assertGreater(len(meta["candidates"]), 0)

    def test_04_moving_retreat_never_freezes_stationary(self):
        enemy = {
            "id": 3, "kind": "mech", "pos": [45.0, 0.0, 0.0], "vel": [-5.0, 0.0, 0.0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 2, "weapons": ["gun"],
        }
        obs = make_obs(tick=200, t=20.0, pos=[0.0, 0.0, 0.0], hp=420.0, mechs=[enemy])
        act = self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "RETREAT_HEAL")
        self.assertGreater(math.hypot(act["move"][0], act["move"][1]), 0.3)

    def test_05_dual_purpose_npc_healthy_farm_vs_low_hp_danger(self):
        bot = MechBot(make_hello())
        npc = {"id": 105, "kind": "npc", "pos": [28.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [0, 0], "hp": 350, "max_hp": 350}
        # Healthy -> farms NPC for 2 cubes
        obs_healthy = make_obs(pos=[0.0, 0.0, 0.0], aim=[0.0, -0.18], hp=1350.0, mechs=[npc])
        act_h = bot.act(obs_healthy)
        self.assertEqual(bot.strategy.mode, "LOOT_NPC")
        self.assertTrue(act_h["fire"][0])

        # Low HP -> NPC included in danger_at for A* avoidance
        self.assertGreater(bot.threats.danger_at(28.0, 0.0, 10.0, include_npcs=True), 1.0)

    def test_06_telemetry_consequence_tracking_and_offline_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "decision_trace.jsonl")
            bot = MechBot(make_hello(), trace_path=trace_file)
            obs1 = make_obs(tick=10, t=1.0, pos=[0.0, 0.0, 0.0], hp=1400.0)
            bot.act(obs1)
            obs2 = make_obs(
                tick=12, t=1.1, pos=[1.1, 0.0, 0.0], hp=1355.0,
                events=[
                    {"type": "damaged", "amount": 45.0, "weapon": "gun", "bearing": 0.0, "by": 2},
                    {"type": "hit", "target": 2, "amount": 45.0, "weapon": "gun"},
                ],
            )
            bot.act(obs2)
            bot.telemetry.finalize()

            summary_path = os.path.join(tmpdir, "match_summary.json")
            perf_path = os.path.join(tmpdir, "strategy_performance.json")
            report_path = os.path.join(tmpdir, "match_report.md")
            fail_path = os.path.join(tmpdir, "failure_patterns.md")
            summary = analyze_trace(trace_file, summary_path, perf_path, report_path, fail_path)
            self.assertEqual(summary["total_ticks"], 2)
            self.assertAlmostEqual(summary["total_damage_dealt"], 45.0)
            self.assertAlmostEqual(summary["total_damage_received"], 45.0)
            self.assertTrue(os.path.isfile(report_path))
            self.assertTrue(os.path.isfile(fail_path))


if __name__ == "__main__":
    unittest.main()
