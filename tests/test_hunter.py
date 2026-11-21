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
        self.assertEqual(self.bot.name, "hunter-v2.1")

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

    def test_bh004_stale_path_cleared_within_5m(self):
        from bot.nav import Navigator
        from bot.world import WorldModel
        w = WorldModel()
        w.load_synthetic((-100, 100, -100, 100), [])
        nav = Navigator(w)
        nav.update_and_steer(10, (0.0, 0.0, 0.0), (0, 0, 0), True, 100.0, (80.0, 0.0))
        move_xy, _, _, wp = nav.update_and_steer(12, (0.0, 0.0, 0.0), (0, 0, 0), True, 100.0, (-3.5, 0.0))
        self.assertLess(move_xy[0], -0.5)
        self.assertAlmostEqual(wp[0], -3.5, places=1)

    def test_bh005_reload_in_cover_preserves_laser_charge(self):
        from bot.nav import Navigator
        from bot.world import Prism, WorldModel
        w = WorldModel()
        wall = Prism("building", -2.0, 25.0, [30, -10, 50, -10, 50, 10, 30, 10], [])
        w.load_synthetic((-120, 120, -120, 120), [wall])
        self.bot.world = w
        self.bot.nav = Navigator(w)
        self.bot.combat.world = w
        self.bot.strategy.world = w
        self.bot.strategy.nav = self.bot.nav
        gun_reloading = {
            "kind": "gun", "cooldown": 0.0, "ammo": 0, "reload": 1.8,
            "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4},
        }
        laser_charging = {
            "kind": "laser", "cooldown": 0.0, "charge": 0.50,
            "stats": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "range": 280},
        }
        enemy = {"id": 6, "kind": "mech", "pos": [0.0, 60.0, 0.0], "vel": [0, 0, 0], "aim": [-math.pi / 2, 0.0], "hp": 1000.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs = make_obs(pos=[0.0, 0.0, 0.0], aim=[math.pi / 2, 0.0], weapons=[gun_reloading, laser_charging], mechs=[enemy])
        act = self.bot.act(obs)
        self.assertTrue(act["fire"][1])

    def test_bh006_final_zone_tiny_radius_clamped(self):
        enemy = {"id": 7, "kind": "mech", "pos": [10.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0.0], "hp": 900.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs = make_obs(
            t=424.0, pos=[4.5, 0.0, 0.0], in_zone=True, mechs=[enemy],
            zone={"center": [0.0, 0.0], "radius": 25.0, "next_center": [0.0, 0.0], "next_radius": 0.0, "shrink_start": 405.0, "shrink_end": 430.0, "dps": 260.0},
        )
        act = self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "ZONE_URGENT")
        self.assertLess(act["move"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
