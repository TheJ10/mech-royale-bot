"""Targeted regression tests for all 6 audit findings (BH-001 through BH-006) in Tactical Survivor v3.1."""
from __future__ import annotations

import math
import unittest

from bot.controller import MechBot
from bot.combat import angle_diff
from bot.nav import Navigator
from bot.world import Prism, WorldModel
from test_bot import ARENA_DIR, make_hello, make_obs


def _make_walled_bot() -> MechBot:
    """Create a MechBot on a synthetic flat arena with a 20x20 building at x=[30..50], y=[-10..10]."""
    w = WorldModel()
    wall = Prism("building", -2.0, 25.0, [30, -10, 50, -10, 50, 10, 30, 10], [])
    w.load_synthetic((-120, 120, -120, 120), [wall])
    bot = MechBot(make_hello())
    bot.world = w
    bot.nav = Navigator(w)
    bot.combat.world = w
    bot.threats.world = w
    bot.resources.world = w
    bot.strategy.world = w
    bot.strategy.nav = bot.nav
    return bot


class TestV31Regressions(unittest.TestCase):
    def test_bh001_retreat_cover_holds_still_without_ping_pong(self):
        bot = _make_walled_bot()
        # Bot is at (20, 0) behind the wall [30..50] from enemy at (75, 0)
        enemy = {
            "id": 2, "kind": "mech", "pos": [75.0, 0.0, 0.0], "vel": [0, 0, 0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 1, "weapons": ["gun"],
        }
        for i in range(6):
            obs = make_obs(tick=100 + i * 2, t=10.0 + i * 0.1, pos=[20.0, 0.0, 0.0], hp=500.0, mechs=[enemy])
            act = bot.act(obs)
            self.assertEqual(bot.strategy.mode, "RETREAT_HEAL")
            self.assertEqual(act["move"], [0.0, 0.0], "Should hold still behind cover instead of ping-ponging")
            self.assertFalse(act["boost"], "Should not boost while already behind cover")

    def test_bh002_leaving_silent_hold_no_false_stuck_jump_or_jet(self):
        bot = MechBot(make_hello(ARENA_DIR))
        # Prime nav.pos_history with 6 stationary entries, then enter ROOF_HOLD
        for i in range(6):
            bot.nav.pos_history.append((100 + i * 2, 0.0, 0.0, 18.0))
        obs_hold = make_obs(tick=120, t=12.0, pos=[0.0, 0.0, 18.0], grounded=True, in_zone=True)
        act_hold = bot.act(obs_hold)
        self.assertEqual(bot.strategy.mode, "ROOF_HOLD")
        self.assertEqual(act_hold["move"], [0.0, 0.0])
        self.assertEqual(len(bot.nav.pos_history), 0)

        # Now leave ROOF_HOLD (zone shifts far away) -> must NOT fire false-stuck jump/jet
        obs_exit = make_obs(
            tick=122, t=12.1, pos=[0.0, 0.0, 18.0], grounded=True, in_zone=True,
            zone={"center": [150.0, 0.0], "radius": 300.0, "next_center": [150.0, 0.0], "next_radius": 80.0, "shrink_start": 60.0, "shrink_end": 110.0, "dps": 20.0},
        )
        act_exit = bot.act(obs_exit)
        self.assertFalse(act_exit["jump"])
        self.assertLess(bot.nav.stuck_until_tick, 0)

    def test_bh003a_long_range_charging_laser_recognized(self):
        bot = MechBot(make_hello())
        sniper = {
            "id": 3, "kind": "mech", "pos": [150.0, 0.0, 0.0], "vel": [0, 0, 0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0,
            "weapons": ["gun", "laser"], "charging": True,
        }
        obs = make_obs(pos=[0.0, 0.0, 0.0], mechs=[sniper])
        bot.threats.update(obs)
        target = bot.combat.select_mech_target(obs, bot.threats, selective=True)
        self.assertIsNotNone(target)
        self.assertEqual(target.id, 3)

    def test_bh003b_defensive_facing_when_holding_fire(self):
        bot = MechBot(make_hello())
        # Non-threatening enemy at (0, 95) facing away -> bot holds fire, but must face +pi/2 to protect rear armor!
        enemy = {
            "id": 4, "kind": "mech", "pos": [0.0, 95.0, 0.0], "vel": [0, 0, 0],
            "aim": [0.0, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        obs = make_obs(pos=[0.0, 0.0, 0.0], aim=[-math.pi / 2, 0.0], hp=950.0, mechs=[enemy])
        act = bot.act(obs)
        self.assertEqual(act["fire"], [False, False])
        self.assertLess(abs(angle_diff(act["aim"][0], math.pi / 2)), 0.15)

    def test_bh003c_refuses_noisy_crate_shooting_with_nearby_enemy(self):
        bot = MechBot(make_hello())
        # Non-attacking enemy 65m away + crate 20m away -> must NOT shoot crate!
        enemy = {
            "id": 5, "kind": "mech", "pos": [65.0, 0.0, 0.0], "vel": [0, 0, 0],
            "aim": [math.pi / 2, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        obs = make_obs(
            pos=[0.0, 0.0, 0.0],
            mechs=[enemy],
            crates=[{"id": 1001, "pos": [-20.0, 0.0, 0.0], "hp": 300.0}],
        )
        act = bot.act(obs)
        self.assertNotEqual(bot.strategy.mode, "LOOT_CRATE")
        self.assertEqual(act["fire"], [False, False])

    def test_bh004_stale_path_cleared_when_goal_within_5m(self):
        w = WorldModel()
        w.load_synthetic((-100, 100, -100, 100), [])
        nav = Navigator(w)
        # Establish distant path going east (+80, 0)
        nav.update_and_steer(10, (0.0, 0.0, 0.0), (0, 0, 0), True, 100.0, (80.0, 0.0))
        # Now give a goal 3.5m west (-3.5, 0) -> must immediately steer west, NOT east!
        move_xy, _, _, wp = nav.update_and_steer(12, (0.0, 0.0, 0.0), (0, 0, 0), True, 100.0, (-3.5, 0.0))
        self.assertLess(move_xy[0], -0.5)
        self.assertAlmostEqual(wp[0], -3.5, places=1)

    def test_bh005_reload_in_cover_preserves_laser_charge_and_clamps_fallback(self):
        bot = _make_walled_bot()
        gun_reloading = {
            "kind": "gun", "cooldown": 0.0, "ammo": 0, "reload": 1.8,
            "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4},
        }
        laser_charging = {
            "kind": "laser", "cooldown": 0.0, "charge": 0.50,
            "stats": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "range": 280},
        }
        enemy = {
            "id": 6, "kind": "mech", "pos": [0.0, 60.0, 0.0], "vel": [0, 0, 0],
            "aim": [-math.pi / 2, 0.0], "hp": 1000.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        obs = make_obs(pos=[0.0, 0.0, 0.0], aim=[math.pi / 2, 0.0], weapons=[gun_reloading, laser_charging], mechs=[enemy])
        act = bot.act(obs)
        self.assertEqual(bot.strategy.mode, "COMBAT_KITE")
        # Laser must continue charging (fire[1] = True), NOT get cancelled by ducking behind cover
        self.assertTrue(act["fire"][1])

    def test_bh006_final_zone_tiny_radius_clamped_inside_boundary(self):
        w = WorldModel()
        wall = Prism("building", -2.0, 25.0, [30, -10, 50, -10, 50, 10, 30, 10], [])
        w.load_synthetic((-100, 100, -100, 100), [wall])
        # Tiny active zone radius = 6.0m -> find_cover_point must NEVER return a point outside 6.0m
        pt = w.find_cover_point((0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (0.0, 0.0), 6.0)
        if pt is not None:
            self.assertLessEqual(math.hypot(pt[0], pt[1]), 6.0)

        # In endgame with active_r = 6.0m and bot at edge (4.5, 0), zone_urgent must trigger to pull inward
        bot = MechBot(make_hello())
        enemy = {"id": 7, "kind": "mech", "pos": [10.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0.0], "hp": 900.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs = make_obs(
            t=424.0, pos=[4.5, 0.0, 0.0], in_zone=True, mechs=[enemy],
            zone={"center": [0.0, 0.0], "radius": 25.0, "next_center": [0.0, 0.0], "next_radius": 0.0, "shrink_start": 405.0, "shrink_end": 430.0, "dps": 260.0},
        )
        act = bot.act(obs)
        self.assertEqual(bot.strategy.mode, "ZONE_URGENT")
        self.assertLess(act["move"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
