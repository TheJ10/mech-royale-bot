"""Unit and integration tests for Project Overdrive (overdrive-v5) decision engine and telemetry."""
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
        self.assertEqual(self.bot.name, "overdrive-v5")
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

    # --- Priority A: Crate Geometry & Anti-Deadlock Regression Tests ---

    def test_07_crate_valid_standoff_geometry_and_cube_scaling(self):
        rp = self.bot.resources
        d0 = rp.min_crate_standoff(muzzle_z=7.0, target_z=1.44)
        self.assertGreaterEqual(d0, 8.0)
        # More cubes -> taller muzzle_height -> larger required horizontal standoff
        me_big = {"cubes": 10, "stats": {"muzzle_height": 7.0}}
        mh_big = rp.effective_muzzle_height(me_big)
        self.assertGreater(mh_big, 9.0)
        d10 = rp.min_crate_standoff(muzzle_z=mh_big, target_z=1.44)
        self.assertGreater(d10, d0)

    def test_08_crate_invalid_close_range_holds_fire_and_backs_out(self):
        bot = MechBot(make_hello())
        crate = {"id": 1001, "pos": [2.0, 0.0, 0.0], "hp": 300.0}
        obs = make_obs(pos=[0.0, 0.0, 0.0], aim=[0.0, -0.7], crates=[crate])
        act = bot.act(obs)
        self.assertEqual(bot.strategy.mode, "LOOT_CRATE")
        self.assertFalse(act["fire"][0], "Must not fire when inside min crate standoff")
        gx, gy = bot.strategy.last_decision_meta["goal"]
        self.assertGreaterEqual(math.hypot(gx - 2.0, gy - 0.0), 8.5)

    def test_09_crate_successful_acquisition_from_valid_standoff(self):
        bot = MechBot(make_hello())
        crate = {"id": 1002, "pos": [12.0, 0.0, 0.0], "hp": 300.0}
        # Pre-aligned toward z + 1.44 at d=12 -> pitch = atan2(1.44 - 7.0, 12.0) = -0.434
        obs = make_obs(pos=[0.0, 0.0, 0.0], aim=[0.0, -0.434], crates=[crate])
        act = bot.act(obs)
        self.assertEqual(bot.strategy.mode, "LOOT_CRATE")
        self.assertTrue(act["fire"][0])
        self.assertAlmostEqual(act["aim"][1], math.atan2(1.44 - 7.0, 12.0), places=2)

    def test_10_crate_blocked_los_rejected_before_entering_loot_crate(self):
        bot = MechBot(make_hello())
        # Enclose crate in a 6x6 building prism so no standoff angle has LoS
        wall = Prism("building", -20.0, 30.0, [9.0, -3.0, 15.0, -3.0, 15.0, 3.0, 9.0, 3.0], [])
        bot.world.load_synthetic((-100.0, 100.0, -100.0, 100.0), [wall], flat_z=0.0)
        crate = {"id": 1003, "pos": [12.0, 0.0, 0.0], "hp": 300.0}
        obs = make_obs(pos=[0.0, 0.0, 0.0], crates=[crate])
        bot.act(obs)
        self.assertNotEqual(bot.strategy.mode, "LOOT_CRATE")

    def test_11_crate_timeout_switches_to_alternative_objective(self):
        bot = MechBot(make_hello())
        crate = {"id": 1004, "pos": [12.0, 0.0, 0.0], "hp": 300.0}
        cube = {"id": 5001, "pos": [-55.0, 0.0, 0.0]}
        obs0 = make_obs(tick=10, t=1.0, pos=[0.0, 0.0, 0.0], aim=[0.0, -0.434], crates=[crate], cubes=[cube])
        obs0["self"]["cubes"] = 1
        bot.act(obs0)
        self.assertEqual(bot.strategy.mode, "LOOT_CRATE")
        # Advance > 7.5s with crate HP stuck at 300 -> must abandon & switch to LOOT_CUBE
        obs1 = make_obs(tick=90, t=9.0, pos=[0.0, 0.0, 0.0], aim=[0.0, -0.434], crates=[crate], cubes=[cube])
        obs1["self"]["cubes"] = 1
        bot.act(obs1)
        self.assertNotEqual(bot.strategy.mode, "LOOT_CRATE")
        self.assertEqual(bot.strategy.mode, "LOOT_CUBE")

    def test_12_blacklisted_crate_not_reselected_within_cooldown(self):
        bot = MechBot(make_hello())
        crate = {"id": 1005, "pos": [12.0, 0.0, 0.0], "hp": 300.0}
        bot.resources.mark_crate_failed(1005, now=10.0, cooldown_s=35.0)
        obs = make_obs(tick=120, t=12.0, pos=[0.0, 0.0, 0.0], crates=[crate])
        bot.act(obs)
        self.assertEqual(bot.strategy.mode, "SAFE_ROTATE")

    def test_13_crate_elevated_terrain_adjusts_geometry(self):
        bot = MechBot(make_hello())
        bot.world.load_synthetic((-100.0, 100.0, -100.0, 100.0), [], flat_z=0.0)
        # Crate at z=5.0m above flat ground -> target_z=6.44, muzzle_z=7.0 -> dz=-0.56 -> valid even closer
        crate_high = {"id": 1006, "pos": [10.0, 0.0, 5.0], "hp": 300.0}
        self.assertTrue(bot.resources.has_valid_crate_geometry((0.0, 0.0, 7.0), (10.0, 0.0, 5.0)))
        # Crate 12m below elevated mech (muzzle_z=19.0, target_z=1.44 -> dz=-17.56): 10m is too steep!
        self.assertFalse(bot.resources.has_valid_crate_geometry((0.0, 0.0, 19.0), (10.0, 0.0, 0.0)))
        self.assertTrue(bot.resources.has_valid_crate_geometry((0.0, 0.0, 19.0), (26.0, 0.0, 0.0)))

    # --- Priority B: Multi-Tick Retreat Effectiveness Regression Tests ---

    def test_14_retreat_multi_tick_increases_separation_and_displacement(self):
        bot = MechBot(make_hello())
        px, py = 0.0, 0.0
        ex, ey = 35.0, 0.0
        start_d = math.hypot(px - ex, py - ey)
        for step in range(15):
            enemy = {
                "id": 7, "kind": "mech", "pos": [ex, ey, 0.0], "vel": [-8.0, 0.0, 0.0],
                "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 1, "weapons": ["gun"],
            }
            obs = make_obs(tick=100 + step * 2, t=10.0 + step * 0.1, pos=[px, py, 0.0], hp=420.0, mechs=[enemy])
            act = bot.act(obs)
            self.assertEqual(bot.strategy.mode, "RETREAT_HEAL")
            px += act["move"][0] * 1.6
            py += act["move"][1] * 1.6
            ex -= 0.8  # pursuer chasing
        end_d = math.hypot(px - ex, py - ey)
        self.assertGreater(end_d, start_d)
        self.assertLess(px, -5.0)

    def test_15_retreat_breaks_los_behind_building_cover(self):
        bot = MechBot(make_hello())
        # Building prism to the west (-22..-12, -10..10) between cover cells and east enemy
        wall = Prism("building", -20.0, 30.0, [-22.0, -12.0, -12.0, -12.0, -12.0, 12.0, -22.0, 12.0], [])
        bot.world.load_synthetic((-120.0, 120.0, -120.0, 120.0), [wall], flat_z=0.0)
        enemy = {
            "id": 8, "kind": "mech", "pos": [45.0, 0.0, 0.0], "vel": [-5.0, 0.0, 0.0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        obs = make_obs(pos=[0.0, 0.0, 0.0], hp=400.0, mechs=[enemy])
        bot.act(obs)
        gx, gy = bot.strategy.last_decision_meta["goal"]
        self.assertFalse(bot.world.has_los((45.0, 0.0, 7.5), (gx, gy, 5.0), step=3.5))

    def test_16_retreat_selective_boost_preserves_energy_when_los_broken(self):
        bot = MechBot(make_hello())
        wall = Prism("building", -20.0, 30.0, [10.0, -15.0, 20.0, -15.0, 20.0, 15.0, 10.0, 15.0], [])
        bot.world.load_synthetic((-120.0, 120.0, -120.0, 120.0), [wall], flat_z=0.0)
        # Enemy behind wall (no LoS) -> still in retreat via recent damage, but must NOT burn boost
        enemy_hidden = {
            "id": 9, "kind": "mech", "pos": [40.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        obs_no_los = make_obs(pos=[0.0, 0.0, 0.0], hp=420.0, energy=80.0, mechs=[enemy_hidden])
        act_no_los = bot.act(obs_no_los)
        self.assertFalse(act_no_los["boost"], "Should preserve energy once LoS is broken and no incoming fire")

        # Enemy with direct LoS -> should boost while energy > 25
        obs_los = make_obs(pos=[0.0, 40.0, 0.0], hp=420.0, energy=80.0, mechs=[enemy_hidden])
        act_los = bot.act(obs_los)
        self.assertTrue(act_los["boost"])

    def test_17_retreat_stays_inside_safe_zone(self):
        bot = MechBot(make_hello())
        enemy = {
            "id": 10, "kind": "mech", "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "aim": [0.0, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        # Bot near east zone edge (45, 0) with next_radius=60 -> must not pick retreat goal outside safe zone
        obs = make_obs(
            pos=[45.0, 0.0, 0.0],
            hp=400.0,
            mechs=[enemy],
            zone={
                "center": [0.0, 0.0], "radius": 60.0,
                "next_center": [0.0, 0.0], "next_radius": 60.0,
                "shrink_start": 100.0, "shrink_end": 130.0, "dps": 20.0,
            },
        )
        bot.act(obs)
        gx, gy = bot.strategy.last_decision_meta["goal"]
        self.assertLessEqual(math.hypot(gx, gy), 56.0)

    def test_18_retreat_avoids_npc_aggro_zone(self):
        bot = MechBot(make_hello())
        enemy = {
            "id": 11, "kind": "mech", "pos": [50.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        npc = {"id": 111, "kind": "npc", "pos": [-42.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [0, 0], "hp": 350, "max_hp": 350}
        obs = make_obs(pos=[0.0, 0.0, 0.0], hp=400.0, mechs=[enemy, npc])
        bot.act(obs)
        gx, gy = bot.strategy.last_decision_meta["goal"]
        self.assertGreater(math.hypot(gx - (-42.0), gy - 0.0), 25.0)

    def test_19_retreat_recovers_when_threat_disappears_or_favourable_finish(self):
        bot = MechBot(make_hello())
        enemy_full = {
            "id": 12, "kind": "mech", "pos": [35.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        bot.act(make_obs(tick=10, t=1.0, pos=[0.0, 0.0, 0.0], hp=420.0, mechs=[enemy_full]))
        self.assertEqual(bot.strategy.mode, "RETREAT_HEAL")

        # Threat disappears -> exits RETREAT_HEAL
        bot.act(make_obs(tick=120, t=12.0, pos=[0.0, 0.0, 0.0], hp=420.0, mechs=[]))
        self.assertNotEqual(bot.strategy.mode, "RETREAT_HEAL")

        # Favourable finish: enemy at 150 HP, bot at 420 HP -> switches to COMBAT_KITE to finish
        enemy_low = {
            "id": 12, "kind": "mech", "pos": [35.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "aim": [math.pi, 0.0], "hp": 150.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"],
        }
        bot.act(make_obs(tick=122, t=12.1, pos=[0.0, 0.0, 0.0], hp=420.0, mechs=[enemy_low]))
        self.assertEqual(bot.strategy.mode, "COMBAT_KITE")

    # --- Priority C: Zero-Cost Telemetry Bypass & Local Tracing Tests ---

    def test_20_telemetry_disabled_mode_fast_path(self):
        prev = os.environ.pop("MECH_TELEMETRY_PATH", None)
        try:
            bot = MechBot(make_hello(), trace_path=None)
            self.assertFalse(bot.telemetry.file_enabled)
            obs = make_obs(tick=10, t=1.0, pos=[0.0, 0.0, 0.0], hp=1400.0)
            bot.act(obs)
            self.assertEqual(len(bot.telemetry.ring), 0)
            self.assertIsNone(bot.telemetry._prev_record)
        finally:
            if prev is not None:
                os.environ["MECH_TELEMETRY_PATH"] = prev

    def test_21_telemetry_enabled_records_retreat_outcomes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.jsonl")
            bot = MechBot(make_hello(), trace_path=trace_file)
            self.assertTrue(bot.telemetry.file_enabled)
            e1 = {"id": 13, "kind": "mech", "pos": [30.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
            e2 = {"id": 13, "kind": "mech", "pos": [42.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
            bot.act(make_obs(tick=10, t=1.0, pos=[0.0, 0.0, 0.0], hp=400.0, mechs=[e1]))
            bot.act(make_obs(tick=12, t=1.1, pos=[-2.0, 0.0, 0.0], hp=400.0, mechs=[e2]))
            bot.telemetry.finalize()
            with open(trace_file, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            self.assertEqual(first["selected_objective"], "RETREAT_HEAL")
            self.assertAlmostEqual(first["consequence"]["retreat_sep_delta"], 14.0, places=1)
            self.assertFalse(first["consequence"]["retreat_los_broken"])

    # --- Project Overdrive (overdrive-v5) Architecture & Regression Tests ---

    def test_22_kill_event_prunes_enemy_and_zeroes_danger(self):
        bot = MechBot(make_hello())
        enemy = {
            "id": 14, "kind": "mech", "pos": [20.0, 0.0, 0.0], "vel": [0, 0, 0],
            "aim": [math.pi, 0], "hp": 40.0, "max_hp": 1400.0, "cubes": 3, "weapons": ["gun"],
        }
        bot.act(make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], mechs=[enemy]))
        self.assertIn(14, bot.threats.enemies)
        self.assertGreater(bot.threats.danger_at(20.0, 0.0, 10.0), 0.5)

        cube = {"id": 5099, "pos": [20.0, 0.0, 0.0]}
        bot.act(make_obs(
            tick=102, t=10.1, pos=[0.0, 0.0, 0.0],
            mechs=[], cubes=[cube],
            events=[{"type": "kill", "target": 14}],
        ))
        self.assertNotIn(14, bot.threats.enemies)
        self.assertEqual(bot.threats.danger_at(20.0, 0.0, 10.1), 0.0)
        self.assertEqual(bot.strategy.mode, "LOOT_CUBE")

    def test_23_beams_never_inject_phantom_enemies_and_close_beam_sets_rear_alert(self):
        bot = MechBot(make_hello())
        obs = make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0])
        obs["beams"] = [
            {"by": 9, "kind": "laser", "path": [[-80.0, 50.0, 7.0], [-40.0, 50.0, 7.0]], "age": 0.1}
        ]
        bot.act(obs)
        self.assertEqual(len(bot.threats.enemies), 0)
        self.assertEqual(bot.threats.danger_at(-60.0, 50.0, 10.0), 0.0)
        self.assertIsNone(bot.threats.rear_alert_bearing)

        obs2 = make_obs(tick=102, t=10.1, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0])
        obs2["beams"] = [
            {"by": 9, "kind": "laser", "path": [[-50.0, 2.0, 7.0], [10.0, 2.0, 7.0]], "age": 0.1}
        ]
        bot.act(obs2)
        self.assertEqual(len(bot.threats.enemies), 0)
        self.assertIsNotNone(bot.threats.rear_alert_bearing)

    def test_24_zero_delay_dual_fire_gun_laser_and_velocity_led_lightning(self):
        bot = MechBot(make_hello())
        weapons_laser = [
            {"kind": "gun", "cooldown": 0.0, "ammo": 10, "reload": 0.0, "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4}},
            {"kind": "laser", "cooldown": 0.0, "charge": 0.25, "stats": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "hold_s": 2.0, "cooldown": 1.65, "range": 280}},
        ]
        # Enemy moving slowly so hitscan_aim and gun_lead are both within gun tol -> fires BOTH with aim locked on hitscan_aim
        e_mid = {"id": 5, "kind": "mech", "pos": [100.0, 0.0, 0.0], "vel": [0.0, 3.0, 0.0], "aim": [math.pi, 0.0], "hp": 1000.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs_l = make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], aim=[0.0, -0.02], weapons=weapons_laser, mechs=[e_mid])
        act_l = bot.act(obs_l)
        self.assertEqual(act_l["fire"], [True, True], "Should dual-fire gun while laser is charging")
        self.assertAlmostEqual(act_l["aim"][0], math.atan2(3.0 * 0.05, 100.0), places=3, msg="Aim must stay locked on hitscan intercept")

        weapons_light = [
            {"kind": "gun", "cooldown": 0.0, "ammo": 10, "reload": 0.0, "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4}},
            {"kind": "lightning", "cooldown": 0.0, "stats": {"damage": 60, "cooldown": 0.25, "range": 45, "cone_deg": 30, "energy": 3}},
        ]
        e_close = {"id": 6, "kind": "mech", "pos": [30.0, 0.0, 0.0], "vel": [0.0, 18.0, 0.0], "aim": [math.pi, 0.0], "hp": 1000.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs_lt = make_obs(tick=104, t=10.2, pos=[0.0, 0.0, 0.0], aim=[0.0, -0.08], weapons=weapons_light, mechs=[e_close])
        act_lt = bot.act(obs_lt)
        self.assertEqual(act_lt["fire"], [True, True])
        self.assertGreater(act_lt["aim"][0], 0.005, "Gun+lightning close dual-fire should apply velocity lead")

    def test_25_anti_laser_duck_breaks_los_with_boost(self):
        bot = MechBot(make_hello())
        wall = Prism("building", -20.0, 30.0, [-14.0, -25.0, -6.0, -25.0, -6.0, -4.0, -14.0, -4.0], [])
        bot.world.load_synthetic((-120.0, 120.0, -120.0, 120.0), [wall], flat_z=0.0)
        sniper = {
            "id": 21, "kind": "mech", "pos": [90.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0,
            "weapons": ["gun", "laser"], "charging": True,
        }
        obs = make_obs(tick=200, t=20.0, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0], hp=1100.0, energy=80.0, mechs=[sniper])
        act = bot.act(obs)
        self.assertEqual(bot.strategy.mode, "ANTI_LASER_DUCK")
        self.assertTrue(act["boost"])
        gx, gy = bot.strategy.last_decision_meta["goal"]
        self.assertFalse(bot.world.has_los((90.0, 0.0, 7.5), (gx, gy, 5.0), step=3.5))

        # If already charging a laser while ducking >7m to cover, cancel/release charge to restore full sprint speed
        weapons_charging = [
            {"kind": "gun", "cooldown": 0.0, "ammo": 10, "reload": 0.0, "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4}},
            {"kind": "laser", "cooldown": 0.0, "charge": 0.25, "stats": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "hold_s": 2.0, "cooldown": 1.65, "range": 280}},
        ]
        obs_chg = make_obs(tick=202, t=20.1, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0], hp=1100.0, energy=80.0, weapons=weapons_charging, mechs=[sniper])
        act_chg = bot.act(obs_chg)
        self.assertEqual(bot.strategy.mode, "ANTI_LASER_DUCK")
        self.assertFalse(act_chg["fire"][1], "Must release/cancel half-speed laser charge when sprinting >7m to duck sniper")

    def test_26_surgical_regen_discipline(self):
        bot = MechBot(make_hello())
        # Last combat 4.0s ago, HP at ~66% (above bad_duel 65%, below 72% regen threshold), enemy far at 110m healthy -> hold fire
        far_e = {"id": 31, "kind": "mech", "pos": [110.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1200.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        bot.threats.last_damaged_t = 6.0
        obs_far = make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0], hp=920.0, mechs=[far_e])
        act_far = bot.act(obs_far)
        self.assertFalse(any(act_far["fire"]), "Should hold fire on distant healthy enemy to keep regen timer")

        # Enemy closes to 50m -> MUST fight immediately
        close_e = {"id": 31, "kind": "mech", "pos": [50.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1200.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs_close = make_obs(tick=102, t=10.1, pos=[0.0, 0.0, 0.0], aim=[0.0, -0.05], hp=920.0, mechs=[close_e])
        act_close = bot.act(obs_close)
        self.assertTrue(act_close["fire"][0], "Must engage close threat even while regenerating")

    def test_27_target_hysteresis_and_defensive_drop_hold(self):
        bot = MechBot(make_hello())
        e1 = {"id": 1, "kind": "mech", "pos": [70.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [0.0, 0.0], "hp": 900.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        e2 = {"id": 2, "kind": "mech", "pos": [0.0, 68.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi / 2, 0.0], "hp": 900.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs1 = make_obs(tick=100, t=10.0, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0], mechs=[e1])
        bot.threats.update(obs1)
        self.assertEqual(bot.combat.select_mech_target(obs1, bot.threats).id, 1)

        obs_tie = make_obs(tick=102, t=10.1, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0], mechs=[e1, e2])
        bot.threats.update(obs_tie)
        self.assertEqual(bot.combat.select_mech_target(obs_tie, bot.threats).id, 1)

        e2_close = {"id": 2, "kind": "mech", "pos": [0.0, 35.0, 0.0], "vel": [0, 0, 0], "aim": [-math.pi / 2, 0.0], "hp": 900.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs_atk = make_obs(
            tick=104, t=10.2, pos=[0.0, 0.0, 0.0], aim=[0.0, 0.0], mechs=[e1, e2_close],
            events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": math.pi / 2, "by": 2}],
        )
        bot.threats.update(obs_atk)
        self.assertEqual(bot.combat.select_mech_target(obs_atk, bot.threats).id, 2)

        bot2 = MechBot(make_hello())
        drop = {"id": 20001, "pos": [2.0, 0.0, 0.0], "radius": 8.0, "lands_in": 0.0, "capture_s": 4.0, "progress": 1.5, "holder": 0}
        challenger = {"id": 3, "kind": "mech", "pos": [45.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1000.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        act_d = bot2.act(make_obs(tick=200, t=20.0, pos=[0.0, 0.0, 0.0], aim=[0.0, -0.05], mechs=[challenger], drops=[drop]))
        self.assertEqual(bot2.strategy.mode, "LOOT_DROP")
        self.assertTrue(act_d["fire"][0])

    def test_28_phase_weapon_swap_upgrades_and_crate_cache(self):
        bot = MechBot(make_hello())
        weapons_laser = [
            {"kind": "gun", "cooldown": 0.0, "ammo": 12, "reload": 0.0, "stats": {}},
            {"kind": "laser", "cooldown": 0.0, "charge": 0.0, "stats": {}},
        ]
        light_pickup = {"id": 8050, "kind": "lightning", "pos": [1.0, 0.0, 0.0]}
        obs_mid = make_obs(tick=300, t=30.0, pos=[0.0, 0.0, 0.0], weapons=weapons_laser, pickups=[light_pickup])
        self.assertEqual(bot.resources.choose_weapon_take(obs_mid, phase="MID_GAME"), (None, None))
        self.assertEqual(bot.resources.choose_weapon_take(obs_mid, phase="LATE_GAME"), (8050, 1))

        offer = {
            "id": 1, "expires": 35.0,
            "choices": [
                {"id": "reactor", "name": "Reactor", "desc": ""},
                {"id": "laser_charge", "name": "Capacitor", "desc": ""},
            ],
        }
        obs_up = make_obs(tick=302, t=30.1, weapons=weapons_laser, offer=offer)
        self.assertEqual(bot.resources.choose_upgrade(obs_up, phase="MID_GAME"), 1)

        pt1 = bot.resources.find_crate_firing_position(obs_mid, (2.0, 0.0, 0.0), bot.threats, (0.0, 0.0), 400.0, crate_id=1010)
        self.assertIsNotNone(pt1)
        self.assertIn(1010, bot.resources._crate_standoff_cache)

    def test_29_proactive_zone_advance_before_urgent(self):
        bot = MechBot(make_hello())
        zone = {
            "center": [0.0, 0.0], "radius": 400.0,
            "next_center": [250.0, 0.0], "next_radius": 140.0,
            "shrink_start": 120.0, "shrink_end": 160.0, "dps": 35.0,
        }
        obs_adv = make_obs(tick=1000, t=95.0, pos=[0.0, 0.0, 0.0], in_zone=True, zone=zone)
        bot.act(obs_adv)
        self.assertEqual(bot.strategy.mode, "ZONE_ADVANCE")


if __name__ == "__main__":
    unittest.main()
