"""Comprehensive unit, geometry, combat, threat, and 10-scenario tests for MechBot."""
from __future__ import annotations

import math
import os
import time
import unittest
from typing import Any, Dict, List, Optional

from bot.combat import CombatPlanner, angle_diff, solve_lead_point, solve_mortar_aim
from bot.controller import MechBot
from bot.nav import Navigator
from bot.resources import ResourcePlanner
from bot.strategy import PROFILES, StrategyController
from bot.threat import ThreatTracker
from bot.world import Prism, WorldModel

ARENA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arena")


def make_hello(map_dir: Optional[str] = None) -> Dict[str, Any]:
    return {
        "type": "hello",
        "protocol": "mechfight/1",
        "slot": 0,
        "seed": 42,
        "map": {"dir": map_dir or "", "files": []},
        "config": {
            "mech": {"radius": 2.5, "height": 9.0, "step_up": 1.2, "min_pitch": -0.7, "max_pitch": 1.3},
            "loot": {"pickup_radius": 4.0},
            "weapons": {
                "gun": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4},
                "mortar": {"damage": 200, "splash": 160, "splash_radius": 7, "cooldown": 2.6, "speed": 72, "gravity": 22},
                "laser": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "hold_s": 2.0, "cooldown": 1.65, "range": 280},
                "lightning": {"damage": 60, "cooldown": 0.25, "range": 45, "cone_deg": 30, "energy": 3},
            },
        },
    }


def make_obs(
    tick: int = 100,
    t: float = 10.0,
    pos: List[float] = None,
    vel: List[float] = None,
    aim: List[float] = None,
    hp: float = 1400.0,
    max_hp: float = 1400.0,
    energy: float = 100.0,
    grounded: bool = True,
    in_zone: bool = True,
    weapons: List[Any] = None,
    mechs: List[Any] = None,
    projectiles: List[Any] = None,
    crates: List[Any] = None,
    cubes: List[Any] = None,
    pickups: List[Any] = None,
    sounds: List[Any] = None,
    events: List[Any] = None,
    drops: List[Any] = None,
    zone: Dict[str, Any] = None,
    offer: Any = None,
) -> Dict[str, Any]:
    if weapons is None:
        weapons = [
            {
                "kind": "gun",
                "cooldown": 0.0,
                "ammo": 12,
                "reload": 0.0,
                "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4},
            },
            None,
        ]
    return {
        "type": "obs",
        "tick": tick,
        "t": t,
        "alive": 12,
        "self": {
            "id": 0,
            "pos": pos or [0.0, 0.0, 0.0],
            "vel": vel or [0.0, 0.0, 0.0],
            "aim": aim or [0.0, 0.0],
            "hp": hp,
            "max_hp": max_hp,
            "energy": energy,
            "max_energy": 100.0,
            "grounded": grounded,
            "cubes": 0,
            "upgrades": [],
            "in_zone": in_zone,
            "weapons": weapons,
            "stats": {"walk_speed": 11.7, "boost_speed": 19.5, "turn_rate": 4.0, "muzzle_height": 7.0},
        },
        "mechs": mechs or [],
        "projectiles": projectiles or [],
        "crates": crates or [],
        "cubes": cubes or [],
        "pickups": pickups or [],
        "sounds": sounds or [],
        "beams": [],
        "events": events or [],
        "drops": drops or [],
        "zone": zone
        or {
            "center": [0.0, 0.0],
            "radius": 400.0,
            "next_center": [0.0, 0.0],
            "next_radius": 250.0,
            "shrink_start": 60.0,
            "shrink_end": 110.0,
            "dps": 20.0,
        },
        "offer": offer,
    }


class TestWorldModel(unittest.TestCase):
    def test_real_arena_loads_fast_and_valid(self):
        t0 = time.perf_counter()
        w = WorldModel(ARENA_DIR)
        elapsed = time.perf_counter() - t0
        self.assertTrue(w.loaded)
        self.assertEqual(len(w.prisms), 259)
        self.assertLess(elapsed, 3.0, f"WorldModel load took too long: {elapsed:.2f}s")
        # Terrain lookup works and returns finite elevation
        z = w.terrain_height(0.0, 0.0)
        self.assertTrue(math.isfinite(z))
        self.assertGreater(sum(w.walkable), 2000)

    def test_synthetic_building_blocks_walk_and_los(self):
        w = WorldModel()
        # 20x20 building centered at (50, 0), height 0..20
        wall = Prism("building", -2.0, 20.0, [40, -10, 60, -10, 60, 10, 40, 10], [])
        w.load_synthetic((-100, 100, -100, 100), [wall])
        self.assertTrue(w.is_blocked_at(50.0, 0.0, 0.0))
        self.assertFalse(w.is_blocked_at(0.0, 0.0, 0.0))
        self.assertFalse(w.has_los((0.0, 0.0, 7.0), (90.0, 0.0, 7.0)))
        self.assertTrue(w.has_los((0.0, 0.0, 25.0), (90.0, 0.0, 25.0)))

    def test_bridge_clearance_allows_walking_under(self):
        w = WorldModel()
        # Bridge with base at z=11 (> mech height 9.0)
        bridge = Prism("bridge", 11.0, 14.0, [40, -10, 60, -10, 60, 10, 40, 10], [])
        w.load_synthetic((-100, 100, -100, 100), [bridge])
        self.assertFalse(w.is_blocked_at(50.0, 0.0, 0.0))

    def test_hollow_prism_courtyard_is_walkable(self):
        w = WorldModel()
        # Outer 60x60 ring with inner 30x30 courtyard hole centered at (0, 0)
        outer = [-30, -30, 30, -30, 30, 30, -30, 30]
        hole = [-15, -15, 15, -15, 15, 15, -15, 15]
        stadium = Prism("stadium", -2.0, 25.0, outer, [hole])
        w.load_synthetic((-100, 100, -100, 100), [stadium])
        self.assertFalse(w.is_blocked_at(0.0, 0.0, 0.0))
        self.assertTrue(w.is_blocked_at(22.0, 0.0, 0.0))


class TestNavigation(unittest.TestCase):
    def test_astar_routes_around_building(self):
        w = WorldModel()
        wall = Prism("building", -2.0, 25.0, [40, -30, 60, -30, 60, 30, 40, 30], [])
        w.load_synthetic((-100, 120, -100, 100), [wall])
        nav = Navigator(w)
        path = nav.plan_path((0.0, 0.0), (90.0, 0.0))
        self.assertTrue(len(path) >= 2)
        # Ensure no waypoint is inside the building
        for px, py in path:
            self.assertFalse(wall.contains_xy(px, py))


class TestCombat(unittest.TestCase):
    def test_gun_velocity_lead(self):
        # Target at (100, 0, 7) moving +Y at 15 m/s, bullet speed 250 m/s -> ~0.4s lead -> y ~ 6m
        pt = solve_lead_point((0.0, 0.0, 7.0), (100.0, 0.0, 7.0), (0.0, 15.0, 0.0), 250.0)
        self.assertGreater(pt[1], 5.0)
        self.assertLess(pt[1], 7.0)

    def test_mortar_ballistics_and_close_guard(self):
        # Too close -> None (prevents self-damage)
        self.assertIsNone(solve_mortar_aim((0, 0, 9.5), (10, 0, 4.5), (0, 0, 0), 72.0, 22.0))
        # Valid mid-range -> valid yaw & pitch
        sol = solve_mortar_aim((0, 0, 9.5), (100, 0, 4.5), (0, 0, 0), 72.0, 22.0)
        self.assertIsNotNone(sol)
        self.assertAlmostEqual(sol[0], 0.0, places=2)
        self.assertGreater(sol[1], 0.1)

    def test_laser_charge_and_release(self):
        w = WorldModel()
        cp = CombatPlanner(w, make_hello()["config"])
        laser_w = {
            "kind": "laser",
            "cooldown": 0.0,
            "charge": 0.0,
            "stats": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "range": 280},
        }
        # Step 1: Not charging yet, aimed at target -> starts charging (fire[1] = True)
        obs1 = make_obs(aim=[0.0, 0.0], weapons=[None, laser_w])
        p1 = cp.plan_attack_on_point(obs1, (100.0, 0.0, 7.0), (0, 0, 0), 5, ["mortar"], 1000.0)
        self.assertTrue(p1.fire[1])

        # Step 2: Charged to 0.70s and aimed -> RELEASES trigger (fire[1] = False) to fire beam!
        laser_w2 = dict(laser_w, charge=0.70)
        obs2 = make_obs(aim=[0.0, 0.0], weapons=[None, laser_w2])
        p2 = cp.plan_attack_on_point(obs2, (100.0, 0.0, 7.0), (0, 0, 0), 5, ["mortar"], 1000.0)
        self.assertFalse(p2.fire[1])


class TestScenarios(unittest.TestCase):
    """Covers all 10 required tactical scenarios."""

    def setUp(self):
        self.bot = MechBot(make_hello(ARENA_DIR))

    def test_01_spawn_near_map_edge(self):
        obs = make_obs(pos=[-505.0, -325.0, 0.0])
        act = self.bot.act(obs)
        # Should move inward (+x, +y) away from map boundary
        self.assertGreater(act["move"][0], 0.0)
        self.assertGreater(act["move"][1], 0.0)

    def test_02_direct_route_blocked_by_building(self):
        w = WorldModel()
        wall = Prism("building", -2.0, 25.0, [30, -25, 50, -25, 50, 25, 30, 25], [])
        w.load_synthetic((-100, 120, -100, 100), [wall])
        bot = MechBot(make_hello())
        bot.world = w
        bot.nav = Navigator(w)
        bot.strategy.world = w
        bot.strategy.nav = bot.nav
        obs = make_obs(
            pos=[0.0, 0.0, 0.0],
            zone={"center": [85.0, 0.0], "radius": 300.0, "next_center": [85.0, 0.0], "next_radius": 20.0, "shrink_start": 12.0, "shrink_end": 25.0, "dps": 20.0},
        )
        act = bot.act(obs)
        # Path should route around wall (non-zero Y component)
        self.assertGreater(abs(act["move"][1]), 0.15)

    def test_03_approaching_zone_closure(self):
        obs = make_obs(
            t=55.0,
            pos=[-250.0, 0.0, 0.0],
            in_zone=False,
            zone={"center": [0.0, 0.0], "radius": 200.0, "next_center": [0.0, 0.0], "next_radius": 120.0, "shrink_start": 50.0, "shrink_end": 90.0, "dps": 35.0},
        )
        act = self.bot.act(obs)
        self.assertTrue(act["boost"])
        self.assertEqual(self.bot.strategy.mode, "ZONE_URGENT")

    def test_04_multiple_nearby_opponents(self):
        enemies = [
            {"id": 2, "kind": "mech", "pos": [25.0, 5.0, 0.0], "vel": [0, 0, 0], "aim": [3.14, 0], "hp": 1400, "max_hp": 1400, "cubes": 2, "weapons": ["gun"]},
            {"id": 3, "kind": "mech", "pos": [20.0, -15.0, 0.0], "vel": [0, 0, 0], "aim": [2.5, 0], "hp": 1400, "max_hp": 1400, "cubes": 1, "weapons": ["gun"]},
        ]
        obs = make_obs(pos=[0.0, 0.0, 0.0], hp=900.0, mechs=enemies)
        self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "RETREAT_HEAL")

    def test_05_low_health_retreat(self):
        enemy = {"id": 4, "kind": "mech", "pos": [50.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [3.14, 0], "hp": 1200, "max_hp": 1400, "cubes": 0, "weapons": ["gun"]}
        obs = make_obs(pos=[0.0, 0.0, 0.0], hp=400.0, mechs=[enemy])
        act = self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "RETREAT_HEAL")
        # Should stop firing at mid-range while retreating to start regen timer
        self.assertFalse(any(act["fire"]))

    def test_06_rear_attack_turns_to_face(self):
        obs = make_obs(
            pos=[0.0, 0.0, 0.0],
            aim=[0.0, 0.0],
            events=[{"type": "damaged", "amount": 90, "weapon": "gun", "bearing": math.pi, "by": 9, "rear": True}],
        )
        act = self.bot.act(obs)
        self.assertLess(abs(angle_diff(act["aim"][0], math.pi)), 0.1)

    def test_07_incoming_mortar_dodges_perpendicular(self):
        obs = make_obs(
            pos=[0.0, 0.0, 0.0],
            projectiles=[{"id": 99, "kind": "mortar", "pos": [35.0, 0.0, 12.0], "vel": [-50.0, 0.0, -5.0]}],
        )
        dodge = self.bot.threats.compute_dodge_vector(obs, (0.0, 0.0))
        self.assertIsNotNone(dodge)
        # Perpendicular to (-50, 0) means predominantly Y movement
        self.assertGreater(abs(dodge[1]), 0.8)

    def test_08_nearby_weapons_and_cubes_and_upgrade(self):
        obs = make_obs(
            pos=[0.0, 0.0, 0.0],
            pickups=[{"id": 8001, "kind": "lightning", "pos": [2.0, 0.0, 0.0]}],
            cubes=[{"id": 5001, "pos": [15.0, 0.0, 0.0]}],
            offer={"id": 1, "expires": 15.0, "choices": [
                {"id": "mortar_splash", "name": "Cluster", "desc": ""},
                {"id": "nanites", "name": "Repair nanites", "desc": ""},
                {"id": "capacitor", "name": "Cap", "desc": ""},
            ]},
        )
        act = self.bot.act(obs)
        self.assertEqual(act.get("take"), 8001)
        self.assertEqual(act.get("pick"), 1)

    def test_09_rooftop_navigation_safe(self):
        obs = make_obs(pos=[0.0, 0.0, 18.0], grounded=True)
        act = self.bot.act(obs)
        self.assertIn("move", act)
        self.assertEqual(len(act["move"]), 2)

    def test_10_final_zone_positioning(self):
        obs = make_obs(
            t=390.0,
            pos=[10.0, 10.0, 0.0],
            zone={"center": [0.0, 0.0], "radius": 25.0, "next_center": [0.0, 0.0], "next_radius": 10.0, "shrink_start": 400.0, "shrink_end": 420.0, "dps": 260.0},
        )
        act = self.bot.act(obs)
        # Should move toward (0, 0)
        self.assertLess(act["move"][0], 0.0)
        self.assertLess(act["move"][1], 0.0)

    def test_11_gun_reloading_falls_through_to_mortar(self):
        w = WorldModel()
        cp = CombatPlanner(w, make_hello()["config"])
        gun_reloading = {
            "kind": "gun",
            "cooldown": 0.0,
            "ammo": 0,
            "reload": 1.8,
            "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12, "reload": 2.4},
        }
        mortar_ready = {
            "kind": "mortar",
            "cooldown": 0.0,
            "stats": {"damage": 200, "splash": 160, "speed": 72, "gravity": 22},
        }
        obs = make_obs(weapons=[gun_reloading, mortar_ready])
        plan = cp.plan_attack_on_point(obs, (80.0, 0.0, 4.5), (0, 0, 0), 5, ["gun"], 1000.0)
        self.assertEqual(plan.weapon_kind, "mortar")

    def test_12_endgame_1v1_allows_combat_kiting(self):
        enemy = {"id": 6, "kind": "mech", "pos": [15.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [3.14, 0], "hp": 900, "max_hp": 1400, "cubes": 1, "weapons": ["gun"]}
        obs = make_obs(
            t=415.0,
            pos=[0.0, 0.0, 0.0],
            in_zone=True,
            mechs=[enemy],
            zone={"center": [0.0, 0.0], "radius": 25.0, "next_center": [0.0, 0.0], "next_radius": 0.0, "shrink_start": 405.0, "shrink_end": 430.0, "dps": 260.0},
        )
        self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "COMBAT_KITE")

    def test_13_selective_engage_skips_noisy_midrange_full_hp_duel(self):
        bot = MechBot(make_hello())
        # Healthy enemy at 110m facing perpendicular (not aiming at us, not wounded, alive=12)
        enemy = {"id": 11, "kind": "mech", "pos": [110.0, 0.0, 0.0], "vel": [0, 0, 0], "aim": [1.57, 0.0], "hp": 1400.0, "max_hp": 1400.0, "cubes": 0, "weapons": ["gun"]}
        obs = make_obs(pos=[0.0, 0.0, 0.0], mechs=[enemy])
        act = bot.act(obs)
        self.assertNotEqual(bot.strategy.mode, "COMBAT_KITE")
        self.assertEqual(act["fire"], [False, False])

    def test_14_rooftop_hold_when_safe_in_zone(self):
        obs = make_obs(pos=[0.0, 0.0, 18.0], grounded=True, in_zone=True)
        act = self.bot.act(obs)
        self.assertEqual(self.bot.strategy.mode, "ROOF_HOLD")
        self.assertEqual(act["move"], [0.0, 0.0])

    def test_15_escape_degree_populated_on_real_arena(self):
        self.assertEqual(len(self.bot.world.escape_deg), len(self.bot.world.walkable))
        self.assertGreater(max(self.bot.world.escape_deg), 4)


if __name__ == "__main__":
    unittest.main()
