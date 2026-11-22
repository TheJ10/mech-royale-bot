"""Top-level Unified Battle Intelligence MechBot controller with bounded telemetry."""
from __future__ import annotations

import atexit
import os
import time
from typing import Any, Dict, Optional

import mechfight as mf

from bot.combat import CombatPlanner
from bot.nav import Navigator
from bot.resources import ResourcePlanner
from bot.strategy import PROFILES, StrategyController
from bot.telemetry import DecisionTelemetry
from bot.threat import ThreatTracker
from bot.world import WorldModel


class MechBot:
    """Project Overdrive autonomous Mech Royale bot (overdrive-v5)."""

    profile_name = os.environ.get("MECH_PROFILE", "unified")
    name = PROFILES.get(profile_name, PROFILES["unified"]).bot_name

    def __init__(self, hello: mf.Hello, trace_path: Optional[str] = None):
        self.profile = PROFILES.get(self.profile_name, PROFILES["unified"])
        self.name = self.profile.bot_name
        config: Dict[str, Any] = hello.get("config") or {}
        mech_cfg = config.get("mech") or {}
        map_dir = (hello.get("map") or {}).get("dir")

        self.world = WorldModel(
            map_dir=map_dir,
            mech_radius=float(mech_cfg.get("radius", 2.5)),
            mech_height=float(mech_cfg.get("height", 9.0)),
            step_up=float(mech_cfg.get("step_up", 1.2)),
        )
        self.nav = Navigator(self.world)
        self.threats = ThreatTracker(self.world, config)
        self.combat = CombatPlanner(self.world, config)
        self.resources = ResourcePlanner(self.world, config, pacifist=self.profile.pacifist)
        self.strategy = StrategyController(
            self.world,
            self.nav,
            self.threats,
            self.combat,
            self.resources,
            self.profile,
            seed=int(hello.get("seed", 0)),
        )
        self.telemetry = DecisionTelemetry(trace_path=trace_path)
        if self.telemetry.file_enabled:
            atexit.register(self.telemetry.finalize)

    def act(self, obs: mf.Observation) -> mf.Action:
        t0 = time.perf_counter()
        self.threats.update(obs)
        act, mode, goal = self.strategy.decide(obs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if self.telemetry.file_enabled:
            self.telemetry.record_tick(obs, self.strategy.last_decision_meta, act, dt_ms)

        me = obs["self"]
        x, y, z = me["pos"]
        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 5)
        mf.draw.label(f"{mode} hp:{me['hp']:.0f} c:{me['cubes']} {dt_ms:.1f}ms")
        return act
