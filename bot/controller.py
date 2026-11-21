"""Top-level Mech Royale bot controller wiring all modules with timing guard."""
from __future__ import annotations

import os
import time
from typing import Any, Dict

import mechfight as mf

from bot.combat import CombatPlanner
from bot.nav import Navigator
from bot.resources import ResourcePlanner
from bot.strategy import PROFILES, StrategyController
from bot.threat import ThreatTracker
from bot.world import WorldModel


class MechBot:
    """Modular autonomous Mech Royale bot."""

    profile_name = os.environ.get("MECH_PROFILE", "pure_pacifist")
    name = PROFILES.get(profile_name, PROFILES["pure_pacifist"]).bot_name

    def __init__(self, hello: mf.Hello):
        self.profile = PROFILES.get(self.profile_name, PROFILES["pure_pacifist"])
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
        )

    def act(self, obs: mf.Observation) -> mf.Action:
        t0 = time.perf_counter()
        self.threats.update(obs)
        act, mode, goal = self.strategy.decide(obs)

        # Lightweight debug drawing (well within caps)
        me = obs["self"]
        x, y, z = me["pos"]
        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 5)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        mf.draw.label(f"{mode} hp:{me['hp']:.0f} c:{me['cubes']} {dt_ms:.1f}ms")
        return act
