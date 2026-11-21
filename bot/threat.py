"""Threat assessment: persistent enemy tracker, sound integration, rear-armor & projectile dodge."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from bot.world import Vec2, Vec3, WorldModel


class TrackedEnemy:
    __slots__ = (
        "id",
        "pos",
        "vel",
        "aim",
        "hp",
        "max_hp",
        "cubes",
        "weapons",
        "charging",
        "last_seen_t",
        "visible_now",
    )

    def __init__(self, actor: Dict[str, Any], now: float):
        self.id: int = int(actor["id"])
        p = actor["pos"]
        v = actor["vel"]
        a = actor["aim"]
        self.pos: Vec3 = (float(p[0]), float(p[1]), float(p[2]))
        self.vel: Vec3 = (float(v[0]), float(v[1]), float(v[2]))
        self.aim: Vec2 = (float(a[0]), float(a[1]))
        self.hp: float = float(actor.get("hp", 1400.0))
        self.max_hp: float = float(actor.get("max_hp", 1400.0))
        self.cubes: int = int(actor.get("cubes", 0))
        self.weapons: List[str] = list(actor.get("weapons") or [])
        self.charging: bool = bool(actor.get("charging", False))
        self.last_seen_t: float = now
        self.visible_now: bool = True

    def update(self, actor: Dict[str, Any], now: float) -> None:
        p = actor["pos"]
        v = actor["vel"]
        a = actor["aim"]
        self.pos = (float(p[0]), float(p[1]), float(p[2]))
        # Smooth velocity slightly for stable lead prediction
        self.vel = (
            self.vel[0] * 0.3 + float(v[0]) * 0.7,
            self.vel[1] * 0.3 + float(v[1]) * 0.7,
            self.vel[2] * 0.3 + float(v[2]) * 0.7,
        )
        self.aim = (float(a[0]), float(a[1]))
        if "hp" in actor:
            self.hp = float(actor["hp"])
        if "max_hp" in actor:
            self.max_hp = float(actor["max_hp"])
        if "cubes" in actor:
            self.cubes = int(actor["cubes"])
        if "weapons" in actor:
            self.weapons = list(actor["weapons"])
        self.charging = bool(actor.get("charging", False))
        self.last_seen_t = now
        self.visible_now = True


class SoundPing:
    __slots__ = ("kind", "x", "y", "t", "weight")

    def __init__(self, kind: str, x: float, y: float, t: float, weight: float):
        self.kind = kind
        self.x = x
        self.y = y
        self.t = t
        self.weight = weight


class ThreatTracker:
    """Tracks enemy mechs, sound threats, combat timers, rear attacks, and incoming projectiles."""

    MEMORY_TTL = 12.0
    NPC_TTL = 6.0
    SOUND_TTL = 3.0

    def __init__(self, world: WorldModel, config: Dict[str, Any]):
        self.world = world
        self.config = config
        self.enemies: Dict[int, TrackedEnemy] = {}
        self.npcs: Dict[int, Dict[str, Any]] = {}
        self.sounds: List[SoundPing] = []
        self.last_damaged_t: float = -100.0
        self.last_npc_damaged_t: float = -100.0
        self.last_dealt_damage_t: float = -100.0
        self.last_hit_by: Dict[int, float] = {}
        self.rear_alert_bearing: Optional[float] = None
        self.rear_alert_until_t: float = -1.0
        self.laser_short_until_t: float = -1.0
        self.damage_taken_recent: float = 0.0
        self._damage_events: List[Tuple[float, float]] = []
        self._mortar_gravity: float = float(
            (config.get("weapons") or {}).get("mortar", {}).get("gravity", 22.0)
        )
        self._short_s: float = float(
            (config.get("weapons") or {}).get("lightning", {}).get("short_s", 2.0)
        )

    def update(self, obs: Dict[str, Any]) -> None:
        now = float(obs["t"])
        me = obs["self"]
        mx, my, mz = me["pos"]

        # Mark all previously tracked enemies and NPCs as not currently visible
        for e in self.enemies.values():
            e.visible_now = False
        for n in self.npcs.values():
            n["visible_now"] = False

        # Update visible mechs and NPC drones
        for m in obs.get("mechs") or []:
            kind = m.get("kind")
            if kind == "npc":
                nid = int(m["id"])
                p = m["pos"]
                self.npcs[nid] = {
                    "id": nid,
                    "pos": (float(p[0]), float(p[1]), float(p[2])),
                    "hp": float(m.get("hp", 350.0)),
                    "last_seen_t": now,
                    "visible_now": True,
                }
                continue
            if kind != "mech":
                continue
            eid = int(m["id"])
            if eid in self.enemies:
                self.enemies[eid].update(m, now)
            else:
                self.enemies[eid] = TrackedEnemy(m, now)

        # Prune stale memories
        stale = [eid for eid, e in self.enemies.items() if now - e.last_seen_t > self.MEMORY_TTL]
        for eid in stale:
            del self.enemies[eid]
        stale_npc = [nid for nid, n in self.npcs.items() if now - n["last_seen_t"] > self.NPC_TTL]
        for nid in stale_npc:
            del self.npcs[nid]

        # Process events
        for ev in obs.get("events") or []:
            etype = ev.get("type")
            if etype == "damaged":
                self.last_damaged_t = now
                if ev.get("weapon") == "npc":
                    self.last_npc_damaged_t = now
                amt = float(ev.get("amount", 0.0))
                self._damage_events.append((now, amt))
                attacker = ev.get("by")
                if attacker is not None:
                    self.last_hit_by[int(attacker)] = now
                bearing = ev.get("bearing")
                if bearing is not None and ev.get("weapon") != "zone":
                    # Face attacker to protect rear armor (1.5x penalty!)
                    if ev.get("rear") or attacker is None or int(attacker) not in self.enemies or not self.enemies[int(attacker)].visible_now:
                        self.rear_alert_bearing = float(bearing)
                        self.rear_alert_until_t = now + 2.2
            elif etype == "hit":
                self.last_dealt_damage_t = now
            elif etype == "effect":
                if ev.get("effect") == "short":
                    self.laser_short_until_t = now + self._short_s

        # Rolling 3s damage taken
        self._damage_events = [(t, a) for t, a in self._damage_events if now - t <= 3.0]
        self.damage_taken_recent = sum(a for _, a in self._damage_events)

        # Process sounds into approximate world coordinates
        self.sounds = [s for s in self.sounds if now - s.t <= self.SOUND_TTL]
        dist_map = {"near": 35.0, "mid": 110.0, "far": 210.0}
        weight_map = {
            "laser_charge": 3.0,
            "lightning": 2.8,
            "laser": 2.4,
            "mortar": 2.2,
            "explosion": 2.0,
            "gun": 1.8,
            "boost": 1.3,
            "jet": 1.2,
            "step": 1.0,
        }
        for snd in obs.get("sounds") or []:
            kind = str(snd.get("kind", ""))
            w = weight_map.get(kind, 0.0)
            if w <= 0.0:
                continue
            d = dist_map.get(str(snd.get("dist", "far")), 180.0)
            b = float(snd.get("bearing", 0.0))
            sx = mx + math.cos(b) * d
            sy = my + math.sin(b) * d
            self.sounds.append(SoundPing(kind, sx, sy, now - float(snd.get("age", 0.0)), w))

        # Cap sound pings to prevent O(N) blowup in A* danger_at during busy 16-mech fights
        if len(self.sounds) > 24:
            self.sounds.sort(key=lambda s: s.weight * max(0.1, 1.0 - (now - s.t) / self.SOUND_TTL), reverse=True)
            del self.sounds[24:]

        # Expire rear alert
        if now > self.rear_alert_until_t:
            self.rear_alert_bearing = None

    def time_since_combat(self, now: float) -> float:
        return now - max(self.last_damaged_t, self.last_dealt_damage_t)

    def visible_enemies(self) -> List[TrackedEnemy]:
        return [e for e in self.enemies.values() if e.visible_now]

    def recent_attackers(self, now: float, window: float = 5.0) -> set[int]:
        return {eid for eid, t in self.last_hit_by.items() if now - t <= window}

    def visible_npcs(self) -> List[Dict[str, Any]]:
        return [n for n in self.npcs.values() if n["visible_now"]]

    def snapshot_sources(self, now: float, include_npcs: bool = False, npc_weight: float = 5.0) -> List[Tuple[float, float, float, float, float]]:
        """Compact list of (x, y, radius, r_sq, weight) for ultra-fast inline A* danger evaluation."""
        sources: List[Tuple[float, float, float, float, float]] = []
        for e in self.enemies.values():
            age = max(0.0, now - e.last_seen_t)
            decay = 1.0 if e.visible_now else max(0.15, 1.0 - age / self.MEMORY_TTL)
            mult = 1.0 + 0.05 * min(e.cubes, 10)
            if e.charging:
                mult *= 1.4
            sources.append((e.pos[0], e.pos[1], 160.0, 25600.0, decay * mult * 6.0))
            if "lightning" in e.weapons:
                sources.append((e.pos[0], e.pos[1], 55.0, 3025.0, decay * 4.5))
        if include_npcs:
            for n in self.npcs.values():
                age = max(0.0, now - n["last_seen_t"])
                decay = 1.0 if n["visible_now"] else max(0.2, 1.0 - age / self.NPC_TTL)
                sources.append((n["pos"][0], n["pos"][1], 52.0, 2704.0, decay * npc_weight))
        for s in self.sounds:
            age = max(0.0, now - s.t)
            decay = max(0.0, 1.0 - age / self.SOUND_TTL)
            if decay > 0.05:
                sources.append((s.x, s.y, 120.0, 14400.0, decay * s.weight * 1.2))
        return sources

    def danger_at(self, x: float, y: float, now: float, include_npcs: bool = False, npc_weight: float = 4.5) -> float:
        """Non-negative threat penalty for A* routing and safe-position scoring."""
        score = 0.0
        for e in self.enemies.values():
            dx = e.pos[0] - x
            dy = e.pos[1] - y
            if dx > 160.0 or dx < -160.0 or dy > 160.0 or dy < -160.0:
                continue
            d2 = dx * dx + dy * dy
            if d2 < 25600.0:
                d = math.sqrt(d2)
                age = max(0.0, now - e.last_seen_t)
                decay = 1.0 if e.visible_now else max(0.15, 1.0 - age / self.MEMORY_TTL)
                prox = (160.0 - d) / 160.0
                mult = 1.0 + 0.05 * min(e.cubes, 10)
                if "lightning" in e.weapons and d < 55.0:
                    mult *= 1.5
                if e.charging and d < 180.0:
                    mult *= 1.4
                score += decay * mult * (prox * prox) * 6.0

        if include_npcs:
            for n in self.npcs.values():
                dx = n["pos"][0] - x
                dy = n["pos"][1] - y
                if dx > 52.0 or dx < -52.0 or dy > 52.0 or dy < -52.0:
                    continue
                d2 = dx * dx + dy * dy
                if d2 < 2704.0:
                    d = math.sqrt(d2)
                    age = max(0.0, now - n["last_seen_t"])
                    decay = 1.0 if n["visible_now"] else max(0.2, 1.0 - age / self.NPC_TTL)
                    prox = (52.0 - d) / 52.0
                    score += decay * npc_weight * (prox * prox)

        for s in self.sounds:
            dx = s.x - x
            dy = s.y - y
            if dx > 120.0 or dx < -120.0 or dy > 120.0 or dy < -120.0:
                continue
            d2 = dx * dx + dy * dy
            if d2 < 14400.0:
                d = math.sqrt(d2)
                age = max(0.0, now - s.t)
                decay = max(0.0, 1.0 - age / self.SOUND_TTL)
                prox = (120.0 - d) / 120.0
                score += decay * s.weight * (prox * prox) * 1.2
        return score

    def compute_dodge_vector(
        self,
        obs: Dict[str, Any],
        zone_center: Vec2,
    ) -> Optional[Vec2]:
        """Compute perpendicular evasion vector for incoming projectiles, mortars, or charging lasers."""
        me = obs["self"]
        mx, my, mz = me["pos"]
        my_center_z = mz + 4.5
        dodge_x = 0.0
        dodge_y = 0.0
        urgency = 0.0

        # 1. Incoming projectiles (gun bullets, mortar shells, NPC bullets)
        for proj in obs.get("projectiles") or []:
            px, py, pz = proj["pos"]
            vx, vy, vz = proj["vel"]
            v2 = vx * vx + vy * vy
            if v2 < 25.0:
                continue
            rx = mx - px
            ry = my - py
            # Time to closest approach in XY
            t_ca = (rx * vx + ry * vy) / v2
            if t_ca < 0.0 or t_ca > 1.2:
                continue
            cx = px + vx * t_ca
            cy = py + vy * t_ca
            miss_xy = math.hypot(mx - cx, my - cy)
            kind = proj.get("kind", "gun")
            danger_rad = 10.0 if kind == "mortar" else 5.0
            if miss_xy > danger_rad:
                continue
            # Check approximate Z overlap (ballistic arc for mortar, linear for bullets)
            if kind == "mortar":
                cz = pz + vz * t_ca - 0.5 * self._mortar_gravity * t_ca * t_ca
                if abs(cz - my_center_z) > 12.0:
                    continue
            else:
                cz = pz + vz * t_ca
                if abs(cz - my_center_z) > 6.5:
                    continue
            # Perpendicular to projectile velocity, away from closest-approach point
            speed = math.sqrt(v2)
            perp1_x, perp1_y = -vy / speed, vx / speed
            perp2_x, perp2_y = vy / speed, -vx / speed
            # Pick side that moves away from impact point and prefers staying toward zone center
            dot1 = (mx - cx) * perp1_x + (my - cy) * perp1_y
            if abs(dot1) < 0.5:
                # Choose side closer to zone center
                zc_dot1 = (zone_center[0] - mx) * perp1_x + (zone_center[1] - my) * perp1_y
                px_dir, py_dir = (perp1_x, perp1_y) if zc_dot1 >= 0 else (perp2_x, perp2_y)
            elif dot1 > 0:
                px_dir, py_dir = perp1_x, perp1_y
            else:
                px_dir, py_dir = perp2_x, perp2_y

            w = (2.5 if kind == "mortar" else 1.4) * (1.0 - t_ca / 1.3)
            dodge_x += px_dir * w
            dodge_y += py_dir * w
            urgency = max(urgency, w)

        # 2. Enemy charging a laser aimed near us
        for e in self.enemies.values():
            if not e.visible_now or not e.charging:
                continue
            dx = mx - e.pos[0]
            dy = my - e.pos[1]
            dist = math.hypot(dx, dy)
            if dist > 260.0 or dist < 1.0:
                continue
            yaw_to_me = math.atan2(dy, dx)
            err = abs(math.remainder(yaw_to_me - e.aim[0], math.tau))
            if err < 0.28:  # ~16 degrees
                # Perpendicular to enemy line of fire
                perp1_x, perp1_y = -dy / dist, dx / dist
                zc_dot = (zone_center[0] - mx) * perp1_x + (zone_center[1] - my) * perp1_y
                if zc_dot < 0:
                    perp1_x, perp1_y = -perp1_x, -perp1_y
                dodge_x += perp1_x * 2.2
                dodge_y += perp1_y * 2.2
                urgency = max(urgency, 2.2)

        if urgency <= 0.1:
            return None
        norm = math.hypot(dodge_x, dodge_y)
        if norm < 1e-6:
            return None
        return (dodge_x / norm, dodge_y / norm)
