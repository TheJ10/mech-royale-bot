"""Resource and upgrade optimization: upgrades, weapon loadout, cubes, crates, NPCs, supply drops."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from bot.threat import ThreatTracker
from bot.world import Vec2, Vec3, WorldModel

# Base upgrade value weights (tuned for survival + snowballing)
_UPGRADE_BASE_WEIGHTS: Dict[str, float] = {
    "nanites": 100.0,       # -1.5s regen delay & +60% regen/s -> huge survival multiplier
    "plating": 95.0,        # +15% max HP
    "armor": 92.0,          # -8% damage taken
    "servos": 88.0,         # +8% walk & boost speed (better zone rotation & dodging)
    "gun_rate": 84.0,       # +15% arm gun fire rate
    "gun_damage": 83.0,     # +15% arm gun damage
    "lightning_range": 82.0,
    "laser_charge": 81.0,
    "laser_damage": 80.0,
    "gun_velocity": 76.0,
    "lightning_efficiency": 75.0,
    "gyros": 72.0,          # +30% turn rate -> faster rear-armor protection & aiming
    "sensors": 70.0,        # +15m prox, +25m vision, 0.5x sound noise
    "reactor": 68.0,
    "jets": 66.0,
    "gun_mag": 65.0,
    "capacitor": 60.0,
    "lightning_chain": 58.0,
    "laser_pierce": 55.0,
    "mortar_reload": 54.0,
    "mortar_velocity": 52.0,
    "mortar_splash": 50.0,
}

# Weapon preference ranking for slot 1 (slot 0 kept as gun)
# Avoid holding mortar unless nothing else, due to +60% laser cook-off vulnerability.
_WEAPON_RANK_OPEN: Dict[str, int] = {
    "laser": 4,
    "lightning": 3,
    "gun": 2,
    "mortar": 1,
}
_WEAPON_RANK_CLOSE: Dict[str, int] = {
    "lightning": 4,
    "laser": 3,
    "gun": 2,
    "mortar": 1,
}


class LootObjective:
    __slots__ = ("kind", "pos", "target_id", "score", "standoff_pos")

    def __init__(
        self,
        kind: str,
        pos: Vec3,
        target_id: Optional[int],
        score: float,
        standoff_pos: Optional[Vec2] = None,
    ):
        self.kind = kind  # "cube" | "pickup" | "crate" | "npc" | "drop"
        self.pos = pos
        self.target_id = target_id
        self.score = score
        self.standoff_pos = standoff_pos


class ResourcePlanner:
    """Evaluates upgrades, weapon pickups/swaps, cubes, crates, NPCs, and supply drops."""

    def __init__(self, world: WorldModel, config: Dict[str, Any], pacifist: bool = False):
        self.world = world
        self.config = config
        self.pacifist = pacifist
        self.pickup_radius = float(config.get("loot", {}).get("pickup_radius", 4.0))
        crates_cfg = config.get("crates") or {}
        mech_cfg = config.get("mech") or {}
        cubes_cfg = config.get("cubes") or {}
        self.crate_height = float(crates_cfg.get("height", 2.4))
        self.crate_radius = float(crates_cfg.get("radius", 1.4))
        self.min_pitch = float(mech_cfg.get("min_pitch", -0.7))
        self.max_pitch = float(mech_cfg.get("max_pitch", 1.3))
        self.base_muzzle_h = float(mech_cfg.get("muzzle_height", 7.0))
        self.mech_radius = float(mech_cfg.get("radius", 2.5))
        self.size_per_cube = float(cubes_cfg.get("size_per_cube", 0.04))
        self._failed_crates: Dict[int, float] = {}
        self._crate_standoff_cache: Dict[int, Tuple[float, Optional[Vec2], Vec2]] = {}

    def mark_crate_failed(self, crate_id: int, now: float, cooldown_s: float = 35.0) -> None:
        """Blacklist an unreachable or timed-out crate for `cooldown_s` seconds."""
        self._failed_crates[int(crate_id)] = now + cooldown_s

    def is_crate_blacklisted(self, crate_id: int, now: float) -> bool:
        until = self._failed_crates.get(int(crate_id))
        if until is None:
            return False
        if now >= until:
            del self._failed_crates[int(crate_id)]
            return False
        return True

    def effective_muzzle_height(self, me: Dict[str, Any]) -> float:
        stat_mh = float((me.get("stats") or {}).get("muzzle_height", self.base_muzzle_h))
        scaled_mh = self.base_muzzle_h * (1.0 + self.size_per_cube * int(me.get("cubes", 0)))
        return max(stat_mh, scaled_mh)

    def crate_aim_point(self, crate_pos: Vec3) -> Vec3:
        """Upper-center aim point on crate cylinder (z + 0.6 * height = z + 1.44m)."""
        return (float(crate_pos[0]), float(crate_pos[1]), float(crate_pos[2]) + self.crate_height * 0.6)

    def min_crate_standoff(self, muzzle_z: float, target_z: float) -> float:
        """Minimum horizontal standoff so downward/upward pitch stays within [min_pitch, max_pitch]."""
        dz = target_z - muzzle_z
        if dz < 0.0:
            pitch_d = abs(dz) / math.tan(max(0.05, abs(self.min_pitch) - 0.04))
        elif dz > 0.0:
            pitch_d = dz / math.tan(max(0.05, self.max_pitch - 0.04))
        else:
            pitch_d = 0.0
        return max(self.crate_radius + self.mech_radius + 1.5, pitch_d + 1.5)

    def has_valid_crate_geometry(self, muzzle_pos: Vec3, crate_pos: Vec3) -> bool:
        """True if muzzle_pos has valid pitch, standoff <= 48m, and clear LoS to crate aim point."""
        target_pt = self.crate_aim_point(crate_pos)
        dx = target_pt[0] - muzzle_pos[0]
        dy = target_pt[1] - muzzle_pos[1]
        d = math.hypot(dx, dy)
        min_d = self.min_crate_standoff(muzzle_pos[2], target_pt[2])
        if d < min_d - 0.5 or d > 48.0:
            return False
        pitch = math.atan2(target_pt[2] - muzzle_pos[2], max(1e-4, d))
        if pitch < self.min_pitch + 0.03 or pitch > self.max_pitch - 0.03:
            return False
        if self.world.loaded and not self.world.has_los(muzzle_pos, target_pt, step=2.0):
            return False
        return True

    def find_crate_firing_position(
        self,
        obs: Dict[str, Any],
        crate_pos: Vec3,
        threats: ThreatTracker,
        zone_center: Vec2,
        safe_radius: float,
        crate_id: Optional[int] = None,
    ) -> Optional[Vec2]:
        """Return current pos if already valid, else best walkable/zone-safe standoff point with LoS."""
        me = obs["self"]
        mx, my, mz = float(me["pos"][0]), float(me["pos"][1]), float(me["pos"][2])
        now = float(obs["t"])
        muzzle_h = self.effective_muzzle_height(me)
        cx, cy, cz = float(crate_pos[0]), float(crate_pos[1]), float(crate_pos[2])

        if (
            math.hypot(mx - zone_center[0], my - zone_center[1]) <= safe_radius
            and self.has_valid_crate_geometry((mx, my, mz + muzzle_h), crate_pos)
        ):
            return (mx, my)

        if crate_id is not None:
            cached = self._crate_standoff_cache.get(crate_id)
            if cached is not None and (now - cached[0]) < 0.6:
                c_pt, from_xy = cached[1], cached[2]
                if c_pt is not None:
                    if (
                        math.hypot(c_pt[0] - zone_center[0], c_pt[1] - zone_center[1]) <= safe_radius
                        and threats.danger_at(c_pt[0], c_pt[1], now) <= 3.2
                    ):
                        return c_pt
                elif math.hypot(mx - from_xy[0], my - from_xy[1]) < 6.0:
                    return None

        target_pt = self.crate_aim_point(crate_pos)
        min_d = self.min_crate_standoff(mz + muzzle_h, target_pt[2])
        pref_d = max(min_d + 2.0, 11.5)
        radii = (pref_d, min_d + 1.2)
        base_ang = math.atan2(my - cy, mx - cx)
        k_order = (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6)

        best_pt: Optional[Vec2] = None
        best_score = -1e18
        for r in radii:
            for k in k_order:
                ang = base_ang + (math.tau * k) / 12.0
                sx = cx + math.cos(ang) * r
                sy = cy + math.sin(ang) * r
                if self.world.loaded:
                    if not self.world.in_bounds(sx, sy, margin=6.0):
                        continue
                    gx, gy = self.world.world_to_grid(sx, sy)
                    if not self.world.is_cell_walkable(gx, gy):
                        continue
                    sz = self.world.terrain_height(sx, sy)
                else:
                    sz = mz
                if math.hypot(sx - zone_center[0], sy - zone_center[1]) > safe_radius:
                    continue
                danger = threats.danger_at(sx, sy, now)
                if danger > 3.2:
                    continue
                if not self.has_valid_crate_geometry((sx, sy, sz + muzzle_h), crate_pos):
                    continue
                travel = math.hypot(sx - mx, sy - my)
                d_crate = math.hypot(sx - cx, sy - cy)
                score = -travel - danger * 15.0 - abs(d_crate - pref_d) * 0.4
                if score > best_score:
                    best_score = score
                    best_pt = (sx, sy)
                if travel <= 14.0 and danger < 0.4:
                    break
            if best_pt is not None and best_score > -14.5:
                break
        if crate_id is not None:
            self._crate_standoff_cache[crate_id] = (now, best_pt, (mx, my))
        return best_pt

    def choose_upgrade(self, obs: Dict[str, Any], phase: str = "MID_GAME") -> Optional[int]:
        offer = obs.get("offer")
        if not offer:
            return None
        choices = offer.get("choices") or []
        if not choices:
            return None

        me = obs["self"]
        hp_frac = float(me["hp"]) / max(1.0, float(me["max_hp"]))
        held_kinds = {w["kind"] for w in me["weapons"] if w is not None}
        best_i = 0
        best_score = -1e9
        for i, ch in enumerate(choices):
            uid = str(ch.get("id", ""))
            score = _UPGRADE_BASE_WEIGHTS.get(uid, 40.0)
            if self.pacifist:
                # Pure pacifist prioritizes pure survivability & mobility
                if uid in ("nanites", "plating", "armor", "servos", "jets", "reactor", "sensors", "capacitor"):
                    score += 50.0
                elif uid.startswith(("gun_", "laser_", "lightning_", "mortar_")):
                    score -= 60.0
            else:
                if "lightning" in held_kinds:
                    if uid == "lightning_range":
                        score += 16.0
                    elif uid == "lightning_efficiency":
                        score += 8.0
                    elif uid == "lightning_chain":
                        score += 6.0
                if "laser" in held_kinds:
                    if uid == "laser_charge":
                        score += 14.0
                    elif uid == "laser_damage":
                        score += 12.0
                if phase in ("LATE_GAME", "FINAL_CIRCLE") or hp_frac < 0.72:
                    if uid == "nanites":
                        score += 20.0
                    elif uid in ("plating", "armor"):
                        score += 16.0
                    elif uid == "gyros":
                        score += 10.0
                elif phase == "EARLY_GAME":
                    if uid in ("servos", "gun_rate", "gun_damage", "sensors"):
                        score += 8.0
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def choose_weapon_take(self, obs: Dict[str, Any], phase: str = "MID_GAME") -> Tuple[Optional[int], Optional[int]]:
        """Return (pickup_id, drop_slot) if a desirable weapon is within pickup_radius."""
        if self.pacifist:
            return (None, None)

        me = obs["self"]
        x, y, z = me["pos"]
        weapons = me["weapons"]
        held_kinds = {w["kind"] for w in weapons if w is not None}
        rank_table = _WEAPON_RANK_CLOSE if phase in ("LATE_GAME", "FINAL_CIRCLE") else _WEAPON_RANK_OPEN

        # Never swap/drop a charging laser
        slot1 = weapons[1]
        slot1_charging = bool(slot1 and slot1["kind"] == "laser" and float(slot1.get("charge", 0.0)) > 0.0)

        best_take: Optional[Tuple[int, Optional[int], int]] = None
        for p in obs.get("pickups") or []:
            px, py, pz = p["pos"]
            if abs(pz - z) > 4.0:
                continue
            if math.hypot(px - x, py - y) > self.pickup_radius - 0.1:
                continue
            kind = str(p["kind"])
            if kind in held_kinds:
                continue
            rank = rank_table.get(kind, 0)
            if rank <= 0:
                continue

            # Empty slot 1: take lightning or laser (avoid mortar to dodge +60% cookoff)
            if slot1 is None:
                if kind == "mortar":
                    continue
                if best_take is None or rank > best_take[2]:
                    best_take = (int(p["id"]), None, rank)
            elif not slot1_charging:
                cur_rank = rank_table.get(slot1["kind"], 0)
                if rank > cur_rank:
                    if best_take is None or rank > best_take[2]:
                        best_take = (int(p["id"]), 1, rank)

        if best_take is None:
            return (None, None)
        return (best_take[0], best_take[1])

    def find_best_loot(
        self,
        obs: Dict[str, Any],
        threats: ThreatTracker,
        zone_center: Vec2,
        safe_radius: float,
        max_detour: float = 65.0,
        allow_noisy_farm: bool = True,
        phase_econ_mult: float = 1.0,
        phase: str = "MID_GAME",
        inward_dir: Optional[Vec2] = None,
    ) -> Optional[LootObjective]:
        """Score visible cubes, desirable weapon pickups, crates, NPC drones, and nearby drops."""
        me = obs["self"]
        x, y, z = me["pos"]
        now = float(obs["t"])
        if len(self._failed_crates) > 32:
            self._failed_crates = {k: v for k, v in self._failed_crates.items() if now < v}
        if len(self._crate_standoff_cache) > 32:
            self._crate_standoff_cache = {k: v for k, v in self._crate_standoff_cache.items() if (now - v[0]) < 2.0}
        hp_frac = float(me["hp"]) / max(1.0, float(me["max_hp"]))
        cubes = int(me.get("cubes", 0))
        # Extra reward when 1 cube away from an upgrade threshold (1, 3, 6, 10)
        threshold_bonus = 18.0 if cubes in (0, 2, 5, 9) else 0.0
        held_kinds = {w["kind"] for w in me["weapons"] if w is not None}
        need_secondary = me["weapons"][1] is None or me["weapons"][1]["kind"] == "mortar"
        on_roof = self.world.loaded and (z - self.world.terrain_height(x, y)) > 3.0

        best: Optional[LootObjective] = None

        def consider(
            kind: str,
            pos: Vec3,
            tid: Optional[int],
            base_val: float,
            max_d: float,
            standoff: Optional[Vec2] = None,
            max_danger: float = 3.2,
        ) -> None:
            nonlocal best
            px, py, pz = pos
            if on_roof:
                if abs(pz - z) > 4.5:
                    return
            elif self.world.loaded:
                if (pz - self.world.terrain_height(px, py)) > 2.8:
                    return
            elif abs(pz - z) > 4.5:
                return
            d = math.hypot(px - x, py - y)
            if d > max_d:
                return
            if math.hypot(px - zone_center[0], py - zone_center[1]) > safe_radius:
                return
            danger = threats.danger_at(px, py, now)
            if danger > max_danger:
                return
            align_bonus = 0.0
            if inward_dir is not None and d > 1.0:
                align_bonus = (((px - x) * inward_dir[0] + (py - y) * inward_dir[1]) / d) * 12.0
            score = base_val * phase_econ_mult - d * 1.10 - danger * 18.0 + align_bonus
            if score > 0.0 and (best is None or score > best.score):
                best = LootObjective(kind, pos, tid, score, standoff_pos=standoff)

        # 1. Power cubes (free +4% HP & +4% damage + upgrades!)
        for c in obs.get("cubes") or []:
            consider("cube", tuple(c["pos"]), int(c["id"]), 88.0 + threshold_bonus, max_detour)

        # 2. Desirable secondary weapon (lightning / laser)
        if not self.pacifist and need_secondary:
            w_val = 112.0 if phase == "EARLY_GAME" else 92.0
            w_detour = max(max_detour, 92.0)
            for p in obs.get("pickups") or []:
                k = str(p["kind"])
                if k in ("lightning", "laser") and k not in held_kinds:
                    consider("pickup", tuple(p["pos"]), int(p["id"]), w_val, w_detour)

        # 3. Crates (300 HP -> 1 cube + 50% weapon; cap expensive standoff search to closest 3 valid crates)
        late_block = phase in ("LATE_GAME", "FINAL_CIRCLE") and cubes >= 3
        if allow_noisy_farm and not late_block and not self.pacifist and hp_frac >= 0.55:
            crate_max_d = min(52.0, max_detour)
            crate_cands: List[Tuple[float, int, Vec3]] = []
            for cr in obs.get("crates") or []:
                cid = int(cr["id"])
                if self.is_crate_blacklisted(cid, now):
                    continue
                cpos = (float(cr["pos"][0]), float(cr["pos"][1]), float(cr["pos"][2]))
                if on_roof:
                    if abs(cpos[2] - z) > 4.5:
                        continue
                elif self.world.loaded:
                    if (cpos[2] - self.world.terrain_height(cpos[0], cpos[1])) > 2.8:
                        continue
                elif abs(cpos[2] - z) > 4.5:
                    continue
                d_cr = math.hypot(cpos[0] - x, cpos[1] - y)
                if d_cr > crate_max_d:
                    continue
                if math.hypot(cpos[0] - zone_center[0], cpos[1] - zone_center[1]) > safe_radius:
                    continue
                if threats.danger_at(cpos[0], cpos[1], now) > 3.2:
                    continue
                crate_cands.append((d_cr, cid, cpos))
            crate_cands.sort(key=lambda item: item[0])
            for _, cid, cpos in crate_cands[:3]:
                standoff = self.find_crate_firing_position(
                    obs, cpos, threats, zone_center, safe_radius, crate_id=cid
                )
                if standoff is None:
                    continue
                consider("crate", cpos, cid, 64.0 + threshold_bonus * 0.5, crate_max_d, standoff=standoff)

        # 4. NPC drones (350 HP -> 2 cubes! Dual-purpose: farm when healthy & isolated)
        if allow_noisy_farm and not late_block and not self.pacifist and hp_frac >= 0.72 and not threats.visible_enemies():
            for m in obs.get("mechs") or []:
                if m.get("kind") == "npc":
                    consider("npc", tuple(m["pos"]), int(m["id"]), 70.0 + threshold_bonus, min(52.0, max_detour))

        # 5. Supply drops (3 cubes + full heal + full energy! Hold ring if close & uncontested/low threat)
        if hp_frac >= 0.50:
            for dr in obs.get("drops") or []:
                lands_in = float(dr.get("lands_in", 99.0))
                if lands_in > 18.0:
                    continue
                holder = dr.get("holder")
                dpos = tuple(dr["pos"])
                d_drop = math.hypot(dpos[0] - x, dpos[1] - y)
                # Avoid walking from far away into an already-contested drop with high threat
                if holder is not None and holder != me["id"] and d_drop > 35.0:
                    continue
                if d_drop <= 105.0:
                    hold_or_close = (d_drop <= 12.0 or holder == me["id"])
                    bonus = 160.0 if hold_or_close else 115.0
                    consider("drop", dpos, int(dr["id"]), bonus, 105.0, max_danger=(6.5 if hold_or_close else 3.2))

        return best
