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
_WEAPON_RANK: Dict[str, int] = {
    "lightning": 3,
    "laser": 3,
    "gun": 2,
    "mortar": 1,
}


class LootObjective:
    __slots__ = ("kind", "pos", "target_id", "score")

    def __init__(self, kind: str, pos: Vec3, target_id: Optional[int], score: float):
        self.kind = kind  # "cube" | "pickup" | "crate" | "npc" | "drop"
        self.pos = pos
        self.target_id = target_id
        self.score = score


class ResourcePlanner:
    """Evaluates upgrades, weapon pickups/swaps, cubes, crates, NPCs, and supply drops."""

    def __init__(self, world: WorldModel, config: Dict[str, Any], pacifist: bool = False):
        self.world = world
        self.config = config
        self.pacifist = pacifist
        self.pickup_radius = float(config.get("loot", {}).get("pickup_radius", 4.0))

    def choose_upgrade(self, obs: Dict[str, Any]) -> Optional[int]:
        offer = obs.get("offer")
        if not offer:
            return None
        choices = offer.get("choices") or []
        if not choices:
            return None

        me = obs["self"]
        hp_frac = float(me["hp"]) / max(1.0, float(me["max_hp"]))
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
                if hp_frac < 0.7 and uid in ("nanites", "plating", "armor"):
                    score += 18.0
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def choose_weapon_take(self, obs: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Return (pickup_id, drop_slot) if a desirable weapon is within pickup_radius."""
        if self.pacifist:
            return (None, None)

        me = obs["self"]
        x, y, z = me["pos"]
        weapons = me["weapons"]
        held_kinds = {w["kind"] for w in weapons if w is not None}

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
            rank = _WEAPON_RANK.get(kind, 0)
            if rank <= 0:
                continue

            # Empty slot 1: take lightning, laser, or mortar (if nothing else)
            if slot1 is None:
                # Prefer lightning/laser; still take mortar if empty? Actually avoid mortar to dodge +60% cookoff
                if kind == "mortar":
                    continue
                if best_take is None or rank > best_take[2]:
                    best_take = (int(p["id"]), None, rank)
            elif not slot1_charging:
                cur_rank = _WEAPON_RANK.get(slot1["kind"], 0)
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
        max_detour: float = 55.0,
    ) -> Optional[LootObjective]:
        """Score visible cubes, desirable weapon pickups, crates, NPC drones, and nearby drops."""
        me = obs["self"]
        x, y, z = me["pos"]
        now = float(obs["t"])
        hp_frac = float(me["hp"]) / max(1.0, float(me["max_hp"]))
        held_kinds = {w["kind"] for w in me["weapons"] if w is not None}
        need_secondary = me["weapons"][1] is None or me["weapons"][1]["kind"] == "mortar"

        best: Optional[LootObjective] = None

        def consider(kind: str, pos: Vec3, tid: Optional[int], base_val: float, max_d: float) -> None:
            nonlocal best
            px, py, pz = pos
            # Filter rooftop items using local terrain height rather than our current Z (handles campus slopes!)
            if self.world.loaded:
                if (pz - self.world.terrain_height(px, py)) > 2.8:
                    return
            elif abs(pz - z) > 4.5:
                return
            d = math.hypot(px - x, py - y)
            if d > max_d:
                return
            # Must stay inside safe zone
            if math.hypot(px - zone_center[0], py - zone_center[1]) > safe_radius:
                return
            danger = threats.danger_at(px, py, now)
            if danger > 3.5:
                return
            score = base_val - d * 1.15 - danger * 18.0
            if score > 0.0 and (best is None or score > best.score):
                best = LootObjective(kind, pos, tid, score)

        # 1. Power cubes (free +4% HP & +4% damage + upgrades!)
        for c in obs.get("cubes") or []:
            consider("cube", tuple(c["pos"]), int(c["id"]), 85.0, max_detour)

        # 2. Desirable secondary weapon (lightning / laser)
        if not self.pacifist and need_secondary:
            for p in obs.get("pickups") or []:
                k = str(p["kind"])
                if k in ("lightning", "laser") and k not in held_kinds:
                    consider("pickup", tuple(p["pos"]), int(p["id"]), 78.0, max_detour)

        # 3. Crates (300 HP -> 1 cube + 50% weapon; quick to break with gun)
        if not self.pacifist and hp_frac >= 0.55:
            for cr in obs.get("crates") or []:
                consider("crate", tuple(cr["pos"]), int(cr["id"]), 62.0, min(48.0, max_detour))

        # 4. NPC drones (350 HP -> 2 cubes! Only when very healthy and completely quiet)
        if not self.pacifist and hp_frac >= 0.80 and not threats.visible_enemies() and threats.danger_at(x, y, now) < 1.0:
            for m in obs.get("mechs") or []:
                if m.get("kind") == "npc":
                    consider("npc", tuple(m["pos"]), int(m["id"]), 64.0, min(46.0, max_detour))

        # 5. Supply drops (3 cubes + full heal + full energy! Skip contested/high-danger drops)
        if hp_frac >= 0.55:
            for dr in obs.get("drops") or []:
                lands_in = float(dr.get("lands_in", 99.0))
                if lands_in > 14.0:
                    continue
                holder = dr.get("holder")
                if holder is not None and holder != me["id"]:
                    continue
                dpos = tuple(dr["pos"])
                d_drop = math.hypot(dpos[0] - x, dpos[1] - y)
                if d_drop <= 85.0 and threats.danger_at(dpos[0], dpos[1], now) < 1.6:
                    bonus = 155.0 if (d_drop <= 12.0 or holder == me["id"]) else 110.0
                    consider("drop", dpos, int(dr["id"]), bonus, 85.0)

        return best
