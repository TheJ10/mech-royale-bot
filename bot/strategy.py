"""Unified Battle Intelligence central utility-based decision controller (UBI-v4)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bot.combat import CombatPlan, CombatPlanner, angle_diff
from bot.nav import Navigator
from bot.resources import LootObjective, ResourcePlanner
from bot.threat import ThreatTracker, TrackedEnemy
from bot.world import Vec2, Vec3, WorldModel


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    bot_name: str
    pacifist: bool = False
    retreat_hp_frac: float = 0.44
    outnumbered_hp_frac: float = 0.70
    engage_range: float = 160.0
    danger_nav_weight: float = 2.2
    loot_detour_m: float = 65.0
    quiet_ring_frac: float = 0.58
    silent_when_hidden: bool = False


PROFILES: Dict[str, StrategyProfile] = {
    "unified": StrategyProfile(
        name="unified",
        bot_name="ubi-v4",
        pacifist=False,
        retreat_hp_frac=0.44,
        outnumbered_hp_frac=0.70,
        engage_range=160.0,
        danger_nav_weight=2.2,
        loot_detour_m=65.0,
        quiet_ring_frac=0.58,
        silent_when_hidden=False,
    ),
    "tactical_survivor": StrategyProfile(
        name="tactical_survivor",
        bot_name="ubi-v4",
        pacifist=False,
        retreat_hp_frac=0.44,
        outnumbered_hp_frac=0.70,
        engage_range=160.0,
        danger_nav_weight=2.2,
        loot_detour_m=65.0,
        quiet_ring_frac=0.58,
        silent_when_hidden=False,
    ),
    "pure_pacifist": StrategyProfile(
        name="pure_pacifist",
        bot_name="pacifist-v2",
        pacifist=True,
        retreat_hp_frac=0.99,
        outnumbered_hp_frac=0.99,
        engage_range=0.0,
        danger_nav_weight=4.2,
        loot_detour_m=45.0,
        quiet_ring_frac=0.68,
        silent_when_hidden=True,
    ),
    "opportunistic_hunter": StrategyProfile(
        name="opportunistic_hunter",
        bot_name="hunter-v2.1",
        pacifist=False,
        retreat_hp_frac=0.42,
        outnumbered_hp_frac=0.68,
        engage_range=175.0,
        danger_nav_weight=1.3,
        loot_detour_m=68.0,
        quiet_ring_frac=0.50,
        silent_when_hidden=False,
    ),
}


class StrategyController:
    """Central utility-based decision engine coordinating Combat, Survival, and Economy systems."""

    HYSTERESIS_BONUS = 12.0

    def __init__(
        self,
        world: WorldModel,
        nav: Navigator,
        threats: ThreatTracker,
        combat: CombatPlanner,
        resources: ResourcePlanner,
        profile: StrategyProfile,
        seed: int = 0,
    ):
        self.world = world
        self.nav = nav
        self.threats = threats
        self.combat = combat
        self.resources = resources
        self.profile = profile
        self._seed: int = int(seed)
        self._angle_bias: float = ((self._seed % 360) / 360.0) * math.tau
        self._initial_yaw: Optional[float] = None
        self.mode: str = "SAFE_ROTATE"
        self.phase: str = "EARLY_GAME"
        self.last_decision_meta: Dict[str, Any] = {}
        self._anchor: Optional[Vec2] = None
        self._anchor_t: float = -100.0
        self._cover_pt: Optional[Vec2] = None
        self._cover_t: float = -100.0
        self._retreat_goal: Optional[Vec2] = None
        self._retreat_goal_t: float = -100.0
        self._crate_id: Optional[int] = None
        self._crate_start_t: float = 0.0
        self._crate_last_hp: float = 0.0
        self._crate_last_progress_t: float = 0.0
        self._crate_start_hp_me: float = 0.0
        self._crate_fire_ticks: int = 0
        self._strafe_dir: float = 1.0
        self._last_strafe_flip_t: float = 0.0

    def _choose_retreat_goal(
        self,
        obs: Dict[str, Any],
        threat_ref: Vec3,
        threat_vel: Vec3,
        play_center: Vec2,
        safe_play_r: float,
    ) -> Tuple[Vec2, bool, float]:
        """Select zone-safe retreat goal maximizing separation gain and LoS breaking without oscillation."""
        me = obs["self"]
        x, y, z = float(me["pos"][0]), float(me["pos"][1]), float(me["pos"][2])
        now = float(obs["t"])
        tx, ty, tz = float(threat_ref[0]), float(threat_ref[1]), float(threat_ref[2])
        e_pred_x = tx + float(threat_vel[0]) * 1.5
        e_pred_y = ty + float(threat_vel[1]) * 1.5
        cur_d_threat = math.hypot(x - e_pred_x, y - e_pred_y)
        t_eye = (tx, ty, tz + 7.5)
        vis_npcs = self.threats.visible_npcs()

        pool: List[Tuple[Vec2, int, bool, bool]] = []  # (pt, deg, prechecked_los, is_current)
        if self.world.loaded:
            for pt, deg in self.world.find_cover_candidates(
                (x, y, z), threat_ref, play_center, max(8.0, safe_play_r), search_radius=62.0, max_results=6
            ):
                pool.append((pt, deg, True, False))

        away_yaw = math.atan2(y - e_pred_y, x - e_pred_x)
        step_r = min(44.0, max(10.0, safe_play_r * 0.85))
        for off in (0.0, 0.45, -0.45, 0.9, -0.9):
            ang = away_yaw + off
            cx = x + math.cos(ang) * step_r
            cy = y + math.sin(ang) * step_r
            d_pc = math.hypot(cx - play_center[0], cy - play_center[1])
            if d_pc > safe_play_r and d_pc > 1e-3:
                scale = max(0.0, safe_play_r - 2.0) / d_pc
                cx = play_center[0] + (cx - play_center[0]) * scale
                cy = play_center[1] + (cy - play_center[1]) * scale
            cx, cy = self.world.clamp_to_bounds(cx, cy, margin=6.0)
            deg = 6
            if self.world.loaded:
                gx, gy = self.world.world_to_grid(cx, cy)
                if not self.world.is_cell_walkable(gx, gy):
                    gx, gy = self.world.nearest_walkable_cell(cx, cy, max_ring=4)
                    if not self.world.is_cell_walkable(gx, gy):
                        continue
                    cx, cy = self.world.grid_to_world(gx, gy)
                if self.world.escape_deg:
                    deg = self.world.escape_deg[gy * self.world.nav_nx + gx]
            pool.append(((cx, cy), deg, False, False))

        if self._retreat_goal is not None:
            rx, ry = self._retreat_goal
            if math.hypot(rx - x, ry - y) > 5.0 and math.hypot(rx - play_center[0], ry - play_center[1]) <= max(4.0, safe_play_r):
                pool.append(((rx, ry), 5, False, True))

        best_pt: Vec2 = play_center
        best_los = False
        best_sep = 0.0
        best_score = -1e18

        for (wx, wy), deg, pre_los, is_cur in pool:
            if math.hypot(wx - play_center[0], wy - play_center[1]) > max(4.0, safe_play_r) + 1.0:
                continue
            d_me = math.hypot(wx - x, wy - y)
            if d_me < 2.5:
                continue
            d_threat = math.hypot(wx - e_pred_x, wy - e_pred_y)
            sep_gain = d_threat - cur_d_threat

            if pre_los:
                breaks_los = True
            elif self.world.loaded:
                wz = self.world.terrain_height(wx, wy)
                breaks_los = not self.world.has_los(t_eye, (wx, wy, wz + 5.0), step=3.5)
            else:
                breaks_los = False

            danger = self.threats.danger_at(wx, wy, now, include_npcs=True)
            npc_pen = 35.0 if any(math.hypot(wx - n["pos"][0], wy - n["pos"][1]) < 48.0 for n in vis_npcs) else 0.0
            inward_margin = max(0.0, safe_play_r - math.hypot(wx - play_center[0], wy - play_center[1]))
            zone_bonus = 12.0 if inward_margin >= 8.0 else 0.0
            corner_bonus = 12.0 if 4 <= deg <= 6 else (-30.0 if deg <= 2 else 0.0)
            commit_bonus = 18.0 if is_cur else 0.0

            score = (
                sep_gain * 1.35
                - max(0.0, -sep_gain) * 2.2
                + (45.0 if breaks_los else 0.0)
                + corner_bonus
                - danger * 26.0
                - npc_pen
                + zone_bonus
                - d_me * 0.40
                + commit_bonus
            )
            if score > best_score:
                best_score = score
                best_pt = (wx, wy)
                best_los = breaks_los
                best_sep = sep_gain

        self._retreat_goal = best_pt
        self._retreat_goal_t = now
        return (best_pt, best_los, best_sep)

    def classify_phase(self, now: float, alive: int, next_radius: float) -> str:
        if next_radius <= 35.0 or alive <= 2:
            return "FINAL_CIRCLE"
        if alive <= 5 or next_radius <= 90.0:
            return "LATE_GAME"
        if now < 90.0 and alive > 10 and next_radius >= 220.0:
            return "EARLY_GAME"
        return "MID_GAME"

    def _pick_early_anchor(
        self,
        obs: Dict[str, Any],
        next_center: Vec2,
        next_radius: float,
        play_center: Vec2,
        safe_play_r: float,
        avoid_npcs: bool = False,
    ) -> Vec2:
        """Spawn-adaptive early-game anchor favoring nearby cover, elevation, loot clusters, and low exposure."""
        now = float(obs["t"])
        me = obs["self"]
        mx, my, _ = float(me["pos"][0]), float(me["pos"][1]), float(me["pos"][2])
        zcx, zcy = play_center
        ncx, ncy = next_center

        if self._anchor is not None and (now - self._anchor_t) < 4.0:
            ax, ay = self._anchor
            if math.hypot(ax - zcx, ay - zcy) <= max(6.0, safe_play_r):
                if self.threats.danger_at(ax, ay, now, include_npcs=avoid_npcs) < 2.4:
                    return self._anchor

        held_kinds = {w["kind"] for w in me["weapons"] if w is not None}
        need_sec = me["weapons"][1] is None or me["weapons"][1]["kind"] == "mortar"
        loot_pts: List[Tuple[float, float]] = [
            (float(c["pos"][0]), float(c["pos"][1])) for c in (obs.get("cubes") or [])
        ]
        if need_sec:
            for p in obs.get("pickups") or []:
                pk = str(p.get("kind", ""))
                if pk in ("lightning", "laser") and pk not in held_kinds:
                    loot_pts.append((float(p["pos"][0]), float(p["pos"][1])))

        best_pt: Vec2 = self.world.clamp_to_bounds(ncx, ncy, margin=12.0)
        best_score = -1e18

        for r in (32.0, 68.0, 102.0):
            for k in range(12):
                ang = self._angle_bias + (math.tau * k) / 12.0
                cx = mx + math.cos(ang) * r
                cy = my + math.sin(ang) * r
                if not self.world.in_bounds(cx, cy, margin=12.0):
                    continue
                gx, gy = self.world.world_to_grid(cx, cy)
                if self.world.loaded and not self.world.is_cell_walkable(gx, gy):
                    continue
                wx, wy = self.world.grid_to_world(gx, gy) if self.world.loaded else (cx, cy)
                if math.hypot(wx - zcx, wy - zcy) > max(8.0, safe_play_r):
                    continue

                danger = self.threats.danger_at(wx, wy, now, include_npcs=avoid_npcs)
                mid_danger = self.threats.danger_at(0.5 * (mx + wx), 0.5 * (my + wy), now, include_npcs=avoid_npcs)
                idx = gy * self.world.nav_nx + gx
                has_cover = bool(self.world.loaded and self.world.cover_adj[idx])
                deg = self.world.escape_deg[idx] if (self.world.loaded and self.world.escape_deg) else 6
                corner_bonus = 34.0 if (has_cover and 4 <= deg <= 6) else (16.0 if has_cover else 0.0)
                if deg <= 2:
                    corner_bonus -= 38.0
                elif self.world.loaded and self.world.is_narrow_choke(gx, gy):
                    corner_bonus -= 24.0
                elev_bonus = self.world.cell_elevation_advantage(gx, gy) * 3.5 if self.world.loaded else 0.0
                edge_dist = min(wx - self.world.min_x, self.world.max_x - wx, wy - self.world.min_y, self.world.max_y - wy)
                edge_pen = 20.0 if edge_dist < 25.0 else 0.0
                loot_bonus = 18.0 if any(math.hypot(wx - lx, wy - ly) <= 36.0 for lx, ly in loot_pts) else 0.0
                dist_next_c = math.hypot(wx - ncx, wy - ncy)

                score = (
                    corner_bonus
                    + elev_bonus
                    + loot_bonus
                    - danger * 28.0
                    - mid_danger * 14.0
                    - dist_next_c * 0.06
                    - edge_pen
                )
                if score > best_score:
                    best_score = score
                    best_pt = (wx, wy)

        self._anchor = best_pt
        self._anchor_t = now
        return best_pt

    def _pick_quiet_zone_anchor(
        self,
        obs: Dict[str, Any],
        zone_center: Vec2,
        next_radius: float,
        avoid_npcs: bool = False,
        phase: str = "MID_GAME",
    ) -> Vec2:
        """Pick a cover-adjacent, dead-end-free, low-threat walkable position inside the safe circle."""
        now = float(obs["t"])
        me = obs["self"]
        mx, my, _ = me["pos"]
        zcx, zcy = zone_center

        if self._anchor is not None and (now - self._anchor_t) < 4.0:
            ax, ay = self._anchor
            if math.hypot(ax - zcx, ay - zcy) <= max(5.0, next_radius - 8.0):
                if self.threats.danger_at(ax, ay, now, include_npcs=avoid_npcs) < 2.6:
                    return self._anchor

        if next_radius <= 24.0:
            best_pt: Vec2 = self.world.clamp_to_bounds(zcx, zcy, margin=8.0)
            if self.world.loaded and next_radius > 8.0:
                best_s = -1e18
                sr = max(4.0, next_radius * 0.55)
                for k in range(8):
                    ang = self._angle_bias + (math.tau * k) / 8.0
                    cx = zcx + math.cos(ang) * sr
                    cy = zcy + math.sin(ang) * sr
                    if not self.world.in_bounds(cx, cy, margin=8.0):
                        continue
                    gx, gy = self.world.world_to_grid(cx, cy)
                    if not self.world.is_cell_walkable(gx, gy):
                        continue
                    wx, wy = self.world.grid_to_world(gx, gy)
                    idx = gy * self.world.nav_nx + gx
                    has_cover = bool(self.world.cover_adj[idx])
                    deg = self.world.escape_deg[idx] if self.world.escape_deg else 6
                    danger = self.threats.danger_at(wx, wy, now, include_npcs=avoid_npcs)
                    s = (18.0 if has_cover else 0.0) + (-20.0 if deg <= 2 else 0.0) - danger * 20.0
                    if s > best_s:
                        best_s = s
                        best_pt = (wx, wy)
            self._anchor = best_pt
            self._anchor_t = now
            return best_pt

        if phase == "FINAL_CIRCLE":
            ring_mults = (0.30, 0.0)
        elif phase == "LATE_GAME":
            ring_mults = (0.48, 0.24)
        else:
            ring_mults = (self.profile.quiet_ring_frac, 0.32)

        best_pt = self.world.clamp_to_bounds(zcx, zcy, margin=8.0)
        best_score = -1e18

        for idx_m, ring_mult in enumerate(ring_mults):
            r = max(4.0, min(next_radius - 12.0, next_radius * ring_mult)) if ring_mult > 0.0 else 0.0
            count = 1 if r == 0.0 else (16 if idx_m == 0 else 8)
            for k in range(count):
                ang = self._angle_bias + (math.tau * k) / count
                cx = zcx + math.cos(ang) * r
                cy = zcy + math.sin(ang) * r
                if not self.world.in_bounds(cx, cy, margin=12.0):
                    continue
                gx, gy = self.world.world_to_grid(cx, cy)
                if self.world.loaded and not self.world.is_cell_walkable(gx, gy):
                    continue
                wx, wy = self.world.grid_to_world(gx, gy) if self.world.loaded else (cx, cy)
                if math.hypot(wx - zcx, wy - zcy) > max(6.0, next_radius - 10.0):
                    continue
                danger = self.threats.danger_at(wx, wy, now, include_npcs=avoid_npcs)
                mid_danger = self.threats.danger_at(0.5 * (mx + wx), 0.5 * (my + wy), now, include_npcs=avoid_npcs)
                dist_me = math.hypot(wx - mx, wy - my)
                idx = gy * self.world.nav_nx + gx
                has_cover = bool(self.world.loaded and self.world.cover_adj[idx])
                deg = self.world.escape_deg[idx] if (self.world.loaded and self.world.escape_deg) else 6
                corner_bonus = 32.0 if (has_cover and 4 <= deg <= 6) else (16.0 if has_cover else 0.0)
                if deg <= 2:
                    corner_bonus -= 35.0
                elif self.world.loaded and self.world.is_narrow_choke(gx, gy):
                    corner_bonus -= 22.0
                elev_bonus = self.world.cell_elevation_advantage(gx, gy) * 3.2 if self.world.loaded else 0.0
                edge_dist = min(wx - self.world.min_x, self.world.max_x - wx, wy - self.world.min_y, self.world.max_y - wy)
                edge_pen = 18.0 if edge_dist < 25.0 else 0.0
                score = (
                    corner_bonus
                    + elev_bonus
                    - danger * 24.0
                    - mid_danger * 14.0
                    - dist_me * 0.14
                    - edge_pen
                )
                if score > best_score:
                    best_score = score
                    best_pt = (wx, wy)

        self._anchor = best_pt
        self._anchor_t = now
        return best_pt

    def decide(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Vec2]:
        """Evaluate candidate objectives via phase-adapted utility scoring and emit coordinated action."""
        me = obs["self"]
        x, y, z = me["pos"]
        pos3: Vec3 = (float(x), float(y), float(z))
        vel3: Vec3 = (float(me["vel"][0]), float(me["vel"][1]), float(me["vel"][2]))
        now = float(obs["t"])
        tick = int(obs["tick"])
        alive = int(obs.get("alive", 16))
        hp = float(me["hp"])
        max_hp = max(1.0, float(me["max_hp"]))
        hp_frac = hp / max_hp
        energy = float(me["energy"])
        cubes = int(me.get("cubes", 0))
        grounded = bool(me["grounded"])
        in_zone = bool(me["in_zone"])
        walk_speed = max(1.0, float(me["stats"].get("walk_speed", 11.7)))
        if self._initial_yaw is None:
            self._initial_yaw = float(me["aim"][0])

        zone = obs["zone"]
        cur_center: Vec2 = (float(zone["center"][0]), float(zone["center"][1]))
        cur_radius = float(zone["radius"])
        next_center: Vec2 = (float(zone["next_center"][0]), float(zone["next_center"][1]))
        next_radius = float(zone["next_radius"])
        shrink_start = float(zone["shrink_start"])
        shrink_end = float(zone["shrink_end"])

        if now <= shrink_start or shrink_end <= shrink_start:
            active_cx, active_cy, active_r = cur_center[0], cur_center[1], cur_radius
        elif now >= shrink_end:
            active_cx, active_cy, active_r = next_center[0], next_center[1], next_radius
        else:
            alpha = (now - shrink_start) / (shrink_end - shrink_start)
            active_cx = cur_center[0] + (next_center[0] - cur_center[0]) * alpha
            active_cy = cur_center[1] + (next_center[1] - cur_center[1]) * alpha
            active_r = cur_radius + (next_radius - cur_radius) * alpha
        active_center: Vec2 = (active_cx, active_cy)

        self.phase = self.classify_phase(now, alive, next_radius)
        safe_radius = max(0.0, next_radius - 14.0)
        dist_to_next_safe = max(0.0, math.hypot(x - next_center[0], y - next_center[1]) - safe_radius)
        dist_to_active_edge = active_r - math.hypot(x - active_cx, y - active_cy)
        walk_s = dist_to_next_safe / walk_speed
        sec_until_shrink = shrink_start - now

        # Strategic Zone Control: urgent deadline vs proactive pre-shrink rotation
        if next_radius <= 10.0:
            edge_margin = min(10.0, active_r * 0.35)
            zone_urgent = (not in_zone) or (active_r > 3.5 and dist_to_active_edge < edge_margin)
            zone_advance = False
        else:
            zone_urgent = (not in_zone) or (dist_to_next_safe > 0.0 and sec_until_shrink <= walk_s * 1.25 + 10.0)
            zone_advance = (not zone_urgent) and dist_to_next_safe > 14.0 and sec_until_shrink <= walk_s * 1.55 + 24.0

        inward_dir: Optional[Vec2] = None
        if zone_advance:
            play_center = next_center
            play_radius = next_radius
            safe_play_r = max(0.0, next_radius - min(12.0, next_radius * 0.25))
            dx_in = next_center[0] - x
            dy_in = next_center[1] - y
            d_in = math.hypot(dx_in, dy_in)
            if d_in > 1.0:
                inward_dir = (dx_in / d_in, dy_in / d_in)
        else:
            use_active_circle = (sec_until_shrink > walk_s + 15.0) or (next_radius <= 10.0)
            play_center = active_center if use_active_circle else next_center
            play_radius = active_r if use_active_circle else next_radius
            safe_play_r = max(0.0, play_radius - min(10.0, play_radius * 0.30))

        laser_shorted = now < self.threats.laser_short_until_t
        charging_laser = any(w and w["kind"] == "laser" and float(w.get("charge", 0.0)) > 0.0 for w in me["weapons"])
        laser_ready = (not laser_shorted) and any(
            w and w["kind"] == "laser" and float(w.get("cooldown", 0.0)) <= 0.05 for w in me["weapons"]
        )

        vis_enemies = self.threats.visible_enemies()
        nearest_enemy: Optional[TrackedEnemy] = None
        nearest_dist = 1e9
        for e in vis_enemies:
            d = math.hypot(e.pos[0] - x, e.pos[1] - y)
            if d < nearest_dist:
                nearest_dist = d
                nearest_enemy = e

        dodge_vec = self.threats.compute_dodge_vector(obs, next_center)

        # Phase-adapted thresholds
        retreat_thresh = (
            0.30
            if (self.phase == "FINAL_CIRCLE" and alive <= 2)
            else (0.36 if self.phase == "FINAL_CIRCLE" else self.profile.retreat_hp_frac)
        )
        outnum_thresh = self.profile.outnumbered_hp_frac
        econ_mult = 1.28 if self.phase == "EARLY_GAME" else (1.05 if self.phase == "MID_GAME" else 0.78)

        close_enemies = [e for e in vis_enemies if math.hypot(e.pos[0] - x, e.pos[1] - y) < 85.0]
        exposed_los_count = (
            self.threats.multi_enemy_exposure_count(pos3)
            if len(vis_enemies) >= 2
            else len(vis_enemies)
        )
        outnumbered = (
            (len(close_enemies) >= 2 and hp_frac < outnum_thresh)
            or (exposed_los_count >= 2 and hp_frac < 0.66 and self.threats.damage_taken_recent > 70.0)
        )
        favourable_finish = (
            nearest_enemy is not None
            and not self.profile.pacifist
            and nearest_enemy.hp <= 220.0
            and hp > nearest_enemy.hp * 1.35
            and len(close_enemies) <= 1
            and not outnumbered
        )
        low_hp_threat = (
            hp_frac < retreat_thresh
            and not favourable_finish
            and (nearest_enemy is not None or self.threats.damage_taken_recent > 80.0)
        )
        early_outgeared = (
            self.phase == "EARLY_GAME"
            and me["weapons"][1] is None
            and nearest_enemy is not None
            and ("lightning" in nearest_enemy.weapons or "laser" in nearest_enemy.weapons)
            and nearest_enemy.hp > hp * 1.28
            and hp_frac < 0.62
        )
        bad_duel = (
            nearest_enemy is not None
            and not self.profile.pacifist
            and not favourable_finish
            and (
                (nearest_enemy.hp > hp * 1.45 and nearest_enemy.hp > 450.0 and hp_frac < 0.65)
                or early_outgeared
            )
        )
        # Dual-purpose NPC emergency check: only retreat from an NPC if critically low (<38% HP), no enemy mech, and taking NPC fire
        npc_emergency = (
            nearest_enemy is None
            and hp_frac < 0.38
            and (now - self.threats.last_npc_damaged_t) <= 2.5
            and bool(self.threats.visible_npcs())
        )
        want_retreat = (
            (self.profile.pacifist and (nearest_enemy is not None and nearest_dist < 110.0))
            or low_hp_threat
            or outnumbered
            or bad_duel
            or npc_emergency
        )
        if not want_retreat:
            self._retreat_goal = None

        # Candidate target & resource evaluation
        prefer_avoid_unfavorable = (
            self.phase == "EARLY_GAME" and me["weapons"][1] is None and cubes <= 1
        )
        mech_target: Optional[TrackedEnemy] = None
        if not self.profile.pacifist:
            mech_target = self.combat.select_mech_target(
                obs,
                self.threats,
                allow_long_range=(not zone_urgent and not zone_advance),
                phase=self.phase,
                prefer_avoid_unfavorable=prefer_avoid_unfavorable,
            )
        target_prio = getattr(self.combat, "last_target_score", 0.0)

        loot_obj: Optional[LootObjective] = None
        if not zone_urgent and not want_retreat:
            loot_obj = self.resources.find_best_loot(
                obs,
                self.threats,
                play_center,
                safe_play_r,
                max_detour=self.profile.loot_detour_m,
                allow_noisy_farm=(nearest_dist > 80.0),
                phase_econ_mult=econ_mult,
                phase=self.phase,
                inward_dir=inward_dir,
            )
            if loot_obj is not None and loot_obj.kind == "crate" and loot_obj.target_id is not None:
                cid = int(loot_obj.target_id)
                crate_hp = 300.0
                for cr in obs.get("crates") or []:
                    if int(cr["id"]) == cid:
                        crate_hp = float(cr.get("hp", 300.0))
                        break
                if self._crate_id != cid:
                    self._crate_id = cid
                    self._crate_start_t = now
                    self._crate_last_hp = crate_hp
                    self._crate_last_progress_t = now
                    self._crate_start_hp_me = hp
                    self._crate_fire_ticks = 0
                else:
                    if crate_hp < self._crate_last_hp - 1.0:
                        self._crate_last_hp = crate_hp
                        self._crate_last_progress_t = now
                    crate_timed_out = (
                        (now - self._crate_start_t) > 7.5
                        or ((now - self._crate_last_progress_t) > 3.5 and self._crate_fire_ticks >= 10)
                        or (self._crate_start_hp_me - hp) > 90.0
                    )
                    if crate_timed_out:
                        self.resources.mark_crate_failed(cid, now, cooldown_s=35.0)
                        self._crate_id = None
                        loot_obj = self.resources.find_best_loot(
                            obs,
                            self.threats,
                            play_center,
                            safe_play_r,
                            max_detour=self.profile.loot_detour_m,
                            allow_noisy_farm=(nearest_dist > 80.0),
                            phase_econ_mult=econ_mult,
                            phase=self.phase,
                            inward_dir=inward_dir,
                        )
            else:
                self._crate_id = None
        else:
            self._crate_id = None

        # --- UTILITY SCORING ACROSS CANDIDATE OBJECTIVES ---
        local_danger = self.threats.danger_at(x, y, now, include_npcs=(hp_frac < 0.65))
        candidates_considered: List[str] = ["SAFE_ROTATE"]
        scores: Dict[str, float] = {}
        benefits: Dict[str, float] = {}
        risks: Dict[str, float] = {}
        reasons: List[str] = [f"PHASE_{self.phase}"]

        # 1. SAFE_ROTATE baseline score
        b_safe = 45.0
        r_safe = local_danger * 8.0
        scores["SAFE_ROTATE"] = round(b_safe - r_safe + (self.HYSTERESIS_BONUS if self.mode == "SAFE_ROTATE" else 0.0), 1)
        benefits["SAFE_ROTATE"] = b_safe
        risks["SAFE_ROTATE"] = round(r_safe, 1)

        # 2. ZONE_URGENT score
        if zone_urgent:
            candidates_considered.append("ZONE_URGENT")
            b_zone = 260.0 + float(zone.get("dps", 20.0)) * 1.5
            r_zone = local_danger * 5.0
            scores["ZONE_URGENT"] = round(b_zone - r_zone, 1)
            benefits["ZONE_URGENT"] = round(b_zone, 1)
            risks["ZONE_URGENT"] = round(r_zone, 1)

        # 2b. ZONE_ADVANCE score
        if zone_advance:
            candidates_considered.append("ZONE_ADVANCE")
            b_zadv = 125.0 + min(40.0, dist_to_next_safe * 0.25)
            r_zadv = local_danger * 6.0
            scores["ZONE_ADVANCE"] = round(b_zadv - r_zadv + (self.HYSTERESIS_BONUS if self.mode == "ZONE_ADVANCE" else 0.0), 1)
            benefits["ZONE_ADVANCE"] = round(b_zadv, 1)
            risks["ZONE_ADVANCE"] = round(r_zadv, 1)

        # 3. RETREAT_HEAL score
        if want_retreat:
            candidates_considered.append("RETREAT_HEAL")
            b_ret = 195.0 + (1.0 - hp_frac) * 90.0
            r_ret = local_danger * 6.0
            scores["RETREAT_HEAL"] = round(b_ret - r_ret + (self.HYSTERESIS_BONUS if self.mode == "RETREAT_HEAL" else 0.0), 1)
            benefits["RETREAT_HEAL"] = round(b_ret, 1)
            risks["RETREAT_HEAL"] = round(r_ret, 1)

        # 4. COMBAT_KITE score
        if mech_target is not None and not self.profile.pacifist and not zone_urgent and not want_retreat:
            candidates_considered.append("COMBAT_KITE")
            cube_adv = (cubes - mech_target.cubes) * 6.0
            b_cmb = 110.0 + min(120.0, target_prio * 0.45) + max(-20.0, min(30.0, cube_adv))
            r_cmb = (1.0 - hp_frac) * 55.0 + (20.0 if len(close_enemies) >= 2 else 0.0)
            scores["COMBAT_KITE"] = round(b_cmb - r_cmb + (self.HYSTERESIS_BONUS if self.mode == "COMBAT_KITE" else 0.0), 1)
            benefits["COMBAT_KITE"] = round(b_cmb, 1)
            risks["COMBAT_KITE"] = round(r_cmb, 1)

        # 5. RESOURCE ECONOMY score
        l_mode = ""
        if loot_obj is not None and not zone_urgent and not want_retreat:
            l_mode = f"LOOT_{loot_obj.kind.upper()}"
            candidates_considered.append(l_mode)
            b_loot = 75.0 + loot_obj.score
            r_loot = local_danger * 10.0 + (25.0 if mech_target is not None else 0.0)
            scores[l_mode] = round(b_loot - r_loot + (self.HYSTERESIS_BONUS if self.mode == l_mode else 0.0), 1)
            benefits[l_mode] = round(b_loot, 1)
            risks[l_mode] = round(r_loot, 1)

        # --- EXECUTE PRIORITY-CONSTRAINED UTILITY SELECTION (COOPERATING LEGS & TORSO) ---
        goal: Vec2 = next_center
        want_boost = False
        combat_plan = CombatPlan()
        danger_weight = self.profile.danger_nav_weight
        avoid_npcs_nav = hp_frac < 0.65

        recent_atk_ids = self.threats.recent_attackers(now, window=4.5)
        immediate_combat = (
            mech_target is not None
            and not self.profile.pacifist
            and (
                nearest_dist <= 65.0
                or mech_target.id in recent_atk_ids
                or (mech_target.hp <= 360.0 and math.hypot(mech_target.pos[0] - x, mech_target.pos[1] - y) <= 115.0)
            )
        )
        drop_ready_to_hold = False
        if loot_obj is not None and loot_obj.kind == "drop" and math.hypot(loot_obj.pos[0] - x, loot_obj.pos[1] - y) <= 8.5:
            for dr in obs.get("drops") or []:
                if int(dr["id"]) == loot_obj.target_id and float(dr.get("lands_in", 99.0)) <= 2.5:
                    drop_ready_to_hold = True
                    break
        holding_drop = drop_ready_to_hold
        combat_wins_utility = (
            mech_target is not None
            and not self.profile.pacifist
            and scores.get("COMBAT_KITE", -1e9) >= max(scores.get(l_mode, -1e9), scores.get("ZONE_ADVANCE", -1e9))
        )

        if zone_urgent:
            self.mode = "ZONE_URGENT"
            reasons.append("OUTSIDE_ZONE" if not in_zone else "SHRINK_DEADLINE")
            goal = self._pick_quiet_zone_anchor(obs, next_center, next_radius, avoid_npcs=avoid_npcs_nav, phase=self.phase)
            want_boost = (not in_zone) or (energy > 30.0 and dist_to_next_safe > 25.0)
            danger_weight = 1.2
            if mech_target is not None and not self.profile.pacifist:
                t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_center, mech_target.vel, mech_target.id,
                    mech_target.weapons, mech_target.hp, is_mech=True,
                    allow_fire=True, laser_shorted=laser_shorted,
                    allow_laser_start=False, target_cubes=mech_target.cubes,
                )

        elif want_retreat:
            self.mode = "RETREAT_HEAL"
            if outnumbered:
                reasons.append("OUTNUMBERED_RETREAT")
            elif bad_duel:
                reasons.append("UNFAVORABLE_DUEL_HP")
            elif npc_emergency:
                reasons.append("NPC_CRITICAL_AVOID")
            else:
                reasons.append("LOW_HP_MOVING_RETREAT")

            if nearest_enemy is not None:
                threat_ref = nearest_enemy.pos
                threat_vel = nearest_enemy.vel
            elif npc_emergency and self.threats.visible_npcs():
                n0 = self.threats.visible_npcs()[0]
                threat_ref = tuple(n0["pos"])
                threat_vel = tuple(n0.get("vel", (0.0, 0.0, 0.0)))
            else:
                threat_ref = (
                    x + math.cos(self.threats.rear_alert_bearing or 0.0) * 40.0,
                    y + math.sin(self.threats.rear_alert_bearing or 0.0) * 40.0,
                    z,
                )
                threat_vel = (0.0, 0.0, 0.0)
            goal, _, _ = self._choose_retreat_goal(obs, threat_ref, threat_vel, play_center, safe_play_r)
            enemy_has_los = nearest_enemy is not None and (
                (not self.world.loaded)
                or self.world.has_los(
                    (nearest_enemy.pos[0], nearest_enemy.pos[1], nearest_enemy.pos[2] + 7.0),
                    (x, y, z + 5.0),
                    step=3.5,
                )
            )
            want_boost = (
                (dodge_vec is not None and energy > 12.0)
                or (
                    (enemy_has_los or self.threats.damage_taken_recent > 40.0)
                    and (nearest_dist < 115.0 or hp_frac < 0.35)
                    and (energy > 25.0 or (hp_frac < 0.28 and energy > 14.0))
                )
            )
            danger_weight = 3.5
            avoid_npcs_nav = True

            if nearest_enemy is not None:
                t_center = (nearest_enemy.pos[0], nearest_enemy.pos[1], nearest_enemy.pos[2] + 4.5)
                allow_defensive_fire = (
                    not self.profile.pacifist
                    and (nearest_enemy.hp <= 260.0 or nearest_dist < 42.0)
                )
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_center, nearest_enemy.vel, nearest_enemy.id,
                    nearest_enemy.weapons, nearest_enemy.hp, is_mech=True,
                    allow_fire=allow_defensive_fire, laser_shorted=laser_shorted,
                    allow_laser_start=False, target_cubes=nearest_enemy.cubes,
                )

        elif holding_drop:
            self.mode = "LOOT_DROP"
            reasons.append("HOLD_SUPPLY_DROP")
            lx, ly = loot_obj.pos[0], loot_obj.pos[1]
            ang = now * 1.8
            goal = self.world.clamp_to_bounds(lx + math.cos(ang) * 3.2, ly + math.sin(ang) * 3.2)
            want_boost = False
            if mech_target is not None and not self.profile.pacifist:
                reasons.append("DEFEND_DROP_RING")
                t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_center, mech_target.vel, mech_target.id,
                    mech_target.weapons, mech_target.hp, is_mech=True,
                    allow_fire=True, laser_shorted=laser_shorted,
                    allow_laser_start=True, target_cubes=mech_target.cubes,
                )

        elif mech_target is not None and not self.profile.pacifist and (immediate_combat or combat_wins_utility or (loot_obj is None and not zone_advance)):
            self.mode = "COMBAT_KITE"
            if mech_target.hp <= 400.0:
                reasons.append("FINISH_WOUNDED_TARGET")
            else:
                reasons.append("FAVORABLE_ENGAGEMENT")
            t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
            combat_plan = self.combat.plan_attack_on_point(
                obs, t_center, mech_target.vel, mech_target.id,
                mech_target.weapons, mech_target.hp, is_mech=True,
                allow_fire=True, laser_shorted=laser_shorted,
                allow_laser_start=True, target_cubes=mech_target.cubes,
            )

            # BH-005: Reload-in-cover when primary gun is reloading & no laser/lightning active
            gun_reloading = any(w and w["kind"] == "gun" and float(w.get("reload", 0.0)) > 0.35 for w in me["weapons"])
            can_lightning = any(w and w["kind"] == "lightning" for w in me["weapons"]) and math.hypot(mech_target.pos[0] - x, mech_target.pos[1] - y) < 43.0
            laser_active = charging_laser or (laser_ready and combat_plan.weapon_kind == "laser")
            if gun_reloading and not can_lightning and not laser_active and self.world.loaded:
                reasons.append("RELOAD_IN_COVER")
                if self._cover_pt is None or (now - self._cover_t) > 1.2:
                    self._cover_pt = self.world.find_cover_point(pos3, mech_target.pos, play_center, play_radius, search_radius=36.0)
                    self._cover_t = now
                if self._cover_pt is not None:
                    goal = self._cover_pt
                else:
                    away_dx = x - mech_target.pos[0]
                    away_dy = y - mech_target.pos[1]
                    away_n = max(1.0, math.hypot(away_dx, away_dy))
                    step_d = min(18.0, max(4.0, safe_play_r))
                    fx = x + (away_dx / away_n) * step_d
                    fy = y + (away_dy / away_n) * step_d
                    d_pc = math.hypot(fx - play_center[0], fy - play_center[1])
                    if d_pc > safe_play_r and d_pc > 1e-3:
                        scale = max(2.0, safe_play_r - 1.5) / d_pc
                        fx = play_center[0] + (fx - play_center[0]) * scale
                        fy = play_center[1] + (fy - play_center[1]) * scale
                    goal = self.world.clamp_to_bounds(fx, fy)
                want_boost = dodge_vec is not None and energy > 25.0
            else:
                flip_interval = 1.15 + ((tick * 17 + self._seed) % 7) * 0.08
                if now - self._last_strafe_flip_t > flip_interval:
                    self._strafe_dir = -self._strafe_dir
                    self._last_strafe_flip_t = now

                dx = mech_target.pos[0] - x
                dy = mech_target.pos[1] - y
                d = max(1.0, math.hypot(dx, dy))
                ux, uy = dx / d, dy / d
                px, py = -uy * self._strafe_dir, ux * self._strafe_dir

                # Bias strafe away from a second visible enemy if present
                other_e = next((e for e in vis_enemies if e.id != mech_target.id), None)
                if other_e is not None:
                    o_dx = other_e.pos[0] - x
                    o_dy = other_e.pos[1] - y
                    if (px * o_dx + py * o_dy) > 0.0 and not (self.world.loaded and self.world.is_blocked_at(x - px * 9.0, y - py * 9.0, z, clearance=0.4)):
                        self._strafe_dir = -self._strafe_dir
                        px, py = -px, -py

                if self.world.loaded and self.world.is_blocked_at(x + px * 9.0, y + py * 9.0, z, clearance=0.4):
                    self._strafe_dir = -self._strafe_dir
                    self._last_strafe_flip_t = now
                    px, py = -px, -py

                has_lightning = any(w and w["kind"] == "lightning" for w in me["weapons"])
                has_laser = any(w and w["kind"] == "laser" for w in me["weapons"])
                enemy_has_lightning = "lightning" in mech_target.weapons
                if has_lightning and not enemy_has_lightning and hp >= mech_target.hp * 0.85:
                    desired_r = 28.0
                elif enemy_has_lightning:
                    desired_r = 68.0
                elif has_laser and not has_lightning:
                    desired_r = 72.0
                elif self.phase == "FINAL_CIRCLE":
                    desired_r = 38.0
                else:
                    desired_r = 52.0

                radial = 0.0
                if d > desired_r + 12.0:
                    radial = 0.55
                elif d < desired_r - 10.0:
                    radial = -0.65

                step_k = min(24.0, max(4.0, safe_play_r))
                gx = x + (px * 0.85 + ux * radial) * step_k
                gy = y + (py * 0.85 + uy * radial) * step_k
                d_pc = math.hypot(gx - play_center[0], gy - play_center[1])
                if d_pc > safe_play_r and d_pc > 1e-3:
                    scale = max(2.0, safe_play_r - 1.5) / d_pc
                    gx = play_center[0] + (gx - play_center[0]) * scale
                    gy = play_center[1] + (gy - play_center[1]) * scale
                goal = self.world.clamp_to_bounds(gx, gy)
                want_boost = dodge_vec is not None and energy > 25.0

        elif loot_obj is not None and (not zone_advance or scores.get(l_mode, 0.0) >= scores.get("ZONE_ADVANCE", 0.0)):
            self.mode = f"LOOT_{loot_obj.kind.upper()}"
            reasons.append(f"ACQUIRE_{loot_obj.kind.upper()}")
            lx, ly = loot_obj.pos[0], loot_obj.pos[1]
            if loot_obj.kind == "crate":
                if loot_obj.standoff_pos is not None:
                    goal = loot_obj.standoff_pos
                else:
                    ang_c = math.atan2(y - ly, x - lx)
                    goal = self.world.clamp_to_bounds(lx + math.cos(ang_c) * 11.5, ly + math.sin(ang_c) * 11.5)
            elif loot_obj.kind == "drop" and math.hypot(lx - x, ly - y) < 6.0:
                ang = now * 1.8
                goal = (lx + math.cos(ang) * 3.5, ly + math.sin(ang) * 3.5)
            elif loot_obj.kind == "npc" and not self.profile.pacifist:
                dx_n, dy_n = lx - x, ly - y
                dn = max(1.0, math.hypot(dx_n, dy_n))
                goal = self.world.clamp_to_bounds(
                    lx - (dy_n / dn) * 10.0 * self._strafe_dir,
                    ly + (dx_n / dn) * 10.0 * self._strafe_dir,
                )
            else:
                goal = (lx, ly)
            want_boost = False
            if loot_obj.kind == "crate" and not self.profile.pacifist:
                t_pos = self.resources.crate_aim_point(loot_obj.pos)
                muzzle_h = self.resources.effective_muzzle_height(me)
                geom_ok = self.resources.has_valid_crate_geometry((x, y, z + muzzle_h), loot_obj.pos)
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_pos, (0.0, 0.0, 0.0), loot_obj.target_id,
                    None, 300.0,
                    is_mech=False, allow_fire=geom_ok, laser_shorted=laser_shorted,
                )
                if any(combat_plan.fire):
                    self._crate_fire_ticks += 1
            elif loot_obj.kind == "npc" and not self.profile.pacifist:
                t_pos = (loot_obj.pos[0], loot_obj.pos[1], loot_obj.pos[2] + 1.8)
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_pos, (0.0, 0.0, 0.0), loot_obj.target_id,
                    None, 350.0,
                    is_mech=False, allow_fire=True, laser_shorted=laser_shorted,
                )
            elif mech_target is not None and not self.profile.pacifist:
                t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_center, mech_target.vel, mech_target.id,
                    mech_target.weapons, mech_target.hp, is_mech=True,
                    allow_fire=True, laser_shorted=laser_shorted,
                    allow_laser_start=(loot_obj.kind == "drop"), target_cubes=mech_target.cubes,
                )

        elif zone_advance:
            self.mode = "ZONE_ADVANCE"
            reasons.append("PROACTIVE_ZONE_ROTATION")
            goal = self._pick_quiet_zone_anchor(obs, next_center, next_radius, avoid_npcs=avoid_npcs_nav, phase=self.phase)
            want_boost = dodge_vec is not None and energy > 25.0
            danger_weight = 1.8
            if mech_target is not None and not self.profile.pacifist:
                t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_center, mech_target.vel, mech_target.id,
                    mech_target.weapons, mech_target.hp, is_mech=True,
                    allow_fire=True, laser_shorted=laser_shorted,
                    allow_laser_start=False, target_cubes=mech_target.cubes,
                )

        else:
            self.mode = "SAFE_ROTATE"
            if self.phase == "EARLY_GAME":
                reasons.append("EARLY_ADAPTIVE_ANCHOR")
                goal = self._pick_early_anchor(obs, next_center, next_radius, play_center, safe_play_r, avoid_npcs=avoid_npcs_nav)
            else:
                reasons.append("CORNER_COVER_ANCHOR")
                goal = self._pick_quiet_zone_anchor(obs, next_center, next_radius, avoid_npcs=avoid_npcs_nav, phase=self.phase)
            want_boost = False
            if mech_target is not None and not self.profile.pacifist:
                t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
                combat_plan = self.combat.plan_attack_on_point(
                    obs, t_center, mech_target.vel, mech_target.id,
                    mech_target.weapons, mech_target.hp, is_mech=True,
                    allow_fire=True, laser_shorted=laser_shorted,
                    allow_laser_start=True, target_cubes=mech_target.cubes,
                )

        if dodge_vec is not None:
            reasons.append("EVADE_INCOMING_FIRE")

        # --- NAVIGATION ---
        suppress_stuck = charging_laser or self.mode == "COMBAT_KITE" or (
            self.profile.silent_when_hidden and self.mode == "SAFE_ROTATE"
        )
        danger_sources = self.threats.snapshot_sources(
            now,
            include_npcs=(self.mode != "LOOT_NPC" and avoid_npcs_nav),
            npc_weight=5.0 if hp_frac < 0.55 else 3.5,
        )
        move_xy, boost, jump, _ = self.nav.update_and_steer(
            tick=tick,
            pos=pos3,
            vel=vel3,
            grounded=grounded,
            energy=energy,
            goal=goal,
            danger_weight=danger_weight,
            want_boost=want_boost,
            dodge_vec=dodge_vec,
            suppress_stuck=suppress_stuck,
            danger_sources=danger_sources,
        )

        if self.profile.silent_when_hidden and self.mode == "SAFE_ROTATE" and dodge_vec is None and not boost:
            scale = min(1.0, 4.2 / walk_speed)
            move_xy = [move_xy[0] * scale, move_xy[1] * scale]

        # --- AIM FALLBACK (Rear-armor protection + opening 360° sweep + scanning) ---
        recent_atk_b = self.threats.recent_attacker_bearing(pos3, now, window=4.0)
        if combat_plan.aim is not None:
            aim = combat_plan.aim
        elif self.threats.rear_alert_bearing is not None:
            aim = [self.threats.rear_alert_bearing, 0.0]
        elif nearest_enemy is not None and (nearest_dist < 165.0 or (nearest_enemy.charging and nearest_dist < 265.0)):
            aim = [math.atan2(nearest_enemy.pos[1] - y, nearest_enemy.pos[0] - x), 0.0]
        elif recent_atk_b is not None:
            aim = [recent_atk_b, 0.0]
        elif now < 2.2 and self._initial_yaw is not None:
            aim = [math.remainder(self._initial_yaw + now * 2.8, math.tau), 0.0]
        else:
            move_yaw = math.atan2(move_xy[1], move_xy[0]) if math.hypot(move_xy[0], move_xy[1]) > 0.1 else float(me["aim"][0])
            scan_offset = math.sin(now * 1.25) * 1.35
            aim = [math.remainder(move_yaw + scan_offset, math.tau), 0.0]

        # --- ASSEMBLE ACTION ---
        act: Dict[str, Any] = {
            "tick": tick,
            "move": move_xy,
            "boost": bool(boost),
            "jump": bool(jump),
            "aim": aim,
            "fire": combat_plan.fire if not self.profile.pacifist else [False, False],
        }

        if not self.profile.pacifist and self.combat.should_early_reload(
            obs, in_combat=(mech_target is not None or combat_plan.target_id is not None)
        ):
            act["reload"] = True

        pick = self.resources.choose_upgrade(obs, phase=self.phase)
        if pick is not None:
            act["pick"] = pick

        take_id, drop_slot = self.resources.choose_weapon_take(obs)
        if take_id is not None:
            act["take"] = take_id
            if drop_slot is not None:
                act["drop"] = drop_slot

        # Record structured decision metadata for DecisionTelemetry
        self.last_decision_meta = {
            "phase": self.phase,
            "selected_objective": self.mode,
            "candidates": candidates_considered,
            "scores": scores,
            "risk_factors": {
                "local_danger": round(local_danger, 2),
                "dist_to_next_safe": round(dist_to_next_safe, 1),
                "sec_until_shrink": round(sec_until_shrink, 1),
                "recent_dmg_3s": round(self.threats.damage_taken_recent, 1),
                "outnumbered": outnumbered,
                "incoming_dodge": dodge_vec is not None,
            },
            "target_id": combat_plan.target_id,
            "target_priority": target_prio if mech_target is not None else (loot_obj.score if loot_obj is not None else 0.0),
            "goal": goal,
            "reason_codes": reasons,
            "expected_benefit": benefits.get(self.mode, 0.0),
            "expected_risk": risks.get(self.mode, 0.0),
        }

        return (act, self.mode, goal)
