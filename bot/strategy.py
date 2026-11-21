"""Strategic decision controller supporting tactical_survivor, pure_pacifist, and opportunistic_hunter."""
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
    selective_engage: bool = True
    hold_silent_at_cover: bool = True
    retreat_hp_frac: float = 0.58
    outnumbered_hp_frac: float = 0.82
    engage_range: float = 130.0
    danger_nav_weight: float = 2.4
    loot_detour_m: float = 55.0
    quiet_ring_frac: float = 0.62
    silent_when_hidden: bool = False


PROFILES: Dict[str, StrategyProfile] = {
    "tactical_survivor": StrategyProfile(
        name="tactical_survivor",
        bot_name="tactical-surv-v3",
        pacifist=False,
        selective_engage=True,
        hold_silent_at_cover=True,
        retreat_hp_frac=0.58,
        outnumbered_hp_frac=0.82,
        engage_range=135.0,
        danger_nav_weight=2.4,
        loot_detour_m=55.0,
        quiet_ring_frac=0.62,
        silent_when_hidden=False,
    ),
    "pure_pacifist": StrategyProfile(
        name="pure_pacifist",
        bot_name="pacifist-v2",
        pacifist=True,
        selective_engage=True,
        hold_silent_at_cover=True,
        retreat_hp_frac=0.99,
        outnumbered_hp_frac=0.99,
        engage_range=0.0,
        danger_nav_weight=4.0,
        loot_detour_m=42.0,
        quiet_ring_frac=0.68,
        silent_when_hidden=True,
    ),
    "opportunistic_hunter": StrategyProfile(
        name="opportunistic_hunter",
        bot_name="hunter-v2.1",
        pacifist=False,
        selective_engage=False,
        hold_silent_at_cover=False,
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
    """High-level mode selection and tactical goal/movement/combat coordination."""

    def __init__(
        self,
        world: WorldModel,
        nav: Navigator,
        threats: ThreatTracker,
        combat: CombatPlanner,
        resources: ResourcePlanner,
        profile: StrategyProfile,
    ):
        self.world = world
        self.nav = nav
        self.threats = threats
        self.combat = combat
        self.resources = resources
        self.profile = profile
        self.mode: str = "SAFE_ROTATE"
        self._anchor: Optional[Vec2] = None
        self._anchor_t: float = -100.0
        self._cover_pt: Optional[Vec2] = None
        self._cover_t: float = -100.0
        self._strafe_dir: float = 1.0
        self._last_strafe_flip_t: float = 0.0

    def _pick_quiet_zone_anchor(
        self,
        obs: Dict[str, Any],
        zone_center: Vec2,
        next_radius: float,
    ) -> Vec2:
        """Pick a cover-adjacent, low-threat walkable position inside the next safe circle."""
        now = float(obs["t"])
        me = obs["self"]
        mx, my, _ = me["pos"]
        zcx, zcy = zone_center

        # Keep existing anchor for 3.5s if still safely inside next_radius and low threat
        if self._anchor is not None and (now - self._anchor_t) < 3.5:
            ax, ay = self._anchor
            if math.hypot(ax - zcx, ay - zcy) <= max(5.0, next_radius - 8.0):
                if self.threats.danger_at(ax, ay, now) < 3.0:
                    return self._anchor

        if next_radius <= 20.0:
            pt = self.world.clamp_to_bounds(zcx, zcy, margin=8.0)
            self._anchor = pt
            self._anchor_t = now
            return pt

        ring_r = max(8.0, min(next_radius - 14.0, next_radius * self.profile.quiet_ring_frac))
        best_pt: Vec2 = self.world.clamp_to_bounds(zcx, zcy, margin=8.0)
        best_score = -1e18

        # Sample 16 directions around the quiet ring + 8 inner ring points
        for ring_mult in (1.0, 0.55):
            r = ring_r * ring_mult
            count = 16 if ring_mult == 1.0 else 8
            for k in range(count):
                ang = (math.tau * k) / count
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
                danger = self.threats.danger_at(wx, wy, now)
                dist_me = math.hypot(wx - mx, wy - my)
                idx = gy * self.world.nav_nx + gx
                has_cover = bool(self.world.loaded and self.world.cover_adj[idx])
                deg = self.world.escape_deg[idx] if (self.world.loaded and self.world.escape_deg) else 6
                # Reward building corners (4..6 neighbors = cover + escape routes), penalize dead ends (<= 2)
                corner_bonus = 34.0 if (has_cover and 4 <= deg <= 6) else (16.0 if has_cover else 0.0)
                if deg <= 2:
                    corner_bonus -= 40.0
                # Stay away from map-boundary pinch walls
                edge_dist = min(wx - self.world.min_x, self.world.max_x - wx, wy - self.world.min_y, self.world.max_y - wy)
                edge_pen = 22.0 if edge_dist < 28.0 else 0.0
                score = corner_bonus - danger * 26.0 - dist_me * 0.14 - edge_pen
                if score > best_score:
                    best_score = score
                    best_pt = (wx, wy)

        self._anchor = best_pt
        self._anchor_t = now
        return best_pt

    def decide(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Vec2]:
        """Compute full action dict, mode label, and current goal for debug drawing."""
        me = obs["self"]
        x, y, z = me["pos"]
        pos3: Vec3 = (float(x), float(y), float(z))
        vel3: Vec3 = (float(me["vel"][0]), float(me["vel"][1]), float(me["vel"][2]))
        now = float(obs["t"])
        tick = int(obs["tick"])
        hp = float(me["hp"])
        max_hp = max(1.0, float(me["max_hp"]))
        hp_frac = hp / max_hp
        energy = float(me["energy"])
        grounded = bool(me["grounded"])
        in_zone = bool(me["in_zone"])
        walk_speed = max(1.0, float(me["stats"].get("walk_speed", 11.7)))

        zone = obs["zone"]
        cur_center: Vec2 = (float(zone["center"][0]), float(zone["center"][1]))
        cur_radius = float(zone["radius"])
        next_center: Vec2 = (float(zone["next_center"][0]), float(zone["next_center"][1]))
        next_radius = float(zone["next_radius"])
        shrink_start = float(zone["shrink_start"])
        shrink_end = float(zone["shrink_end"])

        # Linearly interpolate active zone circle during shrink
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

        safe_radius = max(0.0, next_radius - 14.0)
        dist_to_next_safe = max(0.0, math.hypot(x - next_center[0], y - next_center[1]) - safe_radius)
        dist_to_active_edge = active_r - math.hypot(x - active_cx, y - active_cy)
        walk_s = dist_to_next_safe / walk_speed
        sec_until_shrink = shrink_start - now

        # BH-006 + BH-103: Final-zone boundary clamping with proportional margin & active_center pairing
        if next_radius <= 10.0:
            edge_margin = min(10.0, active_r * 0.35)
            zone_urgent = (not in_zone) or (active_r > 3.5 and dist_to_active_edge < edge_margin)
        else:
            zone_urgent = (not in_zone) or (dist_to_next_safe > 0.0 and sec_until_shrink <= walk_s + 12.0)

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

        # Incoming projectile / charging laser dodge
        dodge_vec = self.threats.compute_dodge_vector(obs, next_center)

        # Evaluate retreat condition
        close_enemies = [e for e in vis_enemies if math.hypot(e.pos[0] - x, e.pos[1] - y) < 85.0]
        outnumbered = len(close_enemies) >= 2 and hp_frac < self.profile.outnumbered_hp_frac
        low_hp_threat = hp_frac < self.profile.retreat_hp_frac and (
            nearest_enemy is not None or self.threats.damage_taken_recent > 80.0
        )
        # Disadvantageous 1v1: enemy has much more HP & cubes and isn't low
        bad_duel = (
            nearest_enemy is not None
            and not self.profile.pacifist
            and nearest_enemy.hp > hp * 1.45
            and nearest_enemy.hp > 450.0
            and hp_frac < 0.65
        )
        want_retreat = (
            (self.profile.pacifist and (nearest_enemy is not None and nearest_dist < 110.0))
            or low_hp_threat
            or outnumbered
            or bad_duel
        )

        # Target selection (if not pacifist)
        mech_target: Optional[TrackedEnemy] = None
        if not self.profile.pacifist:
            mech_target = self.combat.select_mech_target(
                obs,
                self.threats,
                allow_long_range=not zone_urgent,
                selective=self.profile.selective_engage,
            )

        # Loot evaluation when safe
        loot_obj: Optional[LootObjective] = None
        if not zone_urgent and not want_retreat and (mech_target is None or nearest_dist > 85.0):
            loot_obj = self.resources.find_best_loot(
                obs,
                self.threats,
                play_center,
                max(5.0, play_radius - 10.0),
                max_detour=self.profile.loot_detour_m,
            )

        # --- MODE SELECTION ---
        goal: Vec2 = next_center
        want_boost = False
        combat_plan = CombatPlan()
        danger_weight = self.profile.danger_nav_weight

        if zone_urgent:
            self.mode = "ZONE_URGENT"
            goal = self._pick_quiet_zone_anchor(obs, next_center, next_radius)
            want_boost = (not in_zone) or (energy > 30.0 and dist_to_next_safe > 25.0)
            danger_weight = 1.2  # Don't take huge detours outside the zone
            if mech_target is not None and not self.profile.pacifist:
                # Shoot back opportunistically while rotating to safety
                t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
                combat_plan = self.combat.plan_attack_on_point(
                    obs,
                    t_center,
                    mech_target.vel,
                    mech_target.id,
                    mech_target.weapons,
                    mech_target.hp,
                    is_mech=True,
                    allow_fire=True,
                    laser_shorted=laser_shorted,
                )

        elif want_retreat:
            self.mode = "RETREAT_HEAL"
            threat_ref = nearest_enemy.pos if nearest_enemy is not None else (
                x + math.cos(self.threats.rear_alert_bearing or 0.0) * 40.0,
                y + math.sin(self.threats.rear_alert_bearing or 0.0) * 40.0,
                z,
            )
            if self._cover_pt is None or (now - self._cover_t) > 1.5 or math.hypot(self._cover_pt[0] - x, self._cover_pt[1] - y) < 6.0:
                self._cover_pt = self.world.find_cover_point(pos3, threat_ref, play_center, play_radius)
                self._cover_t = now
            cover_pt = self._cover_pt
            if cover_pt is not None:
                goal = cover_pt
            else:
                # Back away toward quiet zone anchor
                away_yaw = math.atan2(y - threat_ref[1], x - threat_ref[0])
                cand_x = x + math.cos(away_yaw) * 45.0
                cand_y = y + math.sin(away_yaw) * 45.0
                if math.hypot(cand_x - play_center[0], cand_y - play_center[1]) <= max(10.0, play_radius - 8.0):
                    goal = self.world.clamp_to_bounds(cand_x, cand_y)
                else:
                    goal = self._pick_quiet_zone_anchor(obs, next_center, next_radius)
            want_boost = energy > 22.0 and (nearest_dist < 70.0 or dodge_vec is not None)
            danger_weight = 3.5

            # Protect rear armor: face nearest threat! Fire ONLY if very close or enemy is finishable
            if nearest_enemy is not None:
                t_center = (nearest_enemy.pos[0], nearest_enemy.pos[1], nearest_enemy.pos[2] + 4.5)
                allow_defensive_fire = (
                    not self.profile.pacifist
                    and (nearest_enemy.hp <= 220.0 or nearest_dist < 32.0)
                )
                combat_plan = self.combat.plan_attack_on_point(
                    obs,
                    t_center,
                    nearest_enemy.vel,
                    nearest_enemy.id,
                    nearest_enemy.weapons,
                    nearest_enemy.hp,
                    is_mech=True,
                    allow_fire=allow_defensive_fire,
                    laser_shorted=laser_shorted,
                )

        elif mech_target is not None and not self.profile.pacifist:
            self.mode = "COMBAT_KITE"
            t_center = (mech_target.pos[0], mech_target.pos[1], mech_target.pos[2] + 4.5)
            combat_plan = self.combat.plan_attack_on_point(
                obs,
                t_center,
                mech_target.vel,
                mech_target.id,
                mech_target.weapons,
                mech_target.hp,
                is_mech=True,
                allow_fire=True,
                laser_shorted=laser_shorted,
            )
            # BH-005: Reload-in-cover preserves active/ready laser charge & clamps fallback inside safe_play_r
            gun_reloading = any(w and w["kind"] == "gun" and float(w.get("reload", 0.0)) > 0.35 for w in me["weapons"])
            can_lightning = any(w and w["kind"] == "lightning" for w in me["weapons"]) and math.hypot(mech_target.pos[0] - x, mech_target.pos[1] - y) < 43.0
            laser_active = charging_laser or (laser_ready and combat_plan.weapon_kind == "laser")
            if gun_reloading and not can_lightning and not laser_active and self.world.loaded:
                if self._cover_pt is None or (now - self._cover_t) > 1.2:
                    self._cover_pt = self.world.find_cover_point(pos3, mech_target.pos, play_center, play_radius, search_radius=36.0)
                    self._cover_t = now
                if self._cover_pt is not None:
                    goal = self._cover_pt
                    want_boost = dodge_vec is not None and energy > 25.0
                else:
                    away_dx = x - mech_target.pos[0]
                    away_dy = y - mech_target.pos[1]
                    away_n = max(1.0, math.hypot(away_dx, away_dy))
                    step_d = min(18.0, max(4.0, safe_play_r))
                    fx = x + (away_dx / away_n) * step_d
                    fy = y + (away_dy / away_n) * step_d
                    if math.hypot(fx - play_center[0], fy - play_center[1]) > safe_play_r:
                        fx, fy = play_center
                    goal = self.world.clamp_to_bounds(fx, fy)
                    want_boost = dodge_vec is not None and energy > 25.0
            else:
                # Flip strafe direction every ~1.4s so enemy lead misses
                if now - self._last_strafe_flip_t > 1.4:
                    self._strafe_dir = -self._strafe_dir
                    self._last_strafe_flip_t = now

                dx = mech_target.pos[0] - x
                dy = mech_target.pos[1] - y
                d = max(1.0, math.hypot(dx, dy))
                ux, uy = dx / d, dy / d
                # Perpendicular strafe vector (flip immediately if strafing into a wall or outside safe zone!)
                px, py = -uy * self._strafe_dir, ux * self._strafe_dir
                if self.world.loaded and self.world.is_blocked_at(x + px * 9.0, y + py * 9.0, z, clearance=0.4):
                    self._strafe_dir = -self._strafe_dir
                    self._last_strafe_flip_t = now
                    px, py = -px, -py

                has_lightning = any(w and w["kind"] == "lightning" for w in me["weapons"])
                enemy_has_lightning = "lightning" in mech_target.weapons
                if has_lightning and not enemy_has_lightning and hp >= mech_target.hp * 0.85:
                    desired_r = 26.0
                elif enemy_has_lightning:
                    desired_r = 68.0
                else:
                    desired_r = 55.0

                radial = 0.0
                if d > desired_r + 12.0:
                    radial = 0.55
                elif d < desired_r - 10.0:
                    radial = -0.65

                step_k = min(24.0, max(4.0, safe_play_r))
                gx = x + (px * 0.85 + ux * radial) * step_k
                gy = y + (py * 0.85 + uy * radial) * step_k
                if math.hypot(gx - play_center[0], gy - play_center[1]) > safe_play_r:
                    gx, gy = play_center
                goal = self.world.clamp_to_bounds(gx, gy)
                want_boost = dodge_vec is not None and energy > 25.0

        elif loot_obj is not None:
            self.mode = f"LOOT_{loot_obj.kind.upper()}"
            lx, ly = loot_obj.pos[0], loot_obj.pos[1]
            # Gentle strafe inside supply-drop 8m ring while capturing, or around NPC drones to dodge 90m/s bullets
            if loot_obj.kind == "drop" and math.hypot(lx - x, ly - y) < 6.0:
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
            if loot_obj.kind in ("crate", "npc") and not self.profile.pacifist:
                z_off = 1.2 if loot_obj.kind == "crate" else 1.8
                t_pos = (loot_obj.pos[0], loot_obj.pos[1], loot_obj.pos[2] + z_off)
                combat_plan = self.combat.plan_attack_on_point(
                    obs,
                    t_pos,
                    (0.0, 0.0, 0.0),
                    loot_obj.target_id,
                    None,
                    300.0 if loot_obj.kind == "crate" else 350.0,
                    is_mech=False,
                    allow_fire=True,
                    laser_shorted=laser_shorted,
                )

        else:
            # Rooftop high-ground hold: if safely on a roof inside the safe zone with low threat, hold it!
            on_roof = self.world.loaded and (z - self.world.terrain_height(x, y)) > 3.0
            if (
                on_roof
                and in_zone
                and math.hypot(x - play_center[0], y - play_center[1]) <= max(8.0, play_radius - 14.0)
                and self.threats.danger_at(x, y, now) < 2.5
            ):
                self.mode = "ROOF_HOLD"
                goal = (x, y)
                want_boost = False
            else:
                self.mode = "SAFE_ROTATE"
                goal = self._pick_quiet_zone_anchor(obs, next_center, next_radius)
                want_boost = False

        # --- NAVIGATION ---
        charging_laser = any(w and w["kind"] == "laser" and float(w.get("charge", 0.0)) > 0.0 for w in me["weapons"])
        hold_quiet_cover = (
            self.profile.hold_silent_at_cover
            and self.mode in ("SAFE_ROTATE", "ROOF_HOLD")
            and dodge_vec is None
            and math.hypot(goal[0] - x, goal[1] - y) <= 7.0
        )
        suppress_stuck = charging_laser or self.mode == "COMBAT_KITE" or hold_quiet_cover or (
            self.profile.silent_when_hidden and self.mode == "SAFE_ROTATE"
        )
        if hold_quiet_cover:
            move_xy, boost, jump = [0.0, 0.0], False, False
        else:
            danger_sources = self.threats.snapshot_sources(now)
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

        # Silent walking for pacifist when already safe & hidden
        if self.profile.silent_when_hidden and self.mode == "SAFE_ROTATE" and dodge_vec is None and not boost:
            # Step sound threshold is 5.0 m/s; keep speed ~4.2 m/s
            scale = min(1.0, 4.2 / walk_speed)
            move_xy = [move_xy[0] * scale, move_xy[1] * scale]

        # --- AIM FALLBACK (Rear-armor protection + 360° scanning) ---
        if combat_plan.aim is not None:
            aim = combat_plan.aim
        elif self.threats.rear_alert_bearing is not None:
            aim = [self.threats.rear_alert_bearing, 0.0]
        else:
            # Smooth torso scan around movement heading so 120° FOV sweeps 360°
            move_yaw = math.atan2(move_xy[1], move_xy[0]) if math.hypot(move_xy[0], move_xy[1]) > 0.1 else float(me["aim"][0])
            scan_offset = math.sin(now * 1.1) * 1.25
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

        pick = self.resources.choose_upgrade(obs)
        if pick is not None:
            act["pick"] = pick

        take_id, drop_slot = self.resources.choose_weapon_take(obs)
        if take_id is not None:
            act["take"] = take_id
            if drop_slot is not None:
                act["drop"] = drop_slot

        return (act, self.mode, goal)
