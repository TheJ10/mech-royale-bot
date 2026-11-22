"""Combat planner: lead aiming, laser charge/release, lightning cone, mortar ballistics, target scoring."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from bot.threat import ThreatTracker, TrackedEnemy
from bot.world import Vec3, WorldModel


def angle_diff(a: float, b: float) -> float:
    return math.remainder(a - b, math.tau)


def aim_angles(src: Vec3, dst: Vec3) -> Tuple[float, float]:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    dz = dst[2] - src[2]
    horiz = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, max(1e-4, horiz))
    return (yaw, pitch)


def solve_lead_point(
    muzzle: Vec3,
    target_pos: Vec3,
    target_vel: Vec3,
    proj_speed: float,
    max_lead_s: float = 1.1,
    gravity: float = 22.0,
    shot_offset: float = 3.0,
    min_z: Optional[float] = None,
) -> Vec3:
    """Iterative 3D projectile intercept point with gravity correction for jumping/falling targets."""
    tx, ty, tz = target_pos
    vx, vy, vz = target_vel
    airborne = abs(vz) > 0.6
    t = 0.0
    for _ in range(3):
        px = tx + vx * t
        py = ty + vy * t
        pz = tz + vz * t - (0.5 * gravity * t * t if airborne else 0.0)
        dist = math.dist(muzzle, (px, py, pz))
        t = min(max_lead_s, max(0.0, dist - shot_offset) / max(1.0, proj_speed))
    rz = tz + vz * t - (0.5 * gravity * t * t if airborne else 0.0)
    if min_z is not None and rz < min_z:
        rz = min_z
    return (
        tx + vx * t,
        ty + vy * t,
        rz,
    )


def solve_mortar_aim(
    muzzle: Vec3,
    target_pos: Vec3,
    target_vel: Vec3,
    speed: float,
    gravity: float,
    high_arc: bool = False,
    min_pitch: float = -0.7,
    max_pitch: float = 1.3,
) -> Optional[Tuple[float, float]]:
    """Ballistic yaw/pitch solution for mortar shell with target velocity lead."""
    # Iterate flight time once with horizontal estimate
    tx, ty, tz = target_pos
    vx, vy, vz = target_vel
    px, py, pz = tx, ty, tz
    for _ in range(2):
        dx = px - muzzle[0]
        dy = py - muzzle[1]
        dz = pz - muzzle[2]
        r = math.hypot(dx, dy)
        if r < 14.0:
            return None
        v2 = speed * speed
        v4 = v2 * v2
        disc = v4 - gravity * (gravity * r * r + 2.0 * dz * v2)
        if disc < 0.0:
            return None
        root = math.sqrt(disc)
        tan_theta = (v2 + root) / (gravity * r) if high_arc else (v2 - root) / (gravity * r)
        pitch = math.atan(tan_theta)
        cos_p = math.cos(pitch)
        if cos_p < 0.1:
            return None
        t_flight = min(6.0, r / (speed * cos_p))
        px = tx + vx * t_flight * 0.75
        py = ty + vy * t_flight * 0.75
        pz = tz + vz * t_flight * 0.25

    dx = px - muzzle[0]
    dy = py - muzzle[1]
    dz = pz - muzzle[2]
    r = math.hypot(dx, dy)
    if r < 14.0:
        return None
    v2 = speed * speed
    disc = v2 * v2 - gravity * (gravity * r * r + 2.0 * dz * v2)
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    tan_theta = (v2 + root) / (gravity * r) if high_arc else (v2 - root) / (gravity * r)
    pitch = math.atan(tan_theta)
    if pitch < min_pitch + 0.01 or pitch > max_pitch - 0.01:
        return None
    return (math.atan2(dy, dx), pitch)


class CombatPlan:
    __slots__ = ("aim", "fire", "reload", "target_id", "target_pos", "weapon_kind")

    def __init__(
        self,
        aim: Optional[List[float]] = None,
        fire: Optional[List[bool]] = None,
        reload: bool = False,
        target_id: Optional[int] = None,
        target_pos: Optional[Vec3] = None,
        weapon_kind: Optional[str] = None,
    ):
        self.aim = aim
        self.fire = fire if fire is not None else [False, False]
        self.reload = reload
        self.target_id = target_id
        self.target_pos = target_pos
        self.weapon_kind = weapon_kind


class CombatPlanner:
    """Selects combat targets, computes weapon-specific aim, and manages fire/charge/reload."""

    def __init__(self, world: WorldModel, config: Dict[str, Any]):
        self.world = world
        self.config = config
        mech_cfg = config.get("mech") or {}
        self.min_pitch = float(mech_cfg.get("min_pitch", -0.7))
        self.max_pitch = float(mech_cfg.get("max_pitch", 1.3))
        self.base_muzzle_h = float(mech_cfg.get("muzzle_height", 7.0))
        self.size_per_cube = float((config.get("cubes") or {}).get("size_per_cube", 0.04))
        self._los_cache_tick: int = -1
        self._los_cache: Dict[int, bool] = {}
        self._last_target_id: Optional[int] = None
        self._last_target_t: float = -100.0
        self.last_target_score: float = 0.0

    def _cached_los(self, tick: int, target_id: Optional[int], p0: Vec3, p1: Vec3) -> bool:
        if not self.world.loaded:
            return True
        if tick != self._los_cache_tick:
            self._los_cache_tick = tick
            self._los_cache.clear()
        if target_id is not None and target_id in self._los_cache:
            return self._los_cache[target_id]
        res = self.world.has_los(p0, p1, step=2.0)
        if target_id is not None:
            self._los_cache[target_id] = res
        return res

    def _clamp_pitch(self, pitch: float) -> float:
        return max(self.min_pitch, min(self.max_pitch, pitch))

    def select_mech_target(
        self,
        obs: Dict[str, Any],
        threats: ThreatTracker,
        allow_long_range: bool = True,
        phase: str = "MID_GAME",
        prefer_avoid_unfavorable: bool = False,
    ) -> Optional[TrackedEnemy]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        cur_aim = me["aim"]
        stat_mh = float(me["stats"].get("muzzle_height", self.base_muzzle_h))
        scaled_mh = self.base_muzzle_h * (1.0 + self.size_per_cube * int(me.get("cubes", 0)))
        muzzle_z = mz + max(stat_mh, scaled_mh)
        muzzle = (mx, my, muzzle_z)
        now = float(obs["t"])
        attackers = threats.recent_attackers(now, window=4.5)
        my_weapons = [w["kind"] for w in me["weapons"] if w is not None]

        best: Optional[TrackedEnemy] = None
        best_score = -1e18

        tick = int(obs["tick"])
        gun_noise_cap = 155.0 if phase == "LATE_GAME" else 165.0
        for e in threats.visible_enemies():
            t_center = (e.pos[0], e.pos[1], e.pos[2] + 4.5)
            dist = math.dist(muzzle, t_center)
            if not allow_long_range and dist > 95.0 and e.id not in attackers:
                continue
            if dist > 265.0:
                continue

            has_los = self._cached_los(tick, e.id, muzzle, t_center)
            has_mortar = "mortar" in my_weapons
            if not has_los and not (has_mortar and 18.0 <= dist <= 220.0):
                continue

            yaw_e_to_me = math.atan2(my - e.pos[1], mx - e.pos[0])
            e_facing_err = abs(angle_diff(yaw_e_to_me, e.aim[0]))
            horiz_dist = max(1.0, math.hypot(mx - e.pos[0], my - e.pos[1]))
            closing_speed = ((mx - e.pos[0]) * e.vel[0] + (my - e.pos[1]) * e.vel[1]) / horiz_dist
            hp_frac = e.hp / max(1.0, e.max_hp)

            if (
                prefer_avoid_unfavorable
                and e.id not in attackers
                and dist > 75.0
                and hp_frac > 0.72
                and not (e.charging and e_facing_err < 0.40)
                and closing_speed < 5.0
            ):
                continue

            # Avoid broadcasting noisy long-range gun shots at full-HP non-attackers unless we have a laser or they are charging a laser at us
            if (
                dist > gun_noise_cap
                and e.id not in attackers
                and e.hp > 650.0
                and "laser" not in my_weapons
                and not (e.charging and e_facing_err < 0.35)
            ):
                continue

            score = 0.0
            # Prioritize closer threats & active rushers
            score += max(0.0, 220.0 - dist) * 0.78
            score += max(0.0, closing_speed) * 4.5
            if "lightning" in e.weapons and dist < 60.0:
                score += 70.0
            # High priority for anyone who recently damaged us
            if e.id in attackers:
                score += 135.0
                if dist < 80.0:
                    score += 45.0
            # High priority for low-HP finishable enemies (cubes + placement!)
            score += (1.0 - hp_frac) * 145.0
            if e.hp <= 360.0:
                score += 95.0
            # Enemy aiming at us vs turned away from us (rear-armor 1.5x damage!)
            if e_facing_err < 0.35:
                score += 70.0
                if e.charging:
                    score += 95.0
            elif e_facing_err > math.pi * 0.5:
                score += 75.0  # Exposed rear armor -> 1.5x damage!
            # Weapon triangle bonuses
            if "laser" in my_weapons and "mortar" in e.weapons:
                score += 65.0  # +60% cookoff!
            if "lightning" in my_weapons and "laser" in e.weapons and dist < 52.0:
                score += 80.0  # short enemy laser!
            # Target-switch hysteresis + torso slew tie-breaker
            if e.id == self._last_target_id and (now - self._last_target_t) <= 1.2:
                score += 34.0
            yaw_me_to_e = math.atan2(e.pos[1] - my, e.pos[0] - mx)
            score -= abs(angle_diff(yaw_me_to_e, cur_aim[0])) * 12.0
            if not has_los:
                score -= 90.0  # Prefer direct LoS targets over blind mortar lobs

            if score > best_score:
                best_score = score
                best = e

        self.last_target_score = best_score if best is not None else 0.0
        if best is not None:
            self._last_target_id = best.id
            self._last_target_t = now
        return best

    def _mortar_launch_clear(self, muzzle: Vec3, yaw: float, pitch: float, speed: float, gravity: float) -> bool:
        """Verify first ~14m of mortar trajectory does not hit a close wall/overhang (avoids self-splash)."""
        if not self.world.loaded:
            return True
        vx = speed * math.cos(pitch) * math.cos(yaw)
        vy = speed * math.cos(pitch) * math.sin(yaw)
        vz = speed * math.sin(pitch)
        prev = muzzle
        for dt in (0.06, 0.12, 0.18, 0.24):
            nxt = (
                muzzle[0] + vx * dt,
                muzzle[1] + vy * dt,
                muzzle[2] + vz * dt - 0.5 * gravity * dt * dt,
            )
            if not self.world.has_los(prev, nxt, step=1.5):
                return False
            prev = nxt
        return True

    def plan_attack_on_point(
        self,
        obs: Dict[str, Any],
        target_pos: Vec3,
        target_vel: Vec3,
        target_id: Optional[int],
        target_weapons: Optional[List[str]] = None,
        target_hp: float = 1400.0,
        is_mech: bool = True,
        allow_fire: bool = True,
        laser_shorted: bool = False,
        allow_laser_start: bool = True,
        target_cubes: int = 0,
    ) -> CombatPlan:
        """Choose best weapon slot, compute aim, and determine trigger state."""
        me = obs["self"]
        mx, my, mz = me["pos"]
        stat_mh = float(me["stats"].get("muzzle_height", self.base_muzzle_h))
        scaled_mh = self.base_muzzle_h * (1.0 + self.size_per_cube * int(me.get("cubes", 0)))
        muzzle_z = mz + max(stat_mh, scaled_mh)
        muzzle = (mx, my, muzzle_z)
        cur_aim = me["aim"]
        energy = float(me["energy"])
        weapons = me["weapons"]
        dist = math.dist(muzzle, target_pos)
        has_los = self._cached_los(int(obs["tick"]), target_id, muzzle, target_pos)

        target_r = (2.5 * (1.0 + self.size_per_cube * target_cubes)) if is_mech else 1.6
        lead_min_z: Optional[float] = (
            (self.world.terrain_height(target_pos[0], target_pos[1]) + 1.5)
            if (self.world.loaded and is_mech)
            else None
        )

        fire = [False, False]
        chosen_aim: Optional[Tuple[float, float]] = None
        chosen_kind: Optional[str] = None

        # Identify available weapon slots by kind
        slot_by_kind: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        for i, w in enumerate(weapons):
            if w is not None:
                slot_by_kind[w["kind"]] = (i, w)

        # 0. If a laser is ALREADY charging (and not shorted), dual-fire gun while charging & release cleanly!
        if not laser_shorted and "laser" in slot_by_kind and float(slot_by_kind["laser"][1].get("charge", 0.0)) > 0.0:
            idx, lw = slot_by_kind["laser"]
            stats = lw["stats"]
            charge = float(lw.get("charge", 0.0))
            full_charge = float(stats.get("charge", 0.75))
            l_range = float(stats.get("range", 280.0))

            hitscan_pt = (
                target_pos[0] + target_vel[0] * 0.05,
                target_pos[1] + target_vel[1] * 0.05,
                target_pos[2],
            )
            raw_h = aim_angles(muzzle, hitscan_pt)
            hitscan_aim = (raw_h[0], self._clamp_pitch(raw_h[1]))
            chosen_aim = hitscan_aim
            chosen_kind = "laser"

            yaw_tol = max(0.028, min(0.065, (target_r + 0.65) / max(15.0, dist)))
            pitch_tol = 0.075
            on_target = abs(angle_diff(hitscan_aim[0], cur_aim[0])) <= yaw_tol and abs(hitscan_aim[1] - cur_aim[1]) <= pitch_tol
            kill_charge = full_charge * 0.45 if target_hp <= 200.0 else full_charge * 0.88

            ready_release = (
                allow_fire
                and has_los
                and dist <= l_range
                and ((charge >= kill_charge and on_target) or charge >= full_charge + 0.35)
            )
            if ready_release:
                fire[idx] = False  # RELEASE triggers the hitscan beam!
            elif has_los and allow_fire and charge < full_charge + 0.4:
                fire[idx] = True   # Keep charging until aligned
                if "gun" in slot_by_kind:
                    g_idx, gw = slot_by_kind["gun"]
                    g_stats = gw["stats"]
                    g_speed = float(g_stats.get("speed", 252.0))
                    g_range = float(g_stats.get("range", 230.0))
                    gun_ready = dist <= g_range - 2.0 and gw.get("ammo", 0) > 0 and float(gw.get("reload", 0.0)) <= 0.0
                    g_lead = solve_lead_point(muzzle, target_pos, target_vel, g_speed, min_z=lead_min_z)
                    g_raw = aim_angles(muzzle, g_lead)
                    g_pitch_ok = (self.min_pitch - 0.03) <= g_raw[1] <= (self.max_pitch + 0.03)
                    g_aim = (g_raw[0], self._clamp_pitch(g_raw[1]))
                    if gun_ready and g_pitch_ok and charge < full_charge * 0.72 and target_hp > 200.0:
                        chosen_aim = g_aim
                    g_tol = max(0.045, min(0.14, (target_r + 1.4) / max(12.0, dist)))
                    if (
                        gun_ready
                        and g_pitch_ok
                        and abs(angle_diff(g_aim[0], cur_aim[0])) <= g_tol
                        and abs(g_aim[1] - cur_aim[1]) <= 0.12
                    ):
                        fire[g_idx] = True
            else:
                fire[idx] = False  # Release/cancel if lost LoS
            return CombatPlan(
                aim=[chosen_aim[0], chosen_aim[1]],
                fire=fire,
                reload=False,
                target_id=target_id,
                target_pos=target_pos,
                weapon_kind=chosen_kind,
            )

        # 1. Close range: Lightning + velocity-led Gun dual-fire
        if "lightning" in slot_by_kind and has_los and allow_fire:
            idx, w = slot_by_kind["lightning"]
            stats = w["stats"]
            l_range = float(stats.get("range", 45.0))
            l_cost = float(stats.get("energy", 3.0))
            min_e = l_cost * 1.2 if target_hp <= 280.0 else max(l_cost * 3.0, 9.0)
            if dist <= l_range - 1.2 and energy >= min_e:
                direct_raw = aim_angles(muzzle, target_pos)
                direct_aim = (direct_raw[0], self._clamp_pitch(direct_raw[1]))
                chosen_aim = direct_aim
                chosen_kind = "lightning"
                cone_rad = math.radians(float(stats.get("cone_deg", 30.0))) * 0.45

                if "gun" in slot_by_kind:
                    g_idx, gw = slot_by_kind["gun"]
                    g_stats = gw["stats"]
                    g_speed = float(g_stats.get("speed", 252.0))
                    g_range = float(g_stats.get("range", 230.0))
                    gun_ready = dist <= g_range - 2.0 and gw.get("ammo", 0) > 0 and float(gw.get("reload", 0.0)) <= 0.0
                    g_lead = solve_lead_point(muzzle, target_pos, target_vel, g_speed, min_z=lead_min_z)
                    g_raw = aim_angles(muzzle, g_lead)
                    g_pitch_ok = (self.min_pitch - 0.03) <= g_raw[1] <= (self.max_pitch + 0.03)
                    g_aim = (g_raw[0], self._clamp_pitch(g_raw[1]))
                    if gun_ready and g_pitch_ok and abs(angle_diff(g_aim[0], direct_aim[0])) <= cone_rad * 0.70:
                        chosen_aim = g_aim
                    if (
                        gun_ready
                        and g_pitch_ok
                        and abs(angle_diff(g_aim[0], cur_aim[0])) <= 0.14
                        and abs(g_aim[1] - cur_aim[1]) <= 0.14
                    ):
                        fire[g_idx] = True

                if abs(angle_diff(direct_aim[0], cur_aim[0])) <= cone_rad:
                    fire[idx] = True
                return CombatPlan(
                    aim=[chosen_aim[0], chosen_aim[1]],
                    fire=fire,
                    reload=False,
                    target_id=target_id,
                    target_pos=target_pos,
                    weapon_kind=chosen_kind,
                )

        # 2. Mid/Long range: Laser (315 dmg hitscan, +60% cookoff vs mortar)
        if allow_laser_start and not laser_shorted and "laser" in slot_by_kind and is_mech and has_los and allow_fire:
            idx, w = slot_by_kind["laser"]
            stats = w["stats"]
            l_range = float(stats.get("range", 280.0))
            cd = float(w.get("cooldown", 0.0))
            prefer_laser = dist > 52.0 or (target_weapons and "mortar" in target_weapons)
            gun_quick_finish = (
                target_hp <= 95.0
                and dist <= 140.0
                and "gun" in slot_by_kind
                and slot_by_kind["gun"][1].get("ammo", 0) > 0
                and float(slot_by_kind["gun"][1].get("reload", 0.0)) <= 0.0
            )
            if cd <= 0.05 and dist <= l_range - 5.0 and prefer_laser and not gun_quick_finish:
                raw_aim = aim_angles(muzzle, target_pos)
                pitch_in_bounds = (self.min_pitch - 0.03) <= raw_aim[1] <= (self.max_pitch + 0.03)
                aim = (raw_aim[0], self._clamp_pitch(raw_aim[1]))
                chosen_aim = aim
                chosen_kind = "laser"
                if allow_fire and pitch_in_bounds and abs(angle_diff(aim[0], cur_aim[0])) <= 0.25:
                    fire[idx] = True  # Start charging!
                    if "gun" in slot_by_kind:
                        g_idx, gw = slot_by_kind["gun"]
                        g_stats = gw["stats"]
                        g_speed = float(g_stats.get("speed", 252.0))
                        g_range = float(g_stats.get("range", 230.0))
                        if dist <= g_range - 2.0 and gw.get("ammo", 0) > 0 and float(gw.get("reload", 0.0)) <= 0.0:
                            g_lead = solve_lead_point(muzzle, target_pos, target_vel, g_speed, min_z=lead_min_z)
                            g_raw = aim_angles(muzzle, g_lead)
                            g_pitch_ok = (self.min_pitch - 0.03) <= g_raw[1] <= (self.max_pitch + 0.03)
                            g_aim = (g_raw[0], self._clamp_pitch(g_raw[1]))
                            g_tol = min(0.16, max(0.032, (target_r + 1.35) / max(10.0, dist)))
                            if g_pitch_ok and abs(angle_diff(g_aim[0], cur_aim[0])) <= g_tol and abs(g_aim[1] - cur_aim[1]) <= 0.12:
                                fire[g_idx] = True
                return CombatPlan(
                    aim=[chosen_aim[0], chosen_aim[1]],
                    fire=fire,
                    reload=False,
                    target_id=target_id,
                    target_pos=target_pos,
                    weapon_kind=chosen_kind,
                )

        # 3. Primary reliable DPS: Gun with 3D velocity lead (fall through to mortar if reloading/empty/out of range)
        gun_fallback_aim: Optional[Tuple[float, float]] = None
        if "gun" in slot_by_kind and has_los:
            idx, w = slot_by_kind["gun"]
            stats = w["stats"]
            g_speed = float(stats.get("speed", 252.0))
            g_range = float(stats.get("range", 230.0))
            lead_pt = solve_lead_point(muzzle, target_pos, target_vel, g_speed, min_z=lead_min_z)
            raw_aim = aim_angles(muzzle, lead_pt)
            pitch_in_bounds = (self.min_pitch - 0.03) <= raw_aim[1] <= (self.max_pitch + 0.03)
            aim = (raw_aim[0], self._clamp_pitch(raw_aim[1]))
            gun_ready = dist <= g_range - 2.0 and w.get("ammo", 0) > 0 and float(w.get("reload", 0.0)) <= 0.0
            if gun_ready:
                chosen_aim = aim
                chosen_kind = "gun"
                if allow_fire and pitch_in_bounds:
                    tol = min(0.16, max(0.032, (target_r + 1.35) / max(10.0, dist)))
                    if abs(angle_diff(aim[0], cur_aim[0])) <= tol and abs(aim[1] - cur_aim[1]) <= 0.12:
                        fire[idx] = True
                return CombatPlan(
                    aim=[chosen_aim[0], chosen_aim[1]],
                    fire=fire,
                    reload=False,
                    target_id=target_id,
                    target_pos=target_pos,
                    weapon_kind=chosen_kind,
                )
            gun_fallback_aim = aim

        # 4. Mortar fallback (when gun is reloading/out of range, or lobbing over buildings)
        if "mortar" in slot_by_kind and allow_fire:
            idx, w = slot_by_kind["mortar"]
            stats = w["stats"]
            m_speed = float(stats.get("speed", 72.0))
            m_grav = float(stats.get("gravity", 22.0))
            m_cd = float(w.get("cooldown", 0.0))
            m_muzzle = (mx, my, mz + 9.5)
            if dist >= 18.0 and m_cd <= 0.1:
                sol = solve_mortar_aim(
                    m_muzzle,
                    target_pos,
                    target_vel,
                    m_speed,
                    m_grav,
                    high_arc=not has_los,
                    min_pitch=self.min_pitch,
                    max_pitch=self.max_pitch,
                )
                if sol is not None and self._mortar_launch_clear(m_muzzle, sol[0], sol[1], m_speed, m_grav):
                    chosen_aim = sol
                    chosen_kind = "mortar"
                    if abs(angle_diff(chosen_aim[0], cur_aim[0])) <= 0.08 and abs(chosen_aim[1] - cur_aim[1]) <= 0.08:
                        fire[idx] = True
                    return CombatPlan(
                        aim=[chosen_aim[0], chosen_aim[1]],
                        fire=fire,
                        reload=False,
                        target_id=target_id,
                        target_pos=target_pos,
                        weapon_kind=chosen_kind,
                    )

        # Default: face target (or gun lead aim) to protect rear armor while reloading/on cooldown
        aim = gun_fallback_aim if gun_fallback_aim is not None else aim_angles(muzzle, target_pos)
        return CombatPlan(
            aim=[aim[0], self._clamp_pitch(aim[1])],
            fire=[False, False],
            reload=False,
            target_id=target_id,
            target_pos=target_pos,
            weapon_kind=None,
        )

    def should_early_reload(self, obs: Dict[str, Any], in_combat: bool) -> bool:
        """Reload gun opportunistically when out of combat and magazine is below half."""
        if in_combat:
            return False
        for w in obs["self"]["weapons"]:
            if w is not None and w["kind"] == "gun":
                ammo = int(w.get("ammo", 0))
                mag = int(w["stats"].get("mag", 12))
                if 0 < ammo <= mag // 2 and float(w.get("reload", 0.0)) <= 0.0:
                    return True
        return False
