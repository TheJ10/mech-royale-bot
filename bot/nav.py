"""Navigation planner: threat-aware A* pathfinding, string-pulling, and unstuck/ledge handling."""
from __future__ import annotations

import heapq
import math
import time
from typing import Callable, List, Optional, Tuple

from bot.world import Vec2, Vec3, WorldModel

_NEIGHBORS_8 = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, 1.41421356),
    (1, -1, 1.41421356),
    (-1, 1, 1.41421356),
    (-1, -1, 1.41421356),
)


class Navigator:
    """Plans obstacle-free, danger-aware paths on the WorldModel navgrid."""

    def __init__(self, world: WorldModel):
        self.world = world
        self.path: List[Vec2] = []
        self.last_goal: Optional[Vec2] = None
        self.last_plan_tick: int = -100
        self.pos_history: List[Tuple[int, float, float, float]] = []
        self.stuck_until_tick: int = -1
        self.stuck_side: float = 1.0
        self.unstuck_jet_until_tick: int = -1

    def plan_path(
        self,
        start: Vec2,
        goal: Vec2,
        danger_fn: Optional[Callable[[float, float], float]] = None,
        danger_weight: float = 1.0,
        max_expansions: int = 220,
        danger_sources: Optional[List[Tuple[float, float, float, float, float]]] = None,
        time_budget_s: float = 0.010,
    ) -> List[Vec2]:
        """Compute an A* path from start to goal on the navgrid, then smooth it."""
        if not self.world.loaded:
            return [goal]

        gx_clamped, gy_clamped = self.world.clamp_to_bounds(goal[0], goal[1], margin=6.0)
        sx, sy = self.world.nearest_walkable_cell(start[0], start[1])
        tx, ty = self.world.nearest_walkable_cell(gx_clamped, gy_clamped)

        if (sx, sy) == (tx, ty):
            return [(gx_clamped, gy_clamped)]

        use_danger = danger_weight > 0.01 and (bool(danger_sources) or danger_fn is not None)
        if not use_danger and self.world.line_walkable(start[0], start[1], gx_clamped, gy_clamped):
            return [(gx_clamped, gy_clamped)]

        nx = self.world.nav_nx
        ny = self.world.nav_ny
        walkable = self.world.walkable
        cell = self.world.NAV_CELL
        min_x = self.world.nav_min_x
        min_y = self.world.nav_min_y

        start_idx = sy * nx + sx
        goal_idx = ty * nx + tx

        g_score = {start_idx: 0.0}
        danger_cache: dict[int, float] = {}
        closed: set[int] = set()
        came_from = {}
        h0 = math.hypot(tx - sx, ty - sy)
        heap: List[Tuple[float, int]] = [(h0, start_idx)]
        best_idx = start_idx
        best_h = h0
        expansions = 0
        sqrt = math.sqrt
        h_weight = 1.55 if danger_weight >= 3.0 else 1.35
        t_start = time.perf_counter()

        while heap and expansions < max_expansions:
            if (expansions & 31) == 31 and (time.perf_counter() - t_start) > time_budget_s:
                break
            _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal_idx:
                best_idx = cur
                break
            expansions += 1
            cx = cur % nx
            cy = cur // nx
            cur_g = g_score[cur]

            for dx, dy, step_cost in _NEIGHBORS_8:
                nx_c = cx + dx
                ny_c = cy + dy
                if nx_c < 0 or nx_c >= nx or ny_c < 0 or ny_c >= ny:
                    continue
                n_idx = ny_c * nx + nx_c
                if n_idx in closed or not walkable[n_idx]:
                    continue
                if dx != 0 and dy != 0:
                    if not walkable[cy * nx + nx_c] or not walkable[ny_c * nx + cx]:
                        continue

                extra = 0.0
                if use_danger:
                    cached = danger_cache.get(n_idx)
                    if cached is None:
                        wx = min_x + (nx_c + 0.5) * cell
                        wy = min_y + (ny_c + 0.5) * cell
                        if danger_sources is not None:
                            dval = 0.0
                            for sx_p, sy_p, rad, r_sq, w_p in danger_sources:
                                ddx = sx_p - wx
                                if ddx > rad or ddx < -rad:
                                    continue
                                ddy = sy_p - wy
                                if ddy > rad or ddy < -rad:
                                    continue
                                d2 = ddx * ddx + ddy * ddy
                                if d2 < r_sq:
                                    prox = (rad - sqrt(d2)) / rad
                                    dval += w_p * prox * prox
                            cached = danger_weight * dval
                        else:
                            cached = danger_weight * danger_fn(wx, wy)
                        danger_cache[n_idx] = cached
                    extra = cached

                tentative_g = cur_g + step_cost + extra
                prev_g = g_score.get(n_idx)
                if prev_g is None or tentative_g < prev_g:
                    g_score[n_idx] = tentative_g
                    came_from[n_idx] = cur
                    h = math.hypot(tx - nx_c, ty - ny_c)
                    if h < best_h:
                        best_h = h
                        best_idx = n_idx
                    heapq.heappush(heap, (tentative_g + h * h_weight, n_idx))

        # Reconstruct path
        raw: List[Vec2] = []
        cur = best_idx
        while cur in came_from:
            raw.append(self.world.grid_to_world(cur % nx, cur // nx))
            cur = came_from[cur]
        raw.reverse()

        if best_idx == goal_idx:
            if raw:
                raw[-1] = (gx_clamped, gy_clamped)
            else:
                raw.append((gx_clamped, gy_clamped))

        if not raw:
            return [(gx_clamped, gy_clamped)]

        # String-pull smoothing when danger_weight is modest
        if len(raw) <= 2:
            return raw
        return self._smooth_path(start, raw)

    def _smooth_path(self, start: Vec2, pts: List[Vec2]) -> List[Vec2]:
        smoothed: List[Vec2] = []
        anchor = start
        i = 0
        n = len(pts)
        while i < n:
            # Greedily jump as far ahead as line_walkable allows (capped at 6 hops so danger detours are preserved)
            farthest = i
            limit = min(n, i + 6)
            for j in range(limit - 1, i, -1):
                if self.world.line_walkable(anchor[0], anchor[1], pts[j][0], pts[j][1]):
                    farthest = j
                    break
            smoothed.append(pts[farthest])
            anchor = pts[farthest]
            i = farthest + 1
        return smoothed

    def update_and_steer(
        self,
        tick: int,
        pos: Vec3,
        vel: Vec3,
        grounded: bool,
        energy: float,
        goal: Vec2,
        danger_fn: Optional[Callable[[float, float], float]] = None,
        danger_weight: float = 1.0,
        want_boost: bool = False,
        dodge_vec: Optional[Vec2] = None,
        suppress_stuck: bool = False,
        danger_sources: Optional[List[Tuple[float, float, float, float, float]]] = None,
    ) -> Tuple[List[float], bool, bool, Vec2]:
        """Return (move_xy, boost, jump, active_waypoint)."""
        x, y, z = pos
        self.pos_history.append((tick, x, y, z))
        if len(self.pos_history) > 12:
            self.pos_history.pop(0)

        goal = self.world.clamp_to_bounds(goal[0], goal[1], margin=6.0)
        dist_to_goal = math.hypot(goal[0] - x, goal[1] - y)

        # Detect stuck state: minimal movement over last ~6 decision ticks while trying to move
        is_stuck = False
        if not suppress_stuck and dist_to_goal > 6.0 and len(self.pos_history) >= 6:
            _, ox, oy, _ = self.pos_history[-6]
            moved = math.hypot(x - ox, y - oy)
            if moved < 1.8:
                is_stuck = True
                if tick > self.stuck_until_tick:
                    self.stuck_until_tick = tick + 10
                    self.stuck_side = -self.stuck_side
                    if energy >= 35.0:
                        self.unstuck_jet_until_tick = tick + 8

        # BH-004 + BH-101: Clear stale distant path when goal is within 5m and reset replan tick
        if dist_to_goal <= 5.0:
            self.path = [goal]
            self.last_goal = goal
            self.last_plan_tick = -100
        else:
            # Replan A* when needed (throttle stuck-triggered replans to >= 6 ticks)
            need_replan = (
                not self.path
                or self.last_goal is None
                or math.hypot(goal[0] - self.last_goal[0], goal[1] - self.last_goal[1]) > 16.0
                or (tick - self.last_plan_tick) >= 24
                or (is_stuck and (tick - self.last_plan_tick) >= 6)
            )
            if need_replan:
                self.path = self.plan_path(
                    (x, y),
                    goal,
                    danger_fn=danger_fn,
                    danger_weight=danger_weight,
                    danger_sources=danger_sources,
                )
                self.last_goal = goal
                self.last_plan_tick = tick

        # Advance waypoints already reached
        while len(self.path) > 1 and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < 7.0:
            self.path.pop(0)

        waypoint = self.path[0] if self.path else goal
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        wp_dist = math.hypot(dx, dy)

        if wp_dist < 1e-3:
            mx, my = 0.0, 0.0
        else:
            mx, my = dx / wp_dist, dy / wp_dist

        # Blend dodge vector if incoming projectile / charging laser
        if dodge_vec is not None:
            mx = mx * 0.35 + dodge_vec[0] * 0.85
            my = my * 0.35 + dodge_vec[1] * 0.85
            norm = math.hypot(mx, my)
            if norm > 1e-6:
                mx /= norm
                my /= norm

        # Apply tangential sidestep if actively unstucking
        jump = False
        boost = want_boost
        if tick <= self.stuck_until_tick:
            # Rotate direction by 65 deg to slide around building corner
            ang = math.atan2(my, mx) + self.stuck_side * 1.15
            mx = math.cos(ang)
            my = math.sin(ang)
            if grounded and energy > 25.0:
                jump = True
            if not grounded and tick <= self.unstuck_jet_until_tick and energy > 20.0:
                boost = True

        # Check immediate wall ahead (3.5 m ahead) and slide/jump if needed
        if self.world.loaded and wp_dist > 4.0:
            ahead_x = x + mx * 4.5
            ahead_y = y + my * 4.5
            if self.world.is_blocked_at(ahead_x, ahead_y, z, clearance=0.3):
                # Try steering left or right 45 degrees
                base_ang = math.atan2(my, mx)
                left_ok = not self.world.is_blocked_at(
                    x + math.cos(base_ang + 0.85) * 4.5,
                    y + math.sin(base_ang + 0.85) * 4.5,
                    z,
                    clearance=0.3,
                )
                right_ok = not self.world.is_blocked_at(
                    x + math.cos(base_ang - 0.85) * 4.5,
                    y + math.sin(base_ang - 0.85) * 4.5,
                    z,
                    clearance=0.3,
                )
                if left_ok and not right_ok:
                    mx, my = math.cos(base_ang + 0.85), math.sin(base_ang + 0.85)
                elif right_ok and not left_ok:
                    mx, my = math.cos(base_ang - 0.85), math.sin(base_ang - 0.85)
                elif left_ok and right_ok:
                    # Pick side closer to final goal instead of jumping straight into the wall
                    lx, ly = math.cos(base_ang + 0.85), math.sin(base_ang + 0.85)
                    rx, ry = math.cos(base_ang - 0.85), math.sin(base_ang - 0.85)
                    gdx, gdy = goal[0] - x, goal[1] - y
                    if (lx * gdx + ly * gdy) >= (rx * gdx + ry * gdy):
                        mx, my = lx, ly
                    else:
                        mx, my = rx, ry
                elif grounded and energy > 30.0:
                    jump = True

        # Gentle slowdown only when very close to final goal with no dodge
        if dist_to_goal < 2.5 and dodge_vec is None:
            scale = dist_to_goal / 2.5
            mx *= scale
            my *= scale

        # Never jet-boost in the air unintentionally unless actively unstuck-jetting
        if not grounded and tick > self.unstuck_jet_until_tick:
            boost = False

        # Protect minimum emergency energy reserve unless urgent
        if boost and energy < 12.0 and dodge_vec is None:
            boost = False

        return ([mx, my], boost, jump, waypoint)
