"""World and geometry model: terrain, building prisms, spatial index, 2D navgrid, and 3D LoS."""
from __future__ import annotations

import array
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def _point_in_ring(x: float, y: float, ring: Sequence[float]) -> bool:
    """Ray-casting point-in-polygon test for flat [x0, y0, x1, y1, ...] ring."""
    n = len(ring) // 2
    if n < 3:
        return False
    inside = False
    j = n - 1
    xj = ring[2 * j]
    yj = ring[2 * j + 1]
    for i in range(n):
        xi = ring[2 * i]
        yi = ring[2 * i + 1]
        if (yi > y) != (yj > y):
            x_int = xj + (y - yj) * (xi - xj) / (yi - yj)
            if x < x_int:
                inside = not inside
        xj, yj = xi, yi
    return inside


def _dist_sq_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-12:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * dx
    cy = ay + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2


def _dist_sq_point_to_ring_edges(x: float, y: float, ring: Sequence[float]) -> float:
    """Minimum squared distance from (x, y) to the boundary segments of `ring`."""
    n = len(ring) // 2
    if n < 2:
        return 1e18
    best = 1e18
    xj = ring[2 * (n - 1)]
    yj = ring[2 * (n - 1) + 1]
    for i in range(n):
        xi = ring[2 * i]
        yi = ring[2 * i + 1]
        d2 = _dist_sq_point_to_segment(x, y, xj, yj, xi, yi)
        if d2 < best:
            best = d2
        xj, yj = xi, yi
    return best


def _point_near_ring(x: float, y: float, ring: Sequence[float], radius: float) -> bool:
    """True if (x, y) is inside ring or within `radius` meters of its boundary."""
    if _point_in_ring(x, y, ring):
        return True
    return _dist_sq_point_to_ring_edges(x, y, ring) <= radius * radius


class Prism:
    __slots__ = (
        "kind",
        "base",
        "top",
        "ring",
        "holes",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
    )

    def __init__(
        self,
        kind: str,
        base: float,
        top: float,
        ring: List[float],
        holes: List[List[float]],
    ):
        self.kind = kind
        self.base = base
        self.top = top
        self.ring = ring
        self.holes = holes
        xs = ring[0::2]
        ys = ring[1::2]
        self.min_x = min(xs) if xs else 0.0
        self.max_x = max(xs) if xs else 0.0
        self.min_y = min(ys) if ys else 0.0
        self.max_y = max(ys) if ys else 0.0

    def contains_xy(self, x: float, y: float) -> bool:
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            return False
        if not _point_in_ring(x, y, self.ring):
            return False
        for h in self.holes:
            if _point_in_ring(x, y, h):
                return False
        return True

    def blocks_cylinder_xy(self, x: float, y: float, radius: float) -> bool:
        if (
            x + radius < self.min_x
            or x - radius > self.max_x
            or y + radius < self.min_y
            or y - radius > self.max_y
        ):
            return False
        if not _point_near_ring(x, y, self.ring, radius):
            return False
        r2 = radius * radius
        for h in self.holes:
            # Strictly inside a hole with `radius` clearance from the hole's inner walls -> open space
            if _point_in_ring(x, y, h) and _dist_sq_point_to_ring_edges(x, y, h) > r2:
                return False
        return True


class WorldModel:
    """Static arena model with O(1) terrain lookup, spatial-hashed prisms, 3D LoS, and 2D navgrid."""

    HASH_CELL = 32.0
    NAV_CELL = 7.0

    def __init__(self, map_dir: Optional[str] = None, mech_radius: float = 2.5, mech_height: float = 9.0, step_up: float = 1.2):
        self.mech_radius = mech_radius
        self.mech_height = mech_height
        self.step_up = step_up
        self.loaded = False

        # Defaults (overwritten when world.json loads)
        self.min_x = -512.0
        self.max_x = 555.0
        self.min_y = -333.0
        self.max_y = 388.0

        # Terrain grid
        self.t_min_x = -513.0
        self.t_min_y = -334.0
        self.t_step = 1.0
        self.t_nx = 0
        self.t_ny = 0
        self.t_scale = 0.01
        self.terrain: Optional[array.array] = None
        self.max_terrain_z: float = 100.0

        # Prisms and spatial hash
        self.prisms: List[Prism] = []
        self._hash: Dict[Tuple[int, int], List[int]] = {}

        # 2D navgrid
        self.nav_nx = 0
        self.nav_ny = 0
        self.nav_min_x = self.min_x
        self.nav_min_y = self.min_y
        self.walkable: bytearray = bytearray()
        self.cover_adj: bytearray = bytearray()
        self.escape_deg: bytearray = bytearray()
        self.cell_z: array.array = array.array("f")
        self.elev_adv: array.array = array.array("f")
        self.narrow_choke: bytearray = bytearray()

        if map_dir:
            self.load(map_dir)

    def load(self, map_dir: str) -> bool:
        world_path = os.path.join(map_dir, "v2", "world.json")
        terrain_path = os.path.join(map_dir, "v2", "terrain.i16")
        if not os.path.isfile(world_path):
            return False

        with open(world_path, "r", encoding="utf-8") as f:
            w = json.load(f)

        bounds = w.get("bounds") or {}
        self.min_x = float(bounds.get("minX", self.min_x))
        self.max_x = float(bounds.get("maxX", self.max_x))
        self.min_y = float(bounds.get("minY", self.min_y))
        self.max_y = float(bounds.get("maxY", self.max_y))

        tmeta = w.get("terrain") or {}
        self.t_min_x = float(tmeta.get("minX", -513.0))
        self.t_min_y = float(tmeta.get("minY", -334.0))
        self.t_step = float(tmeta.get("step", 1.0))
        self.t_nx = int(tmeta.get("nx", 0))
        self.t_ny = int(tmeta.get("ny", 0))
        self.t_scale = float(tmeta.get("scale", 0.01))

        if os.path.isfile(terrain_path) and self.t_nx > 0 and self.t_ny > 0:
            arr = array.array("h")
            with open(terrain_path, "rb") as tf:
                arr.frombytes(tf.read())
            if len(arr) >= self.t_nx * self.t_ny:
                self.terrain = arr
                self.max_terrain_z = max(arr) * self.t_scale

        self.prisms = []
        self._hash = {}
        for idx, p in enumerate(w.get("prisms") or []):
            ring = p.get("ring") or []
            if len(ring) < 6:
                continue
            prism = Prism(
                kind=str(p.get("kind", "building")),
                base=float(p.get("base", -100.0)),
                top=float(p.get("top", 0.0)),
                ring=[float(v) for v in ring],
                holes=[[float(v) for v in h] for h in (p.get("holes") or []) if len(h) >= 6],
            )
            self.prisms.append(prism)
            self._index_prism(len(self.prisms) - 1, prism)

        self._build_navgrid()
        self.loaded = True
        return True

    def load_synthetic(
        self,
        bounds: Tuple[float, float, float, float],
        prisms: Sequence[Prism],
        flat_z: float = 0.0,
    ) -> None:
        """Load a lightweight synthetic world for unit and scenario tests."""
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.terrain = None
        self._flat_z = flat_z
        self.prisms = list(prisms)
        self._hash = {}
        for idx, prism in enumerate(self.prisms):
            self._index_prism(idx, prism)
        self._build_navgrid()
        self.loaded = True

    def _index_prism(self, idx: int, prism: Prism) -> None:
        pad = self.mech_radius + 1.5
        bx0 = int(math.floor((prism.min_x - pad) / self.HASH_CELL))
        bx1 = int(math.floor((prism.max_x + pad) / self.HASH_CELL))
        by0 = int(math.floor((prism.min_y - pad) / self.HASH_CELL))
        by1 = int(math.floor((prism.max_y + pad) / self.HASH_CELL))
        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                key = (bx, by)
                bucket = self._hash.get(key)
                if bucket is None:
                    self._hash[key] = [idx]
                else:
                    bucket.append(idx)

    def terrain_height(self, x: float, y: float) -> float:
        if self.terrain is None:
            return getattr(self, "_flat_z", 0.0)
        fx = (x - self.t_min_x) / self.t_step
        fy = (y - self.t_min_y) / self.t_step
        if fx < 0.0:
            fx = 0.0
        elif fx > self.t_nx - 1.001:
            fx = self.t_nx - 1.001
        if fy < 0.0:
            fy = 0.0
        elif fy > self.t_ny - 1.001:
            fy = self.t_ny - 1.001
        i = int(fx)
        j = int(fy)
        tx = fx - i
        ty = fy - j
        row0 = j * self.t_nx + i
        row1 = row0 + self.t_nx
        t = self.terrain
        h00 = t[row0]
        h10 = t[row0 + 1]
        h01 = t[row1]
        h11 = t[row1 + 1]
        h0 = h00 + (h10 - h00) * tx
        h1 = h01 + (h11 - h01) * tx
        return (h0 + (h1 - h0) * ty) * self.t_scale

    def in_bounds(self, x: float, y: float, margin: float = 4.0) -> bool:
        return (
            self.min_x + margin <= x <= self.max_x - margin
            and self.min_y + margin <= y <= self.max_y - margin
        )

    def clamp_to_bounds(self, x: float, y: float, margin: float = 8.0) -> Vec2:
        cx = max(self.min_x + margin, min(self.max_x - margin, x))
        cy = max(self.min_y + margin, min(self.max_y - margin, y))
        return (cx, cy)

    def is_blocked_at(self, x: float, y: float, z: Optional[float] = None, clearance: float = 0.6) -> bool:
        """True if a mech cylinder standing at (x, y, z) collides with any solid prism."""
        if not self.in_bounds(x, y, margin=self.mech_radius + 0.5):
            return True
        if z is None:
            z = self.terrain_height(x, y)
        bx = int(math.floor(x / self.HASH_CELL))
        by = int(math.floor(y / self.HASH_CELL))
        bucket = self._hash.get((bx, by))
        if not bucket:
            return False
        rad = self.mech_radius + clearance
        head_z = z + self.mech_height
        step_z = z + self.step_up
        for idx in bucket:
            p = self.prisms[idx]
            # Walk under bridges/overhangs that clear our head; step over low curbs
            if p.base >= head_z + 0.2 or p.top <= step_z:
                continue
            if p.blocks_cylinder_xy(x, y, rad):
                return True
        return False

    def surface_height(self, x: float, y: float) -> float:
        """Highest walkable surface (terrain or roof top) at (x, y)."""
        h = self.terrain_height(x, y)
        bx = int(math.floor(x / self.HASH_CELL))
        by = int(math.floor(y / self.HASH_CELL))
        bucket = self._hash.get((bx, by))
        if bucket:
            for idx in bucket:
                p = self.prisms[idx]
                if p.top > h and p.contains_xy(x, y):
                    h = p.top
        return h

    def _build_navgrid(self) -> None:
        cell = self.NAV_CELL
        self.nav_min_x = self.min_x
        self.nav_min_y = self.min_y
        self.nav_nx = max(1, int(math.ceil((self.max_x - self.min_x) / cell)))
        self.nav_ny = max(1, int(math.ceil((self.max_y - self.min_y) / cell)))
        total = self.nav_nx * self.nav_ny
        walkable = bytearray(total)
        cover_adj = bytearray(total)
        escape_deg = bytearray(total)
        cell_z = array.array("f", [0.0]) * total
        elev_adv = array.array("f", [0.0]) * total
        narrow_choke = bytearray(total)

        half = cell * 0.5
        # tan(48 deg) ~= 1.1106 -> 3.33m over 3m = 333 int16 cm
        max_slope_cm = 333
        t_arr = self.terrain
        t_nx = self.t_nx
        t_ny = self.t_ny
        t_min_x = self.t_min_x
        t_min_y = self.t_min_y
        for gy in range(self.nav_ny):
            cy = self.nav_min_y + gy * cell + half
            row = gy * self.nav_nx
            tj = int(cy - t_min_y) if t_arr is not None else 0
            for gx in range(self.nav_nx):
                cx = self.nav_min_x + gx * cell + half
                if self.is_blocked_at(cx, cy, clearance=0.5):
                    continue
                if t_arr is not None and 3 <= tj < t_ny - 3:
                    ti = int(cx - t_min_x)
                    if 3 <= ti < t_nx - 3:
                        idx0 = tj * t_nx + ti
                        h0 = t_arr[idx0]
                        if (
                            abs(t_arr[idx0 + 3] - h0) > max_slope_cm
                            or abs(t_arr[idx0 - 3] - h0) > max_slope_cm
                            or abs(t_arr[idx0 + 3 * t_nx] - h0) > max_slope_cm
                            or abs(t_arr[idx0 - 3 * t_nx] - h0) > max_slope_cm
                        ):
                            continue
                walkable[row + gx] = 1
                cell_z[row + gx] = self.terrain_height(cx, cy)

        # Mark cells adjacent to solid obstacles, narrow chokes, and 8-neighbor escape degree
        nx = self.nav_nx
        ny = self.nav_ny
        for gy in range(1, ny - 1):
            row = gy * nx
            for gx in range(1, nx - 1):
                idx = row + gx
                if not walkable[idx]:
                    continue
                w_l = walkable[idx - 1]
                w_r = walkable[idx + 1]
                w_d = walkable[idx - nx]
                w_u = walkable[idx + nx]
                deg = (
                    w_l
                    + w_r
                    + w_d
                    + w_u
                    + walkable[idx - nx - 1]
                    + walkable[idx - nx + 1]
                    + walkable[idx + nx - 1]
                    + walkable[idx + nx + 1]
                )
                escape_deg[idx] = deg
                if not w_l or not w_r or not w_d or not w_u:
                    cover_adj[idx] = 1
                if deg <= 2 or (not w_l and not w_r) or (not w_d and not w_u):
                    narrow_choke[idx] = 1

        # Compute local elevation advantage relative to surrounding walkable terrain
        offsets = (
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (4, 0), (-4, 0), (0, 4), (0, -4),
            (3, 3), (3, -3), (-3, 3), (-3, -3),
        )
        for gy in range(ny):
            row = gy * nx
            for gx in range(nx):
                idx = row + gx
                if not walkable[idx]:
                    continue
                z0 = cell_z[idx]
                acc = 0.0
                cnt = 0
                for dx, dy in offsets:
                    sx = gx + dx
                    sy = gy + dy
                    if 0 <= sx < nx and 0 <= sy < ny:
                        sidx = sy * nx + sx
                        if walkable[sidx]:
                            acc += cell_z[sidx]
                            cnt += 1
                if cnt > 0:
                    adv = z0 - (acc / cnt)
                    if adv < -6.0:
                        adv = -6.0
                    elif adv > 8.0:
                        adv = 8.0
                    elev_adv[idx] = adv

        self.walkable = walkable
        self.cover_adj = cover_adj
        self.escape_deg = escape_deg
        self.cell_z = cell_z
        self.elev_adv = elev_adv
        self.narrow_choke = narrow_choke

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.nav_min_x) / self.NAV_CELL)
        gy = int((y - self.nav_min_y) / self.NAV_CELL)
        if gx < 0:
            gx = 0
        elif gx >= self.nav_nx:
            gx = self.nav_nx - 1
        if gy < 0:
            gy = 0
        elif gy >= self.nav_ny:
            gy = self.nav_ny - 1
        return (gx, gy)

    def grid_to_world(self, gx: int, gy: int) -> Vec2:
        return (
            self.nav_min_x + (gx + 0.5) * self.NAV_CELL,
            self.nav_min_y + (gy + 0.5) * self.NAV_CELL,
        )

    def is_cell_walkable(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.nav_nx or gy < 0 or gy >= self.nav_ny:
            return False
        return bool(self.walkable[gy * self.nav_nx + gx])

    def cell_elevation_advantage(self, gx: int, gy: int) -> float:
        """O(1) local terrain elevation advantage at navgrid cell (gx, gy)."""
        if not self.elev_adv or gx < 0 or gx >= self.nav_nx or gy < 0 or gy >= self.nav_ny:
            return 0.0
        return float(self.elev_adv[gy * self.nav_nx + gx])

    def is_narrow_choke(self, gx: int, gy: int) -> bool:
        """O(1) check whether navgrid cell (gx, gy) is a narrow choke or dead-end."""
        if not self.narrow_choke or gx < 0 or gx >= self.nav_nx or gy < 0 or gy >= self.nav_ny:
            return False
        return bool(self.narrow_choke[gy * self.nav_nx + gx])

    def nearest_walkable_cell(self, x: float, y: float, max_ring: int = 8) -> Tuple[int, int]:
        gx, gy = self.world_to_grid(x, y)
        if self.is_cell_walkable(gx, gy):
            return (gx, gy)
        for r in range(1, max_ring + 1):
            best = None
            best_d2 = 1e18
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if self.is_cell_walkable(nx, ny):
                        wx, wy = self.grid_to_world(nx, ny)
                        d2 = (wx - x) ** 2 + (wy - y) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best = (nx, ny)
            if best is not None:
                return best
        return (gx, gy)

    def has_los(self, p0: Vec3, p1: Vec3, step: float = 3.5) -> bool:
        """Fast 3D line-of-sight check against solid prisms and terrain."""
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 2.0:
            return True

        # Collect candidate prisms along the 2D bounding box of the segment
        steps = max(2, int(dist / step))
        inv = 1.0 / steps
        checked_buckets = set()
        cand_prisms: List[Prism] = []
        seen_idx = set()
        hcell = self.HASH_CELL

        # Sample buckets along the ray
        b_steps = max(1, int(dist / (hcell * 0.5)))
        for k in range(b_steps + 1):
            t = k / b_steps
            bx = int(math.floor((x0 + dx * t) / hcell))
            by = int(math.floor((y0 + dy * t) / hcell))
            key = (bx, by)
            if key not in checked_buckets:
                checked_buckets.add(key)
                bucket = self._hash.get(key)
                if bucket:
                    for idx in bucket:
                        if idx not in seen_idx:
                            seen_idx.add(idx)
                            cand_prisms.append(self.prisms[idx])

        # Explicit barrel-clearance check at 2.5m (shots stop if barrel pokes into a wall)
        if dist > 3.0 and cand_prisms:
            t_b = 2.5 / dist
            bx_p = x0 + dx * t_b
            by_p = y0 + dy * t_b
            bz_p = z0 + dz * t_b
            for p in cand_prisms:
                if p.base <= bz_p <= p.top and p.contains_xy(bx_p, by_p):
                    return False

        # March interior points along segment
        check_terrain = self.terrain is not None
        max_tz = self.max_terrain_z + 0.5
        for k in range(1, steps):
            t = k * inv
            px = x0 + dx * t
            py = y0 + dy * t
            pz = z0 + dz * t
            if check_terrain and pz < max_tz and (k & 1 == 1):
                if pz < self.terrain_height(px, py) - 0.3:
                    return False
            for p in cand_prisms:
                if p.base <= pz <= p.top and p.contains_xy(px, py):
                    return False
        return True

    def line_walkable(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True if straight 2D line between two points stays on walkable navgrid cells."""
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        steps = max(2, int(dist / (self.NAV_CELL * 0.5)))
        for k in range(1, steps):
            t = k / steps
            gx, gy = self.world_to_grid(x0 + dx * t, y0 + dy * t)
            if not self.is_cell_walkable(gx, gy):
                return False
        return True

    def find_cover_candidates(
        self,
        my_pos: Vec3,
        threat_pos: Vec3,
        zone_center: Vec2,
        zone_radius: float,
        search_radius: float = 62.0,
        max_results: int = 6,
    ) -> List[Tuple[Vec2, int]]:
        """Find up to `max_results` walkable, zone-safe cover cells that break LoS from `threat_pos`."""
        if not self.loaded:
            return []
        mx, my, _ = my_pos
        tx, ty, tz = threat_pos
        t_eye = (tx, ty, tz + 7.5)
        cur_d_threat = math.hypot(mx - tx, my - ty)
        gx0, gy0 = self.world_to_grid(mx, my)
        r_cells = max(2, int(search_radius / self.NAV_CELL))
        zcx, zcy = zone_center
        margin = min(6.0, zone_radius * 0.25)
        safe_r2 = max(0.0, zone_radius - margin) ** 2

        candidates: List[Tuple[float, float, float, int]] = []
        for dy in range(-r_cells, r_cells + 1):
            gy = gy0 + dy
            if gy < 0 or gy >= self.nav_ny:
                continue
            row = gy * self.nav_nx
            for dx in range(-r_cells, r_cells + 1):
                gx = gx0 + dx
                if gx < 0 or gx >= self.nav_nx:
                    continue
                if not self.walkable[row + gx] or not self.cover_adj[row + gx]:
                    continue
                wx, wy = self.grid_to_world(gx, gy)
                if (wx - zcx) ** 2 + (wy - zcy) ** 2 > safe_r2:
                    continue
                d_me = math.hypot(wx - mx, wy - my)
                if d_me < 4.0 or d_me > search_radius:
                    continue
                d_threat = math.hypot(wx - tx, wy - ty)
                sep_gain = d_threat - cur_d_threat
                deg = self.escape_deg[row + gx]
                corner_bonus = 12.0 if 4 <= deg <= 6 else (-28.0 if deg <= 2 else 0.0)
                if self.narrow_choke and self.narrow_choke[row + gx] and deg > 2:
                    corner_bonus -= 16.0
                elev_bonus = max(-3.0, min(6.0, self.elev_adv[row + gx] * 1.2)) if self.elev_adv else 0.0
                score = sep_gain * 1.15 + d_threat * 0.35 - d_me * 0.55 + corner_bonus + elev_bonus
                candidates.append((score, wx, wy, deg))

        if not candidates:
            return []
        candidates.sort(key=lambda c: c[0], reverse=True)
        out: List[Tuple[Vec2, int]] = []
        for _, wx, wy, deg in candidates[:16]:
            wz = self.terrain_height(wx, wy)
            if not self.has_los(t_eye, (wx, wy, wz + 5.0), step=3.5):
                out.append(((wx, wy), deg))
                if len(out) >= max_results:
                    break
        return out

    def find_cover_point(
        self,
        my_pos: Vec3,
        threat_pos: Vec3,
        zone_center: Vec2,
        zone_radius: float,
        search_radius: float = 56.0,
    ) -> Optional[Vec2]:
        """Find a walkable cell inside the safe zone that breaks LoS from `threat_pos`."""
        cands = self.find_cover_candidates(
            my_pos, threat_pos, zone_center, zone_radius, search_radius=search_radius, max_results=1
        )
        return cands[0][0] if cands else None
