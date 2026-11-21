# Mech Royale — Experiment Log

## Official Practice Score History

| Candidate | Branch | Commit SHA | Official Practice Score | Notes |
| --- | --- | --- | ---: | --- |
| Official starter | `main` / `baseline/official-starter` | `14a4d1ef8a8dd0d8d37170b74f4feb95bb554991` | **1.15** | Untouched official Python starter (current best official score) |
| Defensive survival v1 | `baseline/survival-v1` | `eaca8b212e83f6986fb404efc6d26e6f385b3c1c` | **0.85** | Teammate's survival prototype (underperformed starter) |
| Tactical Survivor v2 (Iter 1) | `experiment/tactical-survivor-v2` | *See latest commit on branch* | *Pending manual submission* | Full modular map/nav/threat/combat/resource stack + 10 bug-hunter fixes |

---

## Iteration 1 — Modular Map-Aware Tactical Survivor v2 (`experiment/tactical-survivor-v2`)

### Hypothesis
The official starter (`1.15`) and defensive survival v1 (`0.85`) fail primarily because they:
1. Walk in straight lines into buildings and toward the crowded dead-center of `next_center`.
2. Lack 3D line-of-sight awareness, obstacle-avoiding pathfinding, and building cover selection.
3. Lack projectile velocity lead, proper laser charge/release handling, close-range lightning control, and projectile/laser-charge dodging.
4. Expose rear armor (`1.5×` damage) when rotating and fail to retreat behind cover to trigger out-of-combat HP regeneration (`45 HP/s`).

Replacing straight-line reactive logic with a modular map-aware navigation, threat-tracking, lead-aiming, cover-retreat, and resource-farming architecture should substantially improve survival time and expected placement while staying well under the 50 ms / 1-CPU Docker budget.

### Independent Review (`bug-hunter`) & Fixes Applied Before Finalizing Iter 1
An independent read-only `bug-hunter` audit caught 10 confirmed defects (1 Critical, 4 High, 4 Medium, 1 Low), all fixed and regression-tested:
- **BUG-001 (Critical):** Fixed tautological hole check in `Prism.blocks_cylinder_xy` using boundary-only `_dist_sq_point_to_ring_edges`, restoring walkability inside hollow structures (e.g., Ohio Stadium courtyard).
- **BUG-002 (High):** Added `closed` set in A* `plan_path`, capped `ThreatTracker.sounds` to top 24 recent pings, added AABB pre-checks in `danger_at`, and throttled stuck replans to >= 6 ticks.
- **BUG-003 (High):** Added linearly interpolated active zone circle `(active_cx, active_cy, active_r)` so endgame 1v1 (`next_radius == 0`) allows `COMBAT_KITE` inside the active circle and early-game loot/cover isn't restricted to `next_radius` when plenty of time remains.
- **BUG-004 (High):** Allowed Gun branch to fall through to Mortar fallback when gun is reloading, empty, or out of range.
- **BUG-005 (High):** Passed config `min_pitch`/`max_pitch` into `solve_mortar_aim` and added 14 m launch-arc wall clearance check to prevent mortar self-detonation near cover.
- **BUG-006 (Medium):** Filtered rooftop loot using local ground height (`terrain_height(px, py)`) instead of current mech Z so campus slopes don't discard reachable loot.
- **BUG-007 (Medium):** Added pitch tolerance to lightning+gun dual-fire, tracked `effect == "short"` to bypass shorted lasers, and scaled lightning energy reserve from weapon stats.
- **BUG-008 (Medium):** Fixed wall avoidance when both left/right are clear (steers toward goal side instead of jumping into wall), suppressed stuck false-positives during kiting/laser-charging/silent-walk, and prevented airborne dodge-jetting.
- **BUG-009 (Medium):** Tightened combat `has_los` step to 2.0 m and added explicit 2.5 m barrel-clearance check.
- **BUG-010 (Low):** Prevented early gun reload from interrupting crate/NPC fights and added ballistic Z check for mortar projectile dodging.

### Verification & Performance Results (Iteration 1 Final)
- **Unit & 12-scenario synthetic tests (`tests/test_bot.py`):** `20/20 PASS` (synthetic/local verification only; not official match scores).
- **Host conformance check (`python check/check.py -- python -u example_bot.py`):**
  - `PASS` — `300/300` on time, `0` late, `p50 0.8 ms`, `p90 2.3 ms`, `p99 11.5 ms`, `max 15.5 ms`, `0` wrong tick, `0` bad lines.
- **Docker `linux/amd64` build:** `PASS` (exit `0`, build context `203.00 kB`).
- **Docker sandbox check (`1 CPU`, `512 MB`, `--network none`, `--read-only`, `--tmpfs /tmp:rw,size=64m`, `--user 65534:65534`, `/arena:ro`):**
  - `PASS` — `ready in 5926 ms`, `300/300` on time, `0` late, `p50 5.0 ms`, `p90 12.5 ms`, `p99 31.0 ms`, `max 36.8 ms`, `0` wrong tick, `0` bad lines, `300` debug drawings (`0` over caps).
