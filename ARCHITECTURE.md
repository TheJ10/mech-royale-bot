# Mech Royale Bot Architecture (`experiment/tactical-survivor-v2`)

## Overview

The bot is structured as a modular, stateful decision pipeline using only the Python standard library (zero external dependencies, fast startup, <15 ms P99 latency in 1-CPU `linux/amd64` Docker).

```text
example_bot.py            # Entrypoint (unchanged Dockerfile ENTRYPOINT)
mechfight.py              # Official mechfight/1 stdin/stdout wrapper
bot/
  world.py                # 1. Static geometry, terrain, spatial-hashed prisms, 2D navgrid, 3D LoS
  nav.py                  # 2. Threat-weighted A* pathfinding, string-pulling, unstuck/ledge control
  threat.py               # 3. Persistent enemy memory, sound pings, rear-armor alert, projectile dodge
  combat.py               # 4. Target scoring, 3D gun lead, laser charge/release, lightning, mortar ballistics
  resources.py            # 5. Dynamic upgrade scoring, weapon loadout ranking, safe loot/crate/NPC/drop valuation
  strategy.py             # 6. Tactical mode state machine + strategy profiles (survivor / pacifist / hunter)
  controller.py           # 7. Top-level MechBot coordinator and lightweight debug drawing
```

## Module Responsibilities & Budgets

1. **`WorldModel` (`bot/world.py`)**
   - Loads `v2/world.json` (259 prisms: 223 buildings, 27 stadium, 9 bridges) and `v2/terrain.i16` (`1070×724` int16 cm) once during `hello` (~0.4–0.6 s on host, <2 s total container ready time).
   - Indexes prisms in a 32 m spatial hash and precomputes an 8 m 2D walkable navgrid (`walkable`, `cover_adj`) accounting for bridge head-clearance (`base >= z + 9.2 m`) and walkable curbs (`top <= z + 1.2 m`).
   - Provides O(1) bilinear `terrain_height`, fast spatial-hashed 3D `has_los`, and bounded top-14 `find_cover_point`.

2. **`Navigator` (`bot/nav.py`)**
   - Runs weighted A* (`f = g + 1.3h`, max 360 expansions) with threat-weighted cell costs (`danger_fn`), string-pull path smoothing, and 24-tick replan cadence.
   - Detects stuck movement over 6 decision ticks and executes tangential sidestep + jump/jet recovery.
   - Clamps all goals inside arena boundaries with margin.

3. **`ThreatTracker` (`bot/threat.py`)**
   - Tracks visible mechs with smoothed velocities and 12 s memory decay.
   - Integrates heard sounds (`laser_charge`, `lightning`, `gun`, `mortar`, `explosion`, `boost`, `step`) into decaying spatial threat pings.
   - Tracks `damaged` events to raise rear-armor turn-to-face alerts (avoiding 1.5× rear damage) and combat timers for out-of-combat HP regeneration (`regen_delay`).
   - Computes perpendicular dodge vectors for incoming gun/mortar projectiles and charging enemy lasers.

4. **`CombatPlanner` (`bot/combat.py`)**
   - Prioritizes visible enemies by proximity, recent-attacker status, low HP (kill finish), enemy aim/charge threat, and weapon-triangle matchups (laser vs mortar cookoff, lightning vs laser short).
   - Implements weapon-specific control:
     - `lightning`: close-range instant 240 DPS + laser shorting within cone & energy reserve.
     - `laser`: hold-to-charge, release-to-fire state machine once charged and aligned.
     - `gun`: 3-iteration 3D velocity lead with range-scaled aim tolerance and out-of-combat early reload.
     - `mortar`: ballistic pitch solver (low/high arc) with 18 m close-range self-damage guard.

5. **`ResourcePlanner` (`bot/resources.py`)**
   - Scores upgrades dynamically (`nanites`, `plating`, `armor`, `servos` prioritized for survivability, plus weapon upgrades for held weapons).
   - Prefers `gun` + `lightning`/`laser` loadout and avoids picking up `mortar` to prevent +60% laser cook-off vulnerability.
   - Evaluates safe cubes, weapon pickups, crates, isolated NPC drones, and nearby supply drops.

6. **`StrategyController` (`bot/strategy.py`)**
   - Coordinates 5 tactical modes: `ZONE_URGENT`, `RETREAT_HEAL`, `COMBAT_KITE`, `LOOT_*`, `SAFE_ROTATE`.
   - Rotates to cover-adjacent, low-threat anchor points on the quiet ring of `next_center` instead of rushing the crowded center.
   - Supports three strategy profiles via `StrategyProfile`: `tactical_survivor`, `pure_pacifist`, and `opportunistic_hunter`.
