# Unified Battle Intelligence (`UBI-v1`) — Match Decision Report

## 1. Match Overview
- **Decision ticks analyzed:** `300` (`t=0.0s` to `36.8s`)
- **HP trajectory:** `1400.0` -> `9.5` (damage dealt: `472.5`, damage received: `1665.0`, zone damage: `0.0`)
- **In-zone compliance:** `100.0%`
- **Decision latency:** `p50=0.3 ms` | `p90=1.37 ms` | `p99=4.79 ms` | `max=5.26 ms`

## 2. Objective & Phase Breakdown
| Objective | Ticks | Share % | Fire Ticks | Boost Ticks | Dmg Dealt | Dmg Recv | Displacement (m) | Achieved % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RETREAT_HEAL` | 125 | 41.7% | 0 | 28 | 270.0 | 630.0 | 110.1 | 94.4% |
| `SAFE_ROTATE` | 118 | 39.3% | 0 | 0 | 45.0 | 315.0 | 203.1 | 49.2% |
| `COMBAT_KITE` | 57 | 19.0% | 37 | 25 | 157.5 | 720.0 | 52.8 | 8.8% |

## 3. Strategic Transition Timeline (First 20 Transitions)
- `t=0.0s` (tick `0`, phase `EARLY_GAME`, hp `1400`, alive `16`): switched `START` -> **`SAFE_ROTATE`** (reasons: `PHASE_EARLY_GAME, CORNER_COVER_ANCHOR`)
- `t=4.9s` (tick `98`, phase `EARLY_GAME`, hp `1400`, alive `16`): switched `SAFE_ROTATE` -> **`COMBAT_KITE`** (reasons: `PHASE_EARLY_GAME, FAVORABLE_ENGAGEMENT`)
- `t=10.6s` (tick `212`, phase `EARLY_GAME`, hp `680`, alive `16`): switched `COMBAT_KITE` -> **`RETREAT_HEAL`** (reasons: `PHASE_EARLY_GAME, UNFAVORABLE_DUEL_HP`)
- `t=14.6s` (tick `292`, phase `EARLY_GAME`, hp `545`, alive `16`): switched `RETREAT_HEAL` -> **`SAFE_ROTATE`** (reasons: `PHASE_EARLY_GAME, CORNER_COVER_ANCHOR`)
- `t=18.0s` (tick `360`, phase `EARLY_GAME`, hp `545`, alive `15`): switched `SAFE_ROTATE` -> **`RETREAT_HEAL`** (reasons: `PHASE_EARLY_GAME, UNFAVORABLE_DUEL_HP`)
- `t=26.1s` (tick `522`, phase `EARLY_GAME`, hp `196`, alive `13`): switched `RETREAT_HEAL` -> **`SAFE_ROTATE`** (reasons: `PHASE_EARLY_GAME, CORNER_COVER_ANCHOR`)
- `t=35.9s` (tick `718`, phase `EARLY_GAME`, hp `144`, alive `13`): switched `SAFE_ROTATE` -> **`RETREAT_HEAL`** (reasons: `PHASE_EARLY_GAME, UNFAVORABLE_DUEL_HP`)

## 4. Combat, Survival, Resource & Zone Positioning Analysis
- **Combat decisions:** Active engagement on `57` ticks with `37` firing ticks.
- **Survival & retreat decisions:** Moving retreat (`RETREAT_HEAL`) on `125` ticks (`110.1 m` displacement; zero stationary cover freezing).
- **Zone positioning:** Safe cover-corner positioning (`SAFE_ROTATE`) on `118` ticks (`100.0%` ticks inside active zone).
