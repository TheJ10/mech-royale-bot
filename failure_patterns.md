# Failure Patterns & Evidence-Based Strategy Adjustments

> Note: Correlated consequence metrics from recorded or simulated traces identify candidate friction points; correlation does not prove causation.

## 1. Unachieved Objective Summary
- **Total unachieved objective ticks:** `118 / 300`
  - `SAFE_ROTATE`: `60` ticks where immediate tick progress or combat hit was not realized
  - `COMBAT_KITE`: `52` ticks where immediate tick progress or combat hit was not realized
  - `RETREAT_HEAL`: `6` ticks where immediate tick progress or combat hit was not realized

## 2. Unnecessary Exposure Incidents (`damage_received > 60` with `damage_dealt == 0` outside retreat/zone)
- **Total incidents:** `2`
  - `COMBAT_KITE`: `2` incidents

## 3. Evidence-Based Strategy Recommendations
1. **Maintain active moving retreat (`RETREAT_HEAL`):** Moving retreat keeps displacement positive while breaking LoS behind building corners (`escape_deg` 4..6), avoiding the 55-tick stationary freeze observed in v3.1.
2. **Preserve weapon-slew fire gating:** When `COMBAT_KITE` has unachieved fire ticks, torso yaw/pitch is actively slewing toward the velocity-led intercept point; gating trigger on angular alignment preserves magazine ammo.
3. **Dual-purpose NPC routing:** Routing around 50 m NPC aggro rings when `hp_frac < 0.65` prevents low-HP attrition without interrupting mech combat or healthy 2-cube NPC farming.
