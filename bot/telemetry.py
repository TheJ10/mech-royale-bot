"""Decision Intelligence and Consequence Telemetry Engine (never writes to stdout)."""
from __future__ import annotations

import collections
import json
import math
import os
from typing import Any, Deque, Dict, List, Optional, Tuple


class DecisionTelemetry:
    """Bounded per-tick decision & consequence recorder with optional JSONL file output."""

    MAX_RING = 80
    MAX_FILE_BYTES = 16 * 1024 * 1024  # 16 MB safety cap (well inside 64 MB /tmp limit)

    def __init__(
        self,
        trace_path: Optional[str] = None,
        enabled: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        env_path = os.environ.get("MECH_TELEMETRY_PATH", "").strip()
        self.trace_path: Optional[str] = trace_path if trace_path is not None else (env_path or None)
        self.file_enabled: bool = bool(self.trace_path) if enabled is None else bool(enabled and self.trace_path)
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        self.ring: Deque[Dict[str, Any]] = collections.deque(maxlen=self.MAX_RING)
        self._prev_record: Optional[Dict[str, Any]] = None
        self._prev_state: Optional[Dict[str, Any]] = None
        self._bytes_written: int = 0
        if self.file_enabled and self.trace_path:
            try:
                with open(self.trace_path, "w", encoding="utf-8") as _:
                    pass
            except OSError:
                self.file_enabled = False

    def record_tick(
        self,
        obs: Dict[str, Any],
        decision_meta: Dict[str, Any],
        act: Dict[str, Any],
        elapsed_ms: float,
    ) -> None:
        """Correlate previous decision's consequences with current obs, then record new decision."""
        if not self.file_enabled:
            return
        me = obs["self"]
        now = float(obs["t"])
        tick = int(obs["tick"])
        pos = (float(me["pos"][0]), float(me["pos"][1]), float(me["pos"][2]))
        hp = float(me["hp"])
        max_hp = float(me["max_hp"])
        energy = float(me["energy"])
        cubes = int(me.get("cubes", 0))
        in_zone = bool(me.get("in_zone", True))

        # 1. Correlate consequences on previous decision record
        if self._prev_record is not None and self._prev_state is not None:
            p_pos = self._prev_state["pos"]
            disp = round(math.hypot(pos[0] - p_pos[0], pos[1] - p_pos[1]), 2)
            hp_delta = round(hp - self._prev_state["hp"], 1)
            energy_delta = round(energy - self._prev_state["energy"], 1)
            cubes_gained = max(0, cubes - self._prev_state["cubes"])

            dmg_recv = 0.0
            zone_dmg = 0.0
            dmg_dealt = 0.0
            kills = 0
            for ev in obs.get("events") or []:
                et = ev.get("type")
                if et == "damaged":
                    amt = float(ev.get("amount", 0.0))
                    dmg_recv += amt
                    if ev.get("weapon") == "zone":
                        zone_dmg += amt
                elif et == "hit":
                    dmg_dealt += float(ev.get("amount", 0.0))
                elif et == "kill":
                    kills += 1

            p_goal = self._prev_state["goal"]
            prev_goal_d = math.hypot(p_goal[0] - p_pos[0], p_goal[1] - p_pos[1])
            cur_goal_d = math.hypot(p_goal[0] - pos[0], p_goal[1] - pos[1])
            obj_name = self._prev_record.get("selected_objective", "")
            achieved = (
                (obj_name.startswith("LOOT") and (cubes_gained > 0 or dmg_dealt > 0 or cur_goal_d < prev_goal_d - 0.5))
                or (obj_name == "COMBAT_KITE" and (dmg_dealt > 0 or kills > 0))
                or (obj_name == "ZONE_URGENT" and (in_zone or cur_goal_d < prev_goal_d - 0.5))
                or (obj_name == "RETREAT_HEAL" and (dmg_recv == 0.0 or hp_delta > 0.0))
                or (cur_goal_d <= 6.0 or cur_goal_d < prev_goal_d - 0.3)
            )
            unnecessary_exposure = bool(dmg_recv > 60.0 and dmg_dealt == 0.0 and obj_name not in ("RETREAT_HEAL", "ZONE_URGENT"))

            cur_vis_e = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"]
            cur_min_edist = (
                min(math.hypot(float(m["pos"][0]) - pos[0], float(m["pos"][1]) - pos[1]) for m in cur_vis_e)
                if cur_vis_e else None
            )
            prev_min_edist = self._prev_state.get("min_enemy_dist")
            retreat_sep_delta = (
                round(cur_min_edist - prev_min_edist, 2)
                if (obj_name == "RETREAT_HEAL" and cur_min_edist is not None and prev_min_edist is not None)
                else None
            )
            retreat_los_broken = bool(
                obj_name == "RETREAT_HEAL"
                and self._prev_state.get("had_vis_enemy")
                and not cur_vis_e
            )

            self._prev_record["consequence"] = {
                "next_tick": tick,
                "displacement": disp,
                "hp_delta": hp_delta,
                "energy_delta": energy_delta,
                "damage_received": round(dmg_recv, 1),
                "damage_dealt": round(dmg_dealt, 1),
                "kills": kills,
                "cubes_gained": cubes_gained,
                "zone_damage": round(zone_dmg, 1),
                "objective_achieved": bool(achieved),
                "unnecessary_exposure": unnecessary_exposure,
                "retreat_sep_delta": retreat_sep_delta,
                "retreat_los_broken": retreat_los_broken,
            }
            self._flush_record(self._prev_record, elapsed_ms)

        # 2. Build current decision record
        vis_e = [
            {"id": int(m["id"]), "hp": round(float(m.get("hp", 0)), 1), "dist": round(math.dist(m["pos"], pos), 1), "weapons": m.get("weapons", []), "charging": bool(m.get("charging", False))}
            for m in (obs.get("mechs") or [])
            if m.get("kind") == "mech"
        ]
        vis_n = [
            {"id": int(m["id"]), "hp": round(float(m.get("hp", 0)), 1), "dist": round(math.dist(m["pos"], pos), 1)}
            for m in (obs.get("mechs") or [])
            if m.get("kind") == "npc"
        ]
        w_states = [
            {"kind": w["kind"], "cd": round(float(w.get("cooldown", 0.0)), 2), "ammo": w.get("ammo"), "reload": round(float(w.get("reload", 0.0)), 2), "charge": round(float(w.get("charge", 0.0)), 2)}
            if w is not None else None
            for w in me.get("weapons", [])
        ]
        zone = obs["zone"]
        goal = decision_meta.get("goal", (0.0, 0.0))

        rec: Dict[str, Any] = {
            "t": round(now, 2),
            "tick": tick,
            "hp": round(hp, 1),
            "max_hp": round(max_hp, 1),
            "energy": round(energy, 1),
            "cubes": cubes,
            "alive": int(obs.get("alive", 16)),
            "in_zone": in_zone,
            "zone": {
                "next_center": [round(float(zone["next_center"][0]), 1), round(float(zone["next_center"][1]), 1)],
                "next_radius": round(float(zone["next_radius"]), 1),
                "shrink_start": round(float(zone["shrink_start"]), 1),
            },
            "visible_enemies": vis_e,
            "known_npcs": vis_n,
            "resources_visible": {
                "cubes": len(obs.get("cubes") or []),
                "crates": len(obs.get("crates") or []),
                "pickups": len(obs.get("pickups") or []),
                "drops": len(obs.get("drops") or []),
            },
            "phase": decision_meta.get("phase", "MID_GAME"),
            "selected_objective": decision_meta.get("selected_objective", "SAFE_ROTATE"),
            "candidates_considered": decision_meta.get("candidates", []),
            "scores": decision_meta.get("scores", {}),
            "risk_factors": decision_meta.get("risk_factors", {}),
            "target_id": decision_meta.get("target_id"),
            "target_priority": round(float(decision_meta.get("target_priority", 0.0)), 1),
            "goal": [round(float(goal[0]), 1), round(float(goal[1]), 1)],
            "action": {
                "move": [round(float(act["move"][0]), 3), round(float(act["move"][1]), 3)],
                "boost": bool(act.get("boost", False)),
                "jump": bool(act.get("jump", False)),
                "aim": [round(float(act["aim"][0]), 3), round(float(act["aim"][1]), 3)],
                "fire": list(act.get("fire", [False, False])),
                "reload": bool(act.get("reload", False)),
                "pick": act.get("pick"),
                "take": act.get("take"),
            },
            "weapons": w_states,
            "reason_codes": decision_meta.get("reason_codes", []),
            "expected_benefit": round(float(decision_meta.get("expected_benefit", 0.0)), 1),
            "expected_risk": round(float(decision_meta.get("expected_risk", 0.0)), 1),
            "latency_ms": round(elapsed_ms, 2),
            "consequence": None,
        }
        if self.metadata:
            rec.update(self.metadata)
        min_edist = min((e["dist"] for e in vis_e), default=None)
        self._prev_record = rec
        self._prev_state = {
            "pos": pos,
            "hp": hp,
            "energy": energy,
            "cubes": cubes,
            "goal": goal,
            "min_enemy_dist": min_edist,
            "had_vis_enemy": bool(vis_e),
        }

    def finalize(self) -> None:
        """Flush final pending decision record if not yet written."""
        if self._prev_record is not None:
            self._flush_record(self._prev_record, 0.0)
            self._prev_record = None

    def _flush_record(self, rec: Dict[str, Any], elapsed_ms: float) -> None:
        self.ring.append(rec)
        if not self.file_enabled or not self.trace_path:
            return
        if elapsed_ms > 25.0 or self._bytes_written >= self.MAX_FILE_BYTES:
            return
        try:
            line = json.dumps(rec, separators=(",", ":")) + "\n"
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(line)
            self._bytes_written += len(line)
        except OSError:
            self.file_enabled = False
