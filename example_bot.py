"""Prioritize the safe zone, collect nearby loot, and return fire when attacked."""
import math
import mechfight as mf


ZONE_BUFFER_METERS = 15.0
ZONE_LEAD_SECONDS = 10.0
EARLY_BOOST_MIN_ENERGY = 20.0
LOOT_DETOUR_METERS = 18.0
RETALIATE_SECONDS = 4.0
AIM_TOLERANCE = 0.14


class Bot:
    name = "survival-py"

    def __init__(self, hello: mf.Hello):
        self.reach = hello["config"]["loot"]["pickup_radius"]
        self.last_hit_by: dict[int, float] = {}

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        zone = obs["zone"]
        zone_goal = zone["next_center"]
        goal = zone_goal
        now = obs["t"]

        # Estimate how long it would take to walk into the next circle with room to spare.
        safe_radius = max(0.0, zone["next_radius"] - ZONE_BUFFER_METERS)
        distance_to_safety = max(0.0, math.dist((x, y), zone_goal) - safe_radius)
        walk_seconds = distance_to_safety / me["stats"]["walk_speed"]
        seconds_until_shrink = zone["shrink_start"] - now
        zone_urgent = distance_to_safety > 0 and seconds_until_shrink <= walk_seconds + ZONE_LEAD_SECONDS
        jump = obs["tick"] % 40 == 0
        early_boost = (
            zone_urgent
            and me["grounded"]
            and not jump
            and me["energy"] > EARLY_BOOST_MIN_ENERGY
        )

        for event in obs["events"]:
            if event["type"] == "damaged" and event["by"] is not None:
                self.last_hit_by[event["by"]] = now
        recent_attackers = {
            attacker for attacker, hit_time in self.last_hit_by.items() if now - hit_time <= RETALIATE_SECONDS
        }
        act: mf.Action = {"tick": obs["tick"]}
        if obs["offer"]:
            choices = obs["offer"]["choices"]
            act["pick"] = next((i for i, choice in enumerate(choices) if choice["id"] == "armor"), 0)

        # Take the close-range defensive weapon and only detour for loot that is already safe.
        lightning = None
        if me["weapons"][1] is None:
            lightning = min(
                (p for p in obs["pickups"] if p["kind"] == "lightning" and abs(p["pos"][2] - z) <= 4),
                key=lambda pickup: math.dist(pickup["pos"][:2], (x, y)),
                default=None,
            )
        if lightning and math.dist(lightning["pos"][:2], (x, y)) <= self.reach:
            act["take"] = lightning["id"]
        safe_to_loot = me["in_zone"] and not zone_urgent and not recent_attackers
        if safe_to_loot:
            loot_positions = [cube["pos"] for cube in obs["cubes"]]
            if lightning and "take" not in act:
                loot_positions.append(lightning["pos"])
            goal = min(
                (
                    pos[:2] for pos in loot_positions
                    if abs(pos[2] - z) <= 4
                    and math.dist(pos[:2], (x, y)) <= LOOT_DETOUR_METERS
                    and math.dist(pos[:2], zone_goal) <= safe_radius
                ),
                key=lambda point: math.dist(point, (x, y)),
                default=zone_goal,
            )

        yaw = math.atan2(goal[1] - y, goal[0] - x)
        move_scale = min(1.0, math.dist(goal, (x, y)) / 3.0)
        act.update(
            move=[move_scale * math.cos(yaw), move_scale * math.sin(yaw)],
            aim=[yaw, 0.0],
            boost=not me["in_zone"] or early_boost,
            jump=jump,
        )
        attacker = min(
            (m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] in recent_attackers),
            key=lambda m: math.dist(m["pos"], me["pos"]),
            default=None,
        )
        if attacker and me["in_zone"]:
            t = attacker["pos"]
            muzzle_z = z + me["stats"]["muzzle_height"]
            aim = [math.atan2(t[1] - y, t[0] - x), math.atan2(t[2] + 4.5 - muzzle_z, math.dist(t[:2], (x, y)))]
            act["aim"] = aim
            aimed = all(abs(math.remainder(a - b, math.tau)) <= AIM_TOLERANCE for a, b in zip(aim, me["aim"]))
            distance = math.dist((x, y, muzzle_z), (t[0], t[1], t[2] + 4.5))
            act["fire"] = [
                bool(aimed and weapon and weapon["kind"] in ("gun", "lightning") and distance <= weapon["stats"]["range"])
                for weapon in me["weapons"]
            ]
            mf.draw.line([x, y, z + 7], [t[0], t[1], t[2] + 4.5], "#ff5040")
        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 6)
        mf.draw.label(f"hp {me['hp']:.0f} · {len(recent_attackers)} recent attackers")
        return act


mf.run(Bot)
