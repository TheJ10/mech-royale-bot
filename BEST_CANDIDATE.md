# Best Candidate Tracker & Leaderboard

## Current Best Officially Scored Candidate
- **Candidate:** Tactical Survivor v2 (Iter 2)
- **Branch:** `experiment/tactical-survivor-v2` (preserved)
- **Exact Commit SHA:** `8abe44e217e5b0b97ad86cac7d3453273bfb27cf`
- **Official Practice Score:** **3.44** *(3× official starter 1.15)*

## Full Candidate Leaderboard

| Rank | Candidate | Branch | Exact Commit SHA | Official Score | Docker Sandbox (`linux/amd64`, 2 runs) | Tests | Submission Readiness |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| 1 | **Tactical Survivor v2** | `experiment/tactical-survivor-v2` | `8abe44e217e5b0b97ad86cac7d3453273bfb27cf` | **3.44** | PASS (`0` late, `P99 32.2 ms`, `max 39.8 ms`) | `20/20` | Preserved best official |
| — | **Tactical Survivor v3** *(recommended next)* | `experiment/tactical-survivor-v3` | `5779c3806bc70abd8dc814228598baa8919e4a8d` | *Pending* | **PASS ×2** (`0` late both runs; `P99 8.5–11.6 ms`, `max 13.7–21.9 ms`) | `23/23` | **READY** |
| — | **Opportunistic Hunter v2** | `experiment/opportunistic-hunter-v2` | `2930e146379d757a62880aaab947e047e09777ab` | *Pending* | **PASS ×2** (`0` late both runs; `P99 16.2–31.3 ms`, `max 24.0–38.2 ms`) | `24/24` | **READY** |
| — | **Pure Pacifist v2** | `experiment/pure-pacifist-v2` | `b96810400226bb45541e4521c8f950b5c8817b49` | *Pending* | PASS ×2 (`Run 1: 0 late, P99 15.0 ms, max 16.6 ms`; `Run 2: 1 jitter spike`) | `24/24` | **READY** *(stealth baseline)* |
| 2 | Official starter | `main` / `baseline/official-starter` | `14a4d1ef8a8dd0d8d37170b74f4feb95bb554991` | **1.15** | PASS (`0` late, `P99 28.5 ms`) | — | Preserved baseline |
| 3 | Defensive survival v1 | `baseline/survival-v1` | `eaca8b212e83f6986fb404efc6d26e6f385b3c1c` | **0.85** | PASS (`0` late, `P99 29.6 ms`) | — | Preserved baseline |
| — | Opportunistic Hunter v1 | `experiment/opportunistic-hunter-v1` | `cf14bf70e2825fa52fb12db83509456629dd8712` | *Unsubmitted* | PASS (`0` late, `P99 29.4 ms`) | `24/24` | Preserved parent |
| — | Pure Pacifist v1 | `experiment/pure-pacifist-v1` | `4c3e36cc8f57fd500b3723eb8aaab1b225859421` | *Unsubmitted* | PASS (`0` late, `P99 23.1 ms`, `max 49.1 ms`) | `24/24` | Preserved parent |
