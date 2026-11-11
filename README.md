# Werewolf Agent

An entry for make.fun's **werewolf-agent-01** competition: a program that
plays one seat in 30 games of 8-player Werewolf against seven house players —
15 games as a werewolf, 5 as the seer, 5 as the doctor, 5 as a villager.

## How it plays

For every decision the game server waits on, `main.py`:

1. Renders that game's view (players, private knowledge, the full public log)
   into a compact text, plus a briefing computed in code: revealed roles,
   werewolves remaining, seer claims heard in the debate, vote-history
   suspicion scores, and — for a werewolf — whether its partner has already
   named tonight's victim.
2. Adds a role playbook (werewolf / seer / doctor / villager): when the seer
   should claim, how the wolves agree on a kill, whom the doctor guards, how
   the village coordinates its vote.
3. Asks a model (via the trial gateway) for one JSON answer, validates it
   against the decision's options, and posts it to the env.

If the model cannot be reached, refuses, or answers unreadably, a
deterministic heuristic answers instead — the kill follows the partner
wolf's pick, the seer claims a found wolf, the doctor guards the claimed
seer, votes join the village's best wagon — so no game ever stalls. After
repeated model failures the heuristics take over for the rest of the trial.

The agent is deterministic given the model's replies: no clock is read into
anything sent, all randomness comes from the trial seed through the vendored
SDK's seeded RNG, and every request is built from one game's view alone.

## Layout

- `main.py` — the entry (Python 3.13, standard library only).
- `vendor/makefun_agent.py` — the make.fun agent SDK, vendored as the kit
  instructs (gateway client and seeded RNG).
- `LICENSE` — MIT.

Run by the grader as `python3 -P -s /bot/main.py`.
