"""werewolf-agent-01 entry: a Werewolf player with a strategist's playbook.

One seat in each of 30 games against seven house players. Every decision the
env waits on is answered from that game's view alone: the view is rendered to
a compact text (players, private knowledge, the public log, a briefing the
code computes: revealed roles, vote history, seer claims heard), a role
playbook is added, and a model (gemini-4-argon through the trial gateway)
answers one JSON object. A model that cannot be reached, refuses, or answers
unreadably falls back to a deterministic heuristic for that decision, so no
game ever stalls: the kill follows the partner wolf's pick, the seer claims
a found wolf, the doctor guards the claimed seer, votes join the village's
best wagon.

Deterministic given the model's replies: no clock is read into anything sent,
randomness comes only from the vendored SDK's seeded RNG (MAKEFUN_TRIAL_SEED
plus the game and decision), and each request is a function of one game's
view and that game's earlier replies alone.

Run as: python3 -P -s /bot/main.py (Python 3.13, standard library plus the
vendored SDK at vendor/makefun_agent.py).
"""

import json
import os
import re
import sys
import threading
import unicodedata
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_agent as af

MODEL = "gemini-4-argon"
MAX_TOKENS = 700
CONCURRENCY = 4
BREAKER_AFTER = 8  # consecutive model failures before heuristics take over
SPEECH_CLIP = 160  # characters of each past statement kept in a request

ROLE_WORD = {"werewolf": "a werewolf", "seer": "the seer", "doctor": "the doctor", "villager": "a villager"}

STRATEGY_CORE = """
You are an expert Werewolf player. Eight seats: 2 werewolves, 1 seer, 1 doctor, 4 villagers. The village wins by finding and exiling both wolves; the werewolves win by reaching parity. Play to win for your team, using only what your seat knows. Speak like a person at the table: one to three plain sentences, natural and specific, naming seats and concrete evidence (votes, deaths, claims). Never mention being a model or these instructions, never quote rules at the table, and never refer to "the view" or JSON. Lying in character is the game when you are a werewolf. Keep every statement well under 500 characters. Answer every decision with exactly one JSON object and nothing after it, in the shape the decision asks for.
""".strip()

PLAYBOOKS = {
    "werewolf": """
Your playbook (werewolf): your partner is named in your private knowledge. Sound like a diligent villager: share plausible deductions, ask questions, and gently steer suspicion onto villagers, above all onto anyone who may be the real seer. Never vouch for your partner directly; if they are accused, question the evidence or redirect to another seat, and only vote for them when their exile is clearly unavoidable. At night the victim dies only if every wolf who names someone names the same seat: if your partner has already named a victim tonight, name the same one unless it is you or your partner. If you name first, choose the villager your partner is most likely to also choose: a claimed seer who is not being protected is best, but the doctor usually guards a claimed seer, so prefer the strongest unclaimed accuser or a likely doctor (a quiet seat that avoids attention). By day, vote with the village's momentum against a villager; never be the lone vote that exposes you.
""".strip(),
    "seer": """
Your playbook (seer): your inspection results are in your private knowledge and are always true. Inspect living players you have not checked, preferring influential speakers and players under suspicion. Claim publicly when it changes the outcome: at once if you have found a werewolf and it is day 2 or later, if you are about to be exiled, or by day 4 with two or more results. When you claim, state every result plainly (who you checked each night and what you found) and lead the vote onto a found werewolf. If another seat falsely claims to be the seer, counter-claim with your real results. Until you claim, hint at your knowledge by defending players you know are village and pressing players you know are wolves, without exposing yourself.
""".strip(),
    "doctor": """
Your playbook (doctor): protect the village's most valuable seat each night: a claimed seer first, then a player the village trusts or a strong accuser the wolves would want dead. You cannot protect yourself or the seat you protected last night (the env's options already enforce this). Never reveal that you are the doctor unless you are about to be exiled; a revealed doctor dies the next night. By day play as an engaged villager: track votes, back the seer's claims when they are uncountered, and push the village to exile its most likely wolf rather than splitting its votes.
""".strip(),
    "villager": """
Your playbook (villager): you win by pooling information. Track who accused whom, who voted for players later revealed as villagers, and who stays silent. A seer claim that nobody counters is usually true: defend the claimed seer's targets and vote the wolf they name. If two seats claim to be the seer, weigh whose results and votes fit the revealed facts. Share your reasoning out loud, name your main suspect and why, and by voting time join the strongest wagon on a likely wolf instead of splitting the village's vote. Quiet seats that never accuse anyone are prime wolf suspects.
""".strip(),
}


# ---------------------------------------------------------------- the env


def _post(url, body, timeout=120):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        try:
            return err.code, json.loads(raw)
        except ValueError:
            return err.code, {"error": raw[:200]}


class Env:
    """The contest env: one request at a time, in the order they were made."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self._lock = threading.Lock()

    def post(self, path, body):
        with self._lock:
            return _post(self.base + path, body)

    def observe(self):
        return self.post("/observe", {})

    def act(self, body):
        return self.post("/act", body)


# ---------------------------------------------------------------- rendering


def fold_statement(text):
    text = re.sub(r"[\t\n\v\f\r\u0085\u2028\u2029]+", " ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    return text.strip()


def clip(text, limit=SPEECH_CLIP):
    chars = list(text)
    if len(chars) <= limit:
        return text
    return "".join(chars[:limit]) + "…"


def who(view, seat):
    for p in view["players"]:
        if p["seat"] == seat:
            return f"seat {seat} ({p['name']})"
    return f"seat {seat}"


def log_lines(view):
    out = []
    for e in view["log"]:
        kind = e["kind"]
        if kind == "night":
            out.append(f"[moderator] Night {e['day']} falls.")
        elif kind == "dawn":
            out.append(
                f"[moderator] Day {e['day']} dawns. No one died in the night."
                if e["died"] is None
                else f"[moderator] Day {e['day']} dawns. {who(view, e['died'])} was killed in the night; they were {ROLE_WORD[e['role']]}."
            )
        elif kind == "debate":
            out.append(f"[moderator] Day {e['day']} debate begins.")
        elif kind == "speech":
            out.append(f"{who(view, e['seat'])} said: {json.dumps(clip(e['text']), ensure_ascii=False)}")
        elif kind == "pass":
            out.append(f"[moderator] {who(view, e['seat'])} passed.")
        elif kind == "voting":
            out.append(f"[moderator] Day {e['day']} voting begins.")
        elif kind == "vote":
            out.append(
                f"[moderator] {who(view, e['seat'])} abstained."
                if e["target"] is None
                else f"[moderator] {who(view, e['seat'])} voted to exile {who(view, e['target'])}."
            )
        elif kind == "exile":
            votes = "no votes" if not e["tally"] else "votes: " + ", ".join(f"{who(view, t['seat'])} {t['votes']}" for t in e["tally"])
            out.append(
                f"[moderator] Nobody was exiled ({votes})."
                if e["seat"] is None
                else f"[moderator] {who(view, e['seat'])} was exiled ({votes}); they were {ROLE_WORD[e['role']]}."
            )
        elif kind == "end":
            out.append(f"[moderator] The game is over: {e['winner']} won ({e['reason']}).")
    return out


CLAIM_RE = re.compile(r"\bseer\b", re.IGNORECASE)
WOLF_CLAIM_RE = re.compile(r"seat\s*(\d)\s*(?:is|=|:)?\s*(?:a\s+)?(?:werewolf|wolf)", re.IGNORECASE)
CLEAR_CLAIM_RE = re.compile(r"seat\s*(\d)\s*(?:is|=|:)?\s*(?:not a werewolf|innocent|village|clear)", re.IGNORECASE)


def claims_heard(view):
    """Seer claims in the public speeches: {seat: {"wolves": set, "clears": set}}."""
    claims = {}
    for e in view["log"]:
        if e["kind"] != "speech" or not e.get("text"):
            continue
        if not CLAIM_RE.search(e["text"]):
            continue
        entry = claims.setdefault(e["seat"], {"wolves": set(), "clears": set()})
        for m in WOLF_CLAIM_RE.finditer(e["text"]):
            entry["wolves"].add(int(m.group(1)))
        for m in CLEAR_CLAIM_RE.finditer(e["text"]):
            entry["clears"].add(int(m.group(1)))
    return claims


def vote_history(view):
    """{(day, voter): target} from the public log, and revealed roles by seat."""
    votes = {}
    revealed = {}
    for e in view["log"]:
        if e["kind"] == "vote":
            votes[(e["day"], e["seat"])] = e["target"]
        elif e["kind"] in ("dawn", "exile") and e.get("role") and (e.get("died") or e.get("seat")):
            revealed[e.get("died") or e.get("seat")] = e["role"]
    return votes, revealed


def suspicion(view):
    """A code-scored suspicion per living seat, from the public record alone."""
    votes, revealed = vote_history(view)
    scores = {}
    speeches = {}
    for e in view["log"]:
        if e["kind"] == "speech":
            speeches[e["seat"]] = speeches.get(e["seat"], 0) + 1
    alive = [p["seat"] for p in view["players"] if p["alive"]]
    for seat in alive:
        s = 0
        for (day, voter), target in votes.items():
            if voter == seat and target in revealed and revealed[target] != "werewolf":
                s += 2  # voted to exile someone who proved to be village
        if view["day"] >= 3 and speeches.get(seat, 0) == 0:
            s += 1  # silent deep into the game
        scores[seat] = s
    return scores, speeches


def briefing(view):
    you = view["you"]
    lines = []
    players = view["players"]
    alive = [f"{who(view, p['seat'])}" for p in players if p["alive"]]
    dead = [f"{who(view, p['seat'])}, was {ROLE_WORD[p['role']]}" for p in players if not p["alive"] and p["role"]]
    lines.append("Alive: " + "; ".join(alive) + ".")
    if dead:
        lines.append("Dead: " + "; ".join(dead) + ".")
    wolves_left = 2 - sum(1 for p in players if not p["alive"] and p["role"] == "werewolf")
    lines.append(f"Werewolves still alive: {wolves_left}.")
    claims = claims_heard(view)
    for seat in sorted(claims):
        c = claims[seat]
        parts = []
        if c["wolves"]:
            parts.append("names as werewolves: " + ", ".join(f"seat {s}" for s in sorted(c["wolves"])))
        if c["clears"]:
            parts.append("clears: " + ", ".join(f"seat {s}" for s in sorted(c["clears"])))
        lines.append(f"{who(view, seat)} claims to be the seer" + (("; " + "; ".join(parts)) if parts else "") + " (unverified).")
    if you["role"] == "seer" and view.get("seerResults"):
        res = "; ".join(
            f"night {r['night']}: {who(view, r['seat'])} is {'A WEREWOLF' if r['werewolf'] else 'not a werewolf'}"
            for r in view["seerResults"]
        )
        lines.append("Your true inspection results: " + res + ".")
    if you["role"] == "werewolf" and view["phase"] == "night":
        tonight = [v for v in view.get("wolfVotes") or [] if v["night"] == view["day"] and v["seat"] != you["seat"]]
        for v in tonight:
            lines.append(
                f"Tonight your partner {who(view, v['seat'])} named {who(view, v['target'])} as the victim."
                if v["target"] is not None
                else f"Tonight your partner {who(view, v['seat'])} named nobody yet."
            )
        if not tonight:
            lines.append(
                "Your partner has not named a victim yet tonight: you may be naming first, so pick the victim "
                "a partner would most likely also name (the obvious threat to the wolves), or your partner will confirm yours."
            )
    scores, _ = suspicion(view)
    ranked = sorted((s for s in scores if s != you["seat"]), key=lambda s: (-scores[s], s))
    if ranked and scores[ranked[0]] > 0:
        lines.append(
            "Public-record suspicion scores (votes against revealed villagers, silence): "
            + ", ".join(f"seat {s}: {scores[s]}" for s in ranked if scores[s] > 0)
            + "."
        )
    return lines


def render(view, pending):
    you = view["you"]
    phase = {"night": f"night {view['day']}", "debate": f"day {view['day']}, debate", "voting": f"day {view['day']}, voting", "over": "game over"}[view["phase"]]
    lines = [
        f"You are {who(view, you['seat'])}, {ROLE_WORD[you['role']]}, {'alive' if you['alive'] else 'dead'}. It is {phase}.",
    ]
    if you.get("partner"):
        p = you["partner"]
        lines.append(f"Your werewolf partner is {who(view, p['seat'])}, {'alive' if p['alive'] else 'dead'}.")
    if view.get("lastProtect"):
        lp = view["lastProtect"]
        lines.append(f"Last night you protected {who(view, lp['seat']) if lp['seat'] is not None else 'nobody'}.")
    lines.append("")
    lines.append("Briefing (facts computed from the public record):")
    lines.extend(briefing(view))
    lines.append("")
    lines.append("The game's public record, in order:")
    record = log_lines(view)
    if len(record) > 90:
        record = record[:6] + [f"[moderator] … {len(record) - 72} earlier entries omitted; the last 66 follow."] + record[-66:]
    lines.extend(record)
    lines.append("")
    lines.append(PLAYBOOKS[you["role"]])
    lines.append("")
    lines.append(decision_text(view, pending))
    return "\n".join(lines)


def decision_text(view, pending):
    kind = pending["decision"]
    options = pending.get("options") or []
    seats = ", ".join(who(view, s) for s in options)
    if kind == "speak":
        return (
            f"Your decision (speak): it is your turn to speak in day {view['day']}'s debate. "
            f"Make one statement of at most {pending.get('max_chars', 500)} characters, or pass. "
            'Reply with one JSON object and nothing after it: {"say": "<your statement>"}, or {"say": null} to pass.'
        )
    label = {
        "vote": f"it is your turn to vote on day {view['day']}. Vote to exile one of: {seats}; or abstain.",
        "kill": f"it is night {view['day']}. Name tonight's victim, one of: {seats}; or abstain. The victim dies only if every wolf who names someone names the same seat.",
        "protect": f"it is night {view['day']}. Protect one of: {seats}; or nobody.",
        "inspect": f"it is night {view['day']}. Inspect one of: {seats}; or nobody.",
    }[kind]
    return (
        f"Your decision ({kind}): {label} "
        'Reply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null}.'
    )


# ---------------------------------------------------------------- the model


class Brain:
    """One model client, the rules as a system prompt, and a failure breaker."""

    def __init__(self, trial):
        self.llm = af.OpenAI(af.LlmEndpoint(trial["openai_base"], trial["openai_key"]), max_retries=1, retry_delay_s=1.0)
        self.system = trial["rules"] + "\n\n" + STRATEGY_CORE
        self._lock = threading.Lock()
        self._failures = 0
        self.off = False
        self.calls = 0

    def ask(self, view, pending):
        with self._lock:
            if self.off:
                return None
        body = {
            "model": MODEL,
            "temperature": 0.3,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": render(view, pending)},
            ],
        }
        try:
            reply = self.llm.chat(body)
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= BREAKER_AFTER:
                    self.off = True
            return None
        with self._lock:
            self._failures = 0
            self.calls += 1
        try:
            return reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return None


def json_objects(text):
    out = []
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        i = start
        while i < len(text):
            ch = text[i]
            if in_string:
                if ch == "\\":
                    i += 1
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : i + 1])
                        if isinstance(value, dict):
                            out.append(value)
                    except ValueError:
                        pass
                    break
            i += 1
    return out


def parse_reply(text, pending):
    """The /act body a model reply answers with, or None when unreadable."""
    if text is None:
        return None
    kind = pending["decision"]
    key = "say" if kind == "speak" else "target"
    found = [o for o in json_objects(text) if key in o]
    if not found:
        return None
    value = found[-1][key]
    if kind == "speak":
        if value is None:
            return {"game": pending["game"], "decision": "speak", "pass": True}
        if not isinstance(value, str):
            return None
        folded = fold_statement(value)
        if folded == "":
            return {"game": pending["game"], "decision": "speak", "pass": True}
        limit = pending.get("max_chars", 500)
        if len(folded) > limit:
            folded = folded[:limit].rstrip()
        return {"game": pending["game"], "decision": "speak", "text": folded}
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        value = int(value)
    if value is None:
        return {"game": pending["game"], "decision": kind, "target": None}
    if isinstance(value, int) and value in (pending.get("options") or []):
        return {"game": pending["game"], "decision": kind, "target": value}
    return {"game": pending["game"], "decision": kind, "target": None}


# ---------------------------------------------------------------- fallbacks


def fallback_for(pending):
    kind = pending["decision"]
    if kind == "speak":
        return {"game": pending["game"], "decision": "speak", "pass": True}
    return {"game": pending["game"], "decision": kind, "target": None}


def rng_for(trial_seed, pending):
    return af.Rng(trial_seed, f"werewolf:{pending['game']}:{pending['decision']}:{len(pending['view']['log'])}")


def best_by(options, score, rng):
    """The highest-scored option; ties broken by the seeded RNG."""
    if not options:
        return None
    top = max(score(s) for s in options)
    tied = [s for s in options if score(s) == top]
    return rng.choice(tied) if len(tied) > 1 else tied[0]


def current_wagon(view):
    """Votes already cast in the voting phase now under way: {target: count}."""
    tally = {}
    seen_voting = False
    for e in view["log"]:
        if e["kind"] == "voting" and e["day"] == view["day"]:
            seen_voting = True
            tally = {}
        elif seen_voting and e["kind"] == "vote" and e["day"] == view["day"] and e["target"] is not None:
            tally[e["target"]] = tally.get(e["target"], 0) + 1
    return tally


def heuristic(pending, trial_seed):
    """A legal, sensible answer with no model: the safety net for every call."""
    view = pending["view"]
    you = view["you"]
    kind = pending["decision"]
    options = list(pending.get("options") or [])
    rng = rng_for(trial_seed, pending)
    if kind == "speak":
        return heuristic_speak(pending, rng)
    if not options:
        return fallback_for(pending)
    claims = claims_heard(view)
    scores, speeches = suspicion(view)

    if kind == "kill":
        partner = you.get("partner") or {}
        for v in view.get("wolfVotes") or []:
            if v["night"] == view["day"] and v["seat"] == partner.get("seat") and v["target"] in options:
                return {"game": pending["game"], "decision": kind, "target": v["target"]}
        claimants = sorted(s for s in claims if s in options)
        if claimants:
            return {"game": pending["game"], "decision": kind, "target": rng.choice(claimants)}
        target = best_by(options, lambda s: speeches.get(s, 0), rng)
        return {"game": pending["game"], "decision": kind, "target": target}

    if kind == "inspect":
        seen = {r["seat"] for r in view.get("seerResults") or []}
        fresh = [s for s in options if s not in seen]
        target = best_by(fresh or options, lambda s: scores.get(s, 0) * 10 + speeches.get(s, 0), rng)
        return {"game": pending["game"], "decision": kind, "target": target}

    if kind == "protect":
        claimants = sorted(s for s in claims if s in options)
        if claimants:
            return {"game": pending["game"], "decision": kind, "target": rng.choice(claimants)}
        target = best_by(options, lambda s: speeches.get(s, 0), rng)
        return {"game": pending["game"], "decision": kind, "target": target}

    if kind == "vote":
        target = vote_choice(view, options, claims, scores, rng)
        return {"game": pending["game"], "decision": kind, "target": target}

    return fallback_for(pending)


def vote_choice(view, options, claims, scores, rng):
    you = view["you"]
    partner_seat = (you.get("partner") or {}).get("seat")
    wagon = current_wagon(view)
    if you["role"] == "seer":
        for r in view.get("seerResults") or []:
            if r["werewolf"] and r["seat"] in options:
                return r["seat"]
    if you["role"] != "werewolf":
        named = set()
        for seat, c in claims.items():
            if seat != partner_seat:
                named |= c["wolves"]
        live_named = sorted(s for s in named if s in options)
        if len(claims) == 1 and live_named:
            return live_named[0]
    if wagon:
        leaders = [s for s in options if wagon.get(s, 0) == max(wagon.values()) and wagon.get(s, 0) > 0]
        safe = [s for s in leaders if s != partner_seat]
        if safe:
            return safe[0]
        if leaders and leaders[0] != partner_seat:
            return leaders[0]
    pool = [s for s in options if s != partner_seat] or options
    return best_by(pool, lambda s: scores.get(s, 0), rng)


def heuristic_speak(pending, rng):
    view = pending["view"]
    you = view["you"]
    options_alive = [p["seat"] for p in view["players"] if p["alive"] and p["seat"] != you["seat"]]
    claims = claims_heard(view)
    scores, _ = suspicion(view)

    def say(text):
        folded = fold_statement(text)
        limit = pending.get("max_chars", 500)
        if len(folded) > limit:
            folded = folded[:limit].rstrip()
        return {"game": pending["game"], "decision": "speak", "text": folded}

    if you["role"] == "seer":
        results = view.get("seerResults") or []
        wolves = [r for r in results if r["werewolf"]]
        if wolves and view["day"] >= 2:
            r = wolves[-1]
            return say(
                f"I am the seer. Last night I checked {who(view, r['seat'])}: they are a werewolf. "
                f"Vote {who(view, r['seat'])} out today and we are halfway there."
            )
        if len(results) >= 2 and view["day"] >= 4:
            parts = "; ".join(
                f"{who(view, r['seat'])} {'is a werewolf' if r['werewolf'] else 'is clear'}" for r in results
            )
            return say(f"I am the seer, claiming now so my results are not lost: {parts}. Let us vote on the evidence.")
    if you["role"] != "werewolf" and len(claims) == 1:
        claimant, c = next(iter(claims.items()))
        if c["wolves"]:
            target = sorted(c["wolves"])[0]
            return say(
                f"{who(view, claimant)} claims seer and names {who(view, target)} as a wolf. "
                "Nobody has counter-claimed, so I will back that vote today."
            )
    pool = [s for s in options_alive if s != (you.get("partner") or {}).get("seat")]
    if pool:
        suspect = best_by(pool, lambda s: scores.get(s, 0), rng)
        if scores.get(suspect, 0) > 0:
            return say(
                f"Look at the record: {who(view, suspect)} has voted with the wolves' interests so far. "
                f"They are my main suspect today; I want to hear their defence."
            )
        return say(
            f"No hard evidence yet. I am watching {who(view, suspect)} closely; quiet seats help only the wolves, so everyone should talk today."
        )
    return fallback_for(pending)


# ---------------------------------------------------------------- the run


def rules_from(brief):
    parts = re.split(r"(?m)^(?=## )", brief)
    kept = [p.strip() for p in parts if p.startswith("## The game\n") or p.startswith("## The score\n")]
    return "\n\n".join(kept) if kept else brief.strip()


def decide(env, brain, pending, trial_seed, started_lock, stats):
    act = None
    if brain is not None:
        reply = brain.ask(pending["view"], pending)
        act = parse_reply(reply, pending)
    if act is None:
        act = heuristic(pending, trial_seed)
    status, _ = env.act(act)
    if status == 400 and act != fallback_for(pending):
        status, _ = env.act(fallback_for(pending))
    with started_lock:
        stats["answered" if status == 200 else "failed"] += 1


def main():
    env = Env(os.environ["MAKEFUN_ENV_URL"])
    with open(os.environ["MAKEFUN_PROMPT_FILE"], encoding="utf-8") as f:
        brief = f.read()
    trial = {
        "rules": rules_from(brief),
        "seed": os.environ.get("MAKEFUN_TRIAL_SEED", ""),
        "openai_base": os.environ.get("OPENAI_BASE_URL", ""),
        "openai_key": os.environ.get("OPENAI_API_KEY", ""),
    }
    brain = Brain(trial) if trial["openai_base"] else None
    started = set()
    in_flight = {}
    lock = threading.Lock()
    stats = {"answered": 0, "failed": 0}

    def key_of(p):
        return f"{p['game']}:{p['decision']}:{len(p['view']['log'])}"

    while True:
        status, obs = env.observe()
        if status != 200 or not isinstance(obs, dict):
            raise RuntimeError(f"POST /observe answered {status}")
        if obs.get("done"):
            break
        for p in obs.get("pending", []):
            key = key_of(p)
            with lock:
                if key in started or len(in_flight) >= CONCURRENCY:
                    continue
                started.add(key)
            t = threading.Thread(target=decide, args=(env, brain, p, trial["seed"], lock, stats), daemon=True)
            with lock:
                in_flight[key] = t
            t.start()
        with lock:
            live = [t for t in in_flight.values() if t.is_alive()]
            in_flight = {k: t for k, t in in_flight.items() if t.is_alive()}
        if live:
            live[0].join(timeout=0.25)
    for t in list(in_flight.values()):
        t.join()
    print(f"werewolf agent: {stats['answered']} decisions answered, {stats['failed']} failed; model calls: {brain.calls if brain else 0}")


if __name__ == "__main__":
    main()
