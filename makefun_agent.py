"""makefun agent/v1 entry shim for Python 3.13: standard library only, one file.

The Python twin of agent/sdk/makefun-agent.ts: the same trial, the same seeded RNG (the same values for
the same seed and label), the same store and model clients. The grader runs a Python entry as
``python3 -P -s /bot/main.py`` with ``PYTHONHASHSEED=0``; -P leaves the script's directory off sys.path,
so copy this file to vendor/makefun_agent.py and put that directory on it::

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    import makefun_agent as af

    trial = af.read_trial()                 # the brief, the URLs, the shop token, the seed, the now
    store = af.StoreClient(trial)           # the shop's Admin GraphQL and the supplier's REST API
    llm = af.Anthropic(trial)               # the trial's gateway, the only route to a model
    shop = store.request("{ shop { name currencyCode } }")["shop"]
    reply = llm.messages({"model": "claude-haiku-5-5", "max_tokens": 1024, "messages": [{"role": "user", "content": trial.prompt}]})

What a trial hands the entry (read_trial reads it; TrialConfigError names what is missing):
  MAKEFUN_PROMPT_FILE        the merchant brief, a UTF-8 text file: trial.prompt holds its text
  MAKEFUN_SHOP_GRAPHQL_URL   the shop's Admin GraphQL endpoint (POST, X-Shopify-Access-Token)
  MAKEFUN_SHOP_TOKEN         the shop's access token
  MAKEFUN_SUPPLIER_URL       the supplier's REST API root; when unset, /supplier/v1 on the shop's origin
  MAKEFUN_TRIAL_SEED         the trial's seed, to seed every random choice from (trial_rng)
  MAKEFUN_TRIAL_NOW          the trial's fixed now: an RFC 3339 instant, or epoch milliseconds (trial_now)
  MAKEFUN_ENV_URL            the env's origin, whatever the env simulates; when unset, the shop's origin (env_url)
  ANTHROPIC_BASE_URL          the gateway for Anthropic Messages (default https://api.anthropic.com)
  OPENAI_BASE_URL             the gateway for OpenAI Chat Completions (default https://api.openai.com/v1)
  ANTHROPIC_API_KEY, OPENAI_API_KEY   the trial's key, makefun-trial-<48 hex>, the same on every pass:
                                      the gateway admits that key alone, drops it and sends the real one

The determinism rule. Every trial runs twice: live, where each model call goes through the gateway and is
recorded, then replay, where the gateway answers each request from that recording by the request's
content, and the store must end byte-equal or the run is nondeterministic and unranked. So nothing that
differs between two runs may reach a request, to the model, the shop or the supplier:
  - no wall clock: never put time.time(), datetime.now() or date.today() into a request; take any date
    from trial_now(), the trial's fixed now;
  - no unseeded randomness: never the random module's global functions, uuid4(), secrets or os.urandom();
    draw from trial_rng(trial, label), the same values in both runs (and in makefun-agent.ts);
  - one order: a StoreClient sends one request at a time (a lock serialises threads), in the order they
    were made. Model calls may run in parallel (the replay serves identical requests in their recorded
    order), but what you do with the answers must not depend on which came back first.
Waiting is fine (time may pass); a time value must never be written into anything sent. This file never
reads a clock or a random source itself (time.sleep only waits). HTTP goes through urllib with proxies
off: a trial passes base URLs, never a proxy.

The shop's rate limit is the Admin API's leaky bucket on the store's own clock: each Admin request is one
simulated second, restoring 50 of 1,000 points, and waiting restores nothing. A THROTTLED answer is retried
at once, each retry restoring the next 50, up to max_throttle_retries; every request, a retry or a refusal
too, counts against the trial's 5,000. The model clients retry 408, 429, 5xx and 529 answers with fixed
backoff (never a refusal of the gateway's: budget_exhausted, request_cap, model_not_allowed,
max_tokens_required, which raise LlmRefusalError) and never a connection that failed before an answer,
since a request the gateway never saw has no recording to replay.
"""

import calendar
import datetime
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HARNESS = "agent/v1"

ENV_PROMPT_FILE = "MAKEFUN_PROMPT_FILE"
ENV_SHOP_GRAPHQL_URL = "MAKEFUN_SHOP_GRAPHQL_URL"
ENV_SHOP_TOKEN = "MAKEFUN_SHOP_TOKEN"
ENV_SUPPLIER_URL = "MAKEFUN_SUPPLIER_URL"
ENV_TRIAL_SEED = "MAKEFUN_TRIAL_SEED"
ENV_TRIAL_NOW = "MAKEFUN_TRIAL_NOW"
ENV_ENV_URL = "MAKEFUN_ENV_URL"
ENV_ANTHROPIC_BASE_URL = "ANTHROPIC_BASE_URL"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
#: The supplier API's path on the shop's origin, when MAKEFUN_SUPPLIER_URL is unset.
SUPPLIER_PATH = "/supplier/v1"
#: The gateway's own refusals (its error ``type``): final, never retried.
GATEWAY_REFUSALS = ("budget_exhausted", "request_cap", "model_not_allowed", "max_tokens_required")
#: The model answers retried with backoff.
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
#: The domain separating trial_rng's streams from any other use of the seed.
RNG_DOMAIN = "agent-fun/agent/rng/v1"

#: Seconds a single HTTP exchange may take before it fails (urllib's socket timeout).
HTTP_TIMEOUT_S = 600

# ---------------------------------------------------------------- the trial


class TrialConfigError(Exception):
    """The trial's environment is missing a variable or holds one this shim cannot read."""


class LlmEndpoint:
    """An API root (no trailing slash) and the key sent with it ('' sends none; in a trial, the trial's key, which the gateway swaps for the real one)."""

    __slots__ = ("base_url", "api_key")

    def __init__(self, base_url, api_key=""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def __repr__(self):
        return f"LlmEndpoint(base_url={self.base_url!r})"


class Trial:
    """Everything the trial hands the entry; read_trial() builds it."""

    __slots__ = ("prompt", "prompt_file", "shop_graphql_url", "shop_token", "supplier_url", "env_url", "seed", "now_ms", "now_iso", "anthropic", "openai")

    def __init__(self, prompt, prompt_file, shop_graphql_url, shop_token, supplier_url, seed, now_ms, anthropic, openai, env_url=None):
        #: The merchant brief: the prompt file's text (UTF-8, a leading BOM dropped).
        self.prompt = prompt
        self.prompt_file = prompt_file
        self.shop_graphql_url = shop_graphql_url
        self.shop_token = shop_token
        #: The supplier's REST API root, no trailing slash.
        self.supplier_url = supplier_url.rstrip("/")
        #: The env's origin, no trailing slash: MAKEFUN_ENV_URL, or the shop's origin when unset.
        self.env_url = (_origin(shop_graphql_url) if env_url is None else env_url).rstrip("/")
        #: MAKEFUN_TRIAL_SEED as given.
        self.seed = seed
        #: The trial's now: epoch milliseconds, and as YYYY-MM-DDTHH:MM:SS.sssZ.
        self.now_ms = now_ms
        self.now_iso = format_ms(now_ms)
        self.anthropic = anthropic
        self.openai = openai

    def __repr__(self):
        return f"Trial(shop_graphql_url={self.shop_graphql_url!r}, seed={self.seed!r}, now_iso={self.now_iso!r})"


def _required(env, name):
    v = env.get(name)
    if v is None or not v.strip():
        raise TrialConfigError(f"{name} is not set: run the entry under the agent/v1 harness (or set it as RULES.txt describes)")
    return v


def _http_url(name, text):
    try:
        u = urllib.parse.urlsplit(text)
    except ValueError:
        raise TrialConfigError(f"{name} is not a URL: {json.dumps(text)}") from None
    if u.scheme not in ("http", "https") or not u.netloc:
        raise TrialConfigError(f"{name} is not an http(s) URL: {json.dumps(text)}")
    return text


def _origin(url):
    u = urllib.parse.urlsplit(url)
    return f"{u.scheme}://{u.netloc}"


_RFC3339 = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})")
_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
#: The latest instant a JavaScript Date holds, in ms (the TypeScript twin's bound).
_MAX_MS = 8_640_000_000_000_000


def parse_trial_now(text):
    """A trial now (an RFC 3339 instant with Z or an offset, or epoch milliseconds) as epoch ms; fractions past ms are cut."""
    if re.fullmatch(r"\d{1,16}", text) and int(text) <= _MAX_MS:
        return int(text)
    m = _RFC3339.fullmatch(text)

    def bad():
        return TrialConfigError(f"{ENV_TRIAL_NOW} is not an RFC 3339 instant (2026-11-02T09:00:00Z) or epoch milliseconds: {json.dumps(text)}")

    if m is None:
        raise bad()
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    if y < 1970 or not 1 <= mo <= 12 or d < 1 or d > (29 if mo == 2 and leap else _DAYS[mo - 1]) or h > 23 or mi > 59 or s > 59:
        raise bad()
    ms = int((m.group(7) or "").ljust(3, "0")[:3])
    offset_min = 0
    if m.group(8) != "Z":
        oh, om = int(m.group(8)[1:3]), int(m.group(8)[4:6])
        if oh > 23 or om > 59:
            raise bad()
        offset_min = (-1 if m.group(8).startswith("-") else 1) * (oh * 60 + om)
    at = calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * 1000 + ms - offset_min * 60_000
    if at < 0:
        raise bad()
    return at


_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def _at(ms):
    return _EPOCH + datetime.timedelta(milliseconds=ms)


def format_ms(ms):
    """Epoch milliseconds as YYYY-MM-DDTHH:MM:SS.sssZ, what JavaScript's toISOString() writes."""
    return f"{_at(ms):%Y-%m-%dT%H:%M:%S}.{ms % 1000:03d}Z"


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_trial(env=None, read_file=_read_bytes):
    """Everything the trial hands the entry (see the module doc), read once; raises TrialConfigError."""
    env = os.environ if env is None else env
    prompt_file = _required(env, ENV_PROMPT_FILE)
    try:
        prompt = read_file(prompt_file).decode("utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise TrialConfigError(f"cannot read the prompt {prompt_file} ({ENV_PROMPT_FILE}) as UTF-8 text: {err}") from None
    if prompt.startswith("\ufeff"):
        prompt = prompt[1:]
    shop_graphql_url = _http_url(ENV_SHOP_GRAPHQL_URL, _required(env, ENV_SHOP_GRAPHQL_URL))
    shop_token = _required(env, ENV_SHOP_TOKEN)
    supplier_text = env.get(ENV_SUPPLIER_URL)
    if supplier_text is None or not supplier_text.strip():
        supplier_url = f"{_origin(shop_graphql_url)}{SUPPLIER_PATH}"
    else:
        supplier_url = _http_url(ENV_SUPPLIER_URL, supplier_text)
    env_text = env.get(ENV_ENV_URL)
    seed = _required(env, ENV_TRIAL_SEED)
    now_ms = parse_trial_now(_required(env, ENV_TRIAL_NOW))

    def endpoint(base, key, fallback):
        b = env.get(base)
        return LlmEndpoint(fallback if b is None or not b.strip() else _http_url(base, b), env.get(key) or "")

    return Trial(
        prompt,
        prompt_file,
        shop_graphql_url,
        shop_token,
        supplier_url,
        seed,
        now_ms,
        endpoint(ENV_ANTHROPIC_BASE_URL, ENV_ANTHROPIC_API_KEY, DEFAULT_ANTHROPIC_BASE_URL),
        endpoint(ENV_OPENAI_BASE_URL, ENV_OPENAI_API_KEY, DEFAULT_OPENAI_BASE_URL),
        env_url=None if env_text is None or not env_text.strip() else _http_url(ENV_ENV_URL, env_text),
    )


def trial_now(trial=None, env=None):
    """The trial's fixed now as an aware UTC datetime: the trial's own, or MAKEFUN_TRIAL_NOW's. Never the wall clock."""
    return _at(trial.now_ms if trial is not None else parse_trial_now(_required(os.environ if env is None else env, ENV_TRIAL_NOW)))


def env_url(trial=None, env=None):
    """The env's origin, no trailing slash: the trial's own, or MAKEFUN_ENV_URL's (an entry that reads no trial)."""
    return trial.env_url if trial is not None else _http_url(ENV_ENV_URL, _required(os.environ if env is None else env, ENV_ENV_URL)).rstrip("/")


# ---------------------------------------------------------------- the seeded RNG

_MASK64 = (1 << 64) - 1


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Rng:
    """xoshiro256** seeded from sha256(RNG_DOMAIN, 0, seed, 0, label).

    The same seed and label give the same values here and in makefun-agent.ts's Rng. Use one label per
    purpose, so adding draws to one stream never moves another.
    """

    def __init__(self, seed, label=""):
        if isinstance(seed, bool) or not isinstance(seed, (str, int)):
            raise TypeError(f"an Rng seed is a str or an int, got {type(seed).__name__}")
        self.seed = str(seed)
        self.label = label
        h = hashlib.sha256(RNG_DOMAIN.encode("utf-8") + b"\0" + self.seed.encode("utf-8") + b"\0" + label.encode("utf-8")).digest()
        s = [int.from_bytes(h[i : i + 8], "big") for i in range(0, 32, 8)]
        if not any(s):
            s[0] = 1
        self._s = s

    def next_u64(self):
        """The next 64 random bits."""
        s0, s1, s2, s3 = self._s
        result = (_rotl((s1 * 5) & _MASK64, 7) * 9) & _MASK64
        t = (s1 << 17) & _MASK64
        a2 = s2 ^ s0
        a3 = s3 ^ s1
        self._s = [s0 ^ a3, s1 ^ a2, a2 ^ t, _rotl(a3, 45)]
        return result

    def random(self):
        """A float in [0, 1) from the top 53 bits."""
        return (self.next_u64() >> 11) / 9_007_199_254_740_992

    def randint(self, lo, hi):
        """An integer in [lo, hi], without modulo bias (the TypeScript twin's int(min, max))."""
        if type(lo) is not int or type(hi) is not int or lo > hi:
            raise ValueError(f"Rng.randint needs ints lo <= hi, got {lo!r}, {hi!r}")
        span = hi - lo + 1
        if span > 1 << 64:
            raise ValueError("Rng.randint spans more than 2**64 values")
        limit = (1 << 64) - (1 << 64) % span
        while True:
            x = self.next_u64()
            if x < limit:
                return lo + x % span

    def choice(self, items):
        """One of the items, uniformly."""
        if not items:
            raise IndexError("Rng.choice of an empty sequence")
        return items[self.randint(0, len(items) - 1)]

    def shuffle(self, items):
        """Shuffle a list in place (Fisher-Yates, from the end) and return it."""
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]
        return items


def trial_rng(trial, label=""):
    """The trial's RNG stream named ``label``: the only randomness that may reach a request."""
    return Rng(trial.seed, label)


# ---------------------------------------------------------------- HTTP

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _NoAnswer(Exception):
    """The exchange failed before any HTTP answer."""


def _exchange(method, url, headers, body, timeout=HTTP_TIMEOUT_S):
    """One HTTP exchange: (status, body bytes). Raises _NoAnswer when no answer came."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        finally:
            err.close()
    except (urllib.error.URLError, OSError) as err:
        raise _NoAnswer(str(getattr(err, "reason", err))) from None


def _dumps(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------- the store: Admin GraphQL and the supplier


class StoreHttpError(Exception):
    """The shop or the supplier answered something that is not its API's answer (a 4xx/5xx from the shop, a body that is not JSON, no answer)."""

    def __init__(self, message, status, body):
        super().__init__(message)
        self.status = status
        self.body = body


class GraphQLRequestError(Exception):
    """A GraphQL answer carrying ``errors`` (StoreClient.request): the whole response is kept."""

    def __init__(self, response):
        errors = response.get("errors") or []
        parts = []
        for e in errors:
            code = (e.get("extensions") or {}).get("code") if isinstance(e, dict) else None
            msg = e.get("message", "") if isinstance(e, dict) else str(e)
            parts.append(f"{code}: {msg}" if isinstance(code, str) else msg)
        super().__init__("; ".join(parts) or "GraphQL errors")
        self.errors = errors
        self.response = response


def _throttled(response):
    return any(isinstance(e, dict) and (e.get("extensions") or {}).get("code") == "THROTTLED" for e in response.get("errors") or [])


def _parse_json(status, raw, what):
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        raise StoreHttpError(f"{what} answered {status} with a body that is not JSON: {text[:200]}", status, text) from None


class StoreClient:
    """The shop's Admin GraphQL API and the supplier's REST API, one request at a time.

    A lock makes every call wait for the one before it, so calls reach the store in the order they were
    made (the determinism rule). Use one StoreClient per trial.
    """

    def __init__(self, trial, max_throttle_retries=25, exchange=None):
        self.graphql_url = trial.shop_graphql_url
        self.supplier_url = trial.supplier_url.rstrip("/")
        self._token = trial.shop_token
        self._max_throttle_retries = max_throttle_retries
        self._exchange = exchange or _exchange
        self._lock = threading.Lock()
        #: HTTP requests sent so far, retries included (the store takes 5,000 per trial).
        self.requests = 0

    def _send(self, method, url, headers, body, what):
        self.requests += 1
        try:
            return self._exchange(method, url, headers, body)
        except _NoAnswer as err:
            raise StoreHttpError(f"{what}: no answer from {url}: {err}", 0, "") from None

    def graphql(self, query, variables=None, operation_name=None):
        """One GraphQL operation, its whole answer dict (data, errors, extensions.cost); THROTTLED is retried.

        Raises StoreHttpError when the shop answers other than 200.
        """
        payload = {"query": query}
        if variables is not None:
            payload["variables"] = variables
        if operation_name is not None:
            payload["operationName"] = operation_name
        body = _dumps(payload)
        headers = {"Content-Type": "application/json", "Accept": "application/json", "X-Shopify-Access-Token": self._token}
        with self._lock:
            attempt = 0
            while True:
                status, raw = self._send("POST", self.graphql_url, headers, body, "the shop")
                out = _parse_json(status, raw, "the shop")
                if status != 200:
                    raise StoreHttpError(f"the shop answered {status}: {raw.decode('utf-8', errors='replace')[:300]}", status, raw.decode("utf-8", errors="replace"))
                if not isinstance(out, dict):
                    raise StoreHttpError(f"the shop answered a JSON {type(out).__name__}, not a GraphQL response", status, raw.decode("utf-8", errors="replace"))
                if not _throttled(out) or attempt >= self._max_throttle_retries:
                    return out
                attempt += 1

    def request(self, query, variables=None, operation_name=None):
        """One GraphQL operation's data; raises GraphQLRequestError when the answer carries errors. userErrors are data: check them."""
        r = self.graphql(query, variables, operation_name)
        if r.get("errors"):
            raise GraphQLRequestError(r)
        return r.get("data")

    def supplier(self, method, path, query=None, body=None):
        """One supplier request: {"status": int, "body": the JSON answer}.

        ``path`` is under the supplier root ('/products', '/products/NW-100', '/quotes'), ``query`` its
        parameters in the order given (None values left out). Any status comes back with its body (a 409 is
        an answer: out of stock); a body that is not JSON raises StoreHttpError.
        """
        if method not in ("GET", "POST"):
            raise ValueError(f"a supplier request is GET or POST, got {method!r}")
        if path and not path.startswith("/"):
            raise ValueError(f"a supplier path starts with '/', got {path!r}")
        pairs = []
        for k, v in (query or {}).items():
            if v is None:
                continue
            pairs.append((k, ("true" if v else "false") if isinstance(v, bool) else str(v)))
        qs = urllib.parse.urlencode(pairs)
        url = f"{self.supplier_url}{path}{'?' + qs if qs else ''}"
        data = _dumps({} if body is None else body) if method == "POST" else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        with self._lock:
            status, raw = self._send(method, url, headers, data, "the supplier")
            return {"status": status, "body": _parse_json(status, raw, "the supplier")}

    def supplier_get(self, path, query=None):
        return self.supplier("GET", path, query=query)

    def supplier_post(self, path, body):
        return self.supplier("POST", path, body=body)


# ---------------------------------------------------------------- the model clients


class LlmError(Exception):
    """A model call that failed: an error answer (status, the vendor's error type) or no answer at all (status 0)."""

    def __init__(self, message, route, status, type_, body):
        super().__init__(message)
        self.route = route
        self.status = status
        #: The error's ``type`` (or ``code``) from the vendor's error shape, when there is one.
        self.type = type_
        self.body = body


class LlmRefusalError(LlmError):
    """The gateway refused the call (the trial's budget, request cap, model allowlist or max_tokens rule): final."""


def _error_of(route, status, raw):
    text = raw.decode("utf-8", errors="replace")
    try:
        body = json.loads(text)
    except ValueError:
        body = text
    e = body.get("error") if isinstance(body, dict) else None
    o = e if isinstance(e, dict) else {}
    type_ = o.get("type") if isinstance(o.get("type"), str) else o.get("code") if isinstance(o.get("code"), str) else None
    message = o.get("message") if isinstance(o.get("message"), str) else body[:300] if isinstance(body, str) else ""
    what = f"{route} answered {status}{'' if type_ is None else ' ' + type_}{'' if not message else ': ' + message}"
    refusal = type_ in GATEWAY_REFUSALS or o.get("code") in GATEWAY_REFUSALS
    return (LlmRefusalError if refusal else LlmError)(what, route, status, type_, body)


def _endpoint_of(source, which):
    if isinstance(source, LlmEndpoint):
        return LlmEndpoint(source.base_url, source.api_key)
    e = getattr(source, which)
    return LlmEndpoint(e.base_url, e.api_key)


def _positive_int(v):
    return type(v) is int and v > 0


class _LlmClient:
    def __init__(self, source, which, max_retries=2, retry_delay_s=1.0, sleep=None, exchange=None):
        self.endpoint = _endpoint_of(source, which)
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._sleep = sleep or time.sleep
        self._exchange = exchange or _exchange

    def _call(self, route, method, url, headers, body):
        attempt = 0
        while True:
            try:
                status, raw = self._exchange(method, url, headers, body)
            except _NoAnswer as err:
                raise LlmError(f"{route}: cannot reach {url}: {err}", route, 0, None, None) from None
            if 200 <= status < 300:
                try:
                    return json.loads(raw.decode("utf-8"))
                except ValueError:
                    raise LlmError(f"{route} answered {status} with a body that is not JSON", route, status, None, raw.decode("utf-8", errors="replace")) from None
            err = _error_of(route, status, raw)
            if isinstance(err, LlmRefusalError) or status not in RETRY_STATUSES or attempt >= self._max_retries:
                raise err
            self._sleep(self._retry_delay_s * 2**attempt)
            attempt += 1


class Anthropic(_LlmClient):
    """Anthropic Messages (POST {base_url}/v1/messages), JSON only: no streaming."""

    def __init__(self, source, version=ANTHROPIC_VERSION, beta=(), **opts):
        super().__init__(source, "anthropic", **opts)
        self.version = version
        self.beta = tuple(beta)

    def _headers(self, json_body):
        h = {"Accept": "application/json", "anthropic-version": self.version}
        if json_body:
            h["Content-Type"] = "application/json"
        if self.beta:
            h["anthropic-beta"] = ",".join(self.beta)
        if self.endpoint.api_key:
            h["x-api-key"] = self.endpoint.api_key
        return h

    def messages(self, body):
        """One message. max_tokens is required (the gateway refuses a call without it); stream is refused."""
        if not _positive_int(body.get("max_tokens")):
            raise ValueError("Anthropic.messages: max_tokens must be a positive int (the gateway refuses a call without it)")
        if body.get("stream") is True:
            raise ValueError("Anthropic.messages: this minimal client takes JSON answers only; stream is not supported")
        return self._call("anthropic.messages", "POST", f"{self.endpoint.base_url}/v1/messages", self._headers(True), _dumps(body))

    def models(self):
        """The models the endpoint serves (in a trial: the competition's allowlist)."""
        return self._call("models", "GET", f"{self.endpoint.base_url}/v1/models", self._headers(False), None)


class OpenAI(_LlmClient):
    """OpenAI Chat Completions (POST {base_url}/chat/completions), JSON only: no streaming."""

    def __init__(self, source, **opts):
        super().__init__(source, "openai", **opts)

    def _headers(self, json_body):
        h = {"Accept": "application/json"}
        if json_body:
            h["Content-Type"] = "application/json"
        if self.endpoint.api_key:
            h["Authorization"] = f"Bearer {self.endpoint.api_key}"
        return h

    def chat(self, body):
        """One chat completion. max_completion_tokens (or max_tokens) is required; stream is refused."""
        if not _positive_int(body.get("max_completion_tokens")) and not _positive_int(body.get("max_tokens")):
            raise ValueError("OpenAI.chat: max_completion_tokens (or max_tokens) must be a positive int (the gateway refuses a call without it)")
        if body.get("stream") is True:
            raise ValueError("OpenAI.chat: this minimal client takes JSON answers only; stream is not supported")
        return self._call("openai.chat", "POST", f"{self.endpoint.base_url}/chat/completions", self._headers(True), _dumps(body))

    def models(self):
        """The models the endpoint serves (in a trial: the competition's allowlist)."""
        return self._call("models", "GET", f"{self.endpoint.base_url}/models", self._headers(False), None)
