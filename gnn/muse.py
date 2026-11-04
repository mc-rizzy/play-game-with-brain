"""Meta Model API client: Muse is integral to the brain-to-game pipeline.

Every demo run calls Meta's hosted Muse models (default
``muse-spark-1.3``) through the OpenAI-compatible endpoint
``https://api.meta.ai/v1`` — no extra SDK needed, stdlib only:

1. **Calibration coach** (:meth:`MuseClient.analyze_calibration`): reads the
   judge's fresh 15-second EEG summary (per-command band powers, class
   separability, adaptation fit) and returns personalized coaching plus
   tuned live-gate thresholds. The GNN learns *patterns*; Muse explains
   them and tunes the game to this brain.
2. **Intent referee** (:meth:`MuseClient.interpret_command`, optional live
   fallback): when the GNN is in the gray zone, Muse disambiguates intent
   from game context + recent history instead of dropping the input.

Auth: ``META_TOKEN`` (hackathon key in ``.env``) or ``MODEL_API_KEY`` env.
Without a key/network the pipeline still runs via offline fallbacks — but
judges should see the live calls: each call is timed, logged to stdout,
and stored in the calibration report (``muse.calls`` with model name).
"""

import json
import os
import time
import urllib.error
import urllib.request

META_API_BASE = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"

CALIBRATION_SYSTEM = (
    "You are the calibration coach for a brain-controlled video game. "
    "A player just did a 15-second EEG calibration (8 electrodes, motor "
    "imagery commands). Given their signal summary, reply with JSON only: "
    '{"coaching": ["<2-4 short tips>"], '
    '"thresholds": {"confidence": <0.45-0.8>, "cooldown_s": <0.3-0.8>, '
    '"ema": <0.5-0.85>}, "verdict": "<one sentence>"} '
    "Raise confidence when separability is low, lower it when the fit is "
    "clean so the game feels responsive."
)

REFEREE_SYSTEM = (
    "You referee a brain-controlled game. The EEG classifier is uncertain. "
    "Given recent predictions and game context, reply with JSON only: "
    '{"command": "<UP|DOWN|LEFT|RIGHT|INTERACT|ABSTAIN>", '
    '"reason": "<under 12 words>"}. ABSTAIN unless the context clearly '
    "supports one command — a wrong move is worse than no move."
)


class MuseError(RuntimeError):
    """Meta Model API call failed (key, network, or API error)."""


def _ssl_context():
    """TLS context: certifi bundle when installed, else platform default."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def load_dotenv(root=None):
    """Minimal KEY=VALUE .env parser (no dependency). Returns dict."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    try:
        with open(os.path.join(root, ".env")) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def api_key():
    """Hackathon key: env META_TOKEN/MODEL_API_KEY, else .env (or None)."""
    for var in ("META_TOKEN", "MODEL_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    file_vars = load_dotenv()
    return file_vars.get("META_TOKEN") or file_vars.get("MODEL_API_KEY")


def _extract_json(text):
    """Parse the first {...} JSON object in free-form model text."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


class MuseClient:
    """Thin Meta Model API client (OpenAI-compatible chat completions)."""

    def __init__(self, api_key=None, model=DEFAULT_MODEL,
                 base_url=META_API_BASE, timeout=30):
        self.key = api_key if api_key is not None else globals()["api_key"]()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.calls = []  # audit trail: every Meta model call, for judges

    @property
    def available(self):
        return bool(self.key)

    def chat(self, messages, max_tokens=1024, temperature=0.3):
        """POST chat completions -> reply text (raises MuseError).

        Budgets include hidden reasoning tokens (Muse Spark reasons
        before answering); keep max_tokens >= ~400 for short replies.
        """
        if not self.key:
            raise MuseError("no API key: set META_TOKEN in .env")
        body = json.dumps({"model": self.model, "messages": messages,
                           "max_tokens": max_tokens,
                           "temperature": temperature,
                           "reasoning_effort": "low"}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                        context=_ssl_context()) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            self.calls.append({"model": self.model, "ok": False,
                               "error": f"HTTP {exc.code}: {detail}"})
            raise MuseError(f"Meta API HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            self.calls.append({"model": self.model, "ok": False,
                               "error": f"{type(exc).__name__}: {exc}"})
            raise MuseError(f"Meta API call failed: {exc}") from exc
        latency = time.time() - t0
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise MuseError(f"unexpected API response: {payload!r}") from exc
        if not text:
            raise MuseError("empty reply from model "
                            "(reasoning consumed max_tokens?)")
        self.calls.append({"model": self.model, "ok": True,
                           "latency_s": round(latency, 2)})
        print(f"[muse] {self.model} replied in {latency:.1f}s "
              f"({len(text)} chars)")
        return text

    # ------------------------------------------------ high-level pipeline
    def analyze_calibration(self, summary):
        """Muse coaches the player + tunes gates. Never raises.

        Returns {"coaching": [...], "thresholds": {...}, "verdict": str,
        "offline": bool}. Offline fallback uses separability heuristics.
        """
        sep = float(summary.get("separability", 0.5))
        try:
            text = self.chat(
                [{"role": "system", "content": CALIBRATION_SYSTEM},
                 {"role": "user",
                  "content": "Calibration summary:\n"
                             + json.dumps(summary, indent=1)
                             + "\nReply with JSON only."}],
                max_tokens=1600)
            data = _extract_json(text)
            raw = data.get("thresholds", {})
            result = {
                "coaching": [str(c) for c in data.get("coaching", [])][:4],
                "thresholds": {
                    "confidence": _clamp(raw.get("confidence"), 0.45, 0.8,
                                         0.60),
                    "cooldown_s": _clamp(raw.get("cooldown_s"), 0.3, 0.8,
                                         0.5),
                    "ema": _clamp(raw.get("ema"), 0.5, 0.85, 0.7)},
                "verdict": str(data.get("verdict", ""))[:200],
                "offline": False, "model": self.model}
            if not result["coaching"]:
                result["coaching"] = [result["verdict"] or "Looking good — "
                                      "play on!"]
            return result
        except Exception as exc:
            print(f"[muse] offline fallback ({exc}); "
                  f"key={'set' if self.key else 'missing'}")
            # Heuristic fallback: uncertain signals get stricter gates.
            conf = 0.70 if sep < 0.4 else (0.55 if sep > 0.7 else 0.60)
            return {
                "coaching": ["Imagine the movement vividly, keep eyes "
                             "forward — calibration kept defaults."],
                "thresholds": {"confidence": conf, "cooldown_s": 0.5,
                               "ema": 0.7},
                "verdict": "Offline defaults applied (Meta API unreachable).",
                "offline": True, "model": self.model}

    def interpret_command(self, context):
        """Gray-zone referee -> {"command": ..., "reason": ...}.

        Never raises: falls back to ABSTAIN (drop the uncertain input).
        """
        try:
            text = self.chat(
                [{"role": "system", "content": REFEREE_SYSTEM},
                 {"role": "user",
                  "content": "Game context:\n"
                             + json.dumps(context, indent=1)
                             + "\nReply with JSON only."}],
                max_tokens=800, temperature=0.1)
            data = _extract_json(text)
            cmd = str(data.get("command", "ABSTAIN")).upper()
            if cmd not in ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT",
                           "ABSTAIN"):
                cmd = "ABSTAIN"
            return {"command": cmd,
                    "reason": str(data.get("reason", ""))[:120],
                    "offline": False}
        except Exception as exc:
            print(f"[muse] referee offline ({exc}) -> ABSTAIN")
            return {"command": "ABSTAIN", "reason": "referee unreachable",
                    "offline": True}

    def ping(self):
        """Minimal live check -> True (raises MuseError on failure)."""
        text = self.chat([{"role": "user",
                           "content": "Reply with exactly: ok"}],
                         max_tokens=400, temperature=0.0)
        return "ok" in text.lower()
