"""`localevidence serve` — a zero-dependency local backend + PWA host.

A single-process HTTP server that loads the passage index, the knowledge
ledger, and the embedding model once at startup and keeps them warm, so the
phone (a thin PWA client) can:
  - read every previously-worked answer (offline-cacheable),
  - ask a question: a warm/similar one returns its worked answer instantly; a
    novel one returns the live evidence (retrieval) and is QUEUED for the next
    home deep-run (synthesis stays a Claude-in-the-loop step, no API).

Stdlib only. Single-threaded on purpose: it's a personal service, requests
serialise, and that sidesteps SQLite thread-safety entirely. Bind to localhost
(default) or the tailnet; never expose it to the open internet.
"""

from __future__ import annotations

import datetime as dt
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import config

WEBAPP = Path(__file__).resolve().parent / "webapp"
QUEUE_PATH = config.ROOT / "ledger" / "queue.jsonl"

_CTYPE = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
          ".json": "application/manifest+json", ".svg": "image/svg+xml",
          ".css": "text/css", ".png": "image/png"}

# Loaded once at startup, kept warm.
_INDEX = None
_LEDGER = None


def _boot(verbose: bool = True):
    global _INDEX, _LEDGER
    from .index import PassageIndex
    from .ledger import Ledger
    if verbose:
        print("  loading passage index + ledger + model ...", flush=True)
    _INDEX = PassageIndex()
    _LEDGER = Ledger()
    # warm the embedding model so the first /api/ask isn't slow
    from . import embedding
    embedding.get_model()
    if verbose:
        print(f"  ready: {_INDEX.stats()['passages']} passages, "
              f"{_LEDGER.stats()['answered']} answered questions", flush=True)


# Per-process topic dedupe for acquire-on-miss: one pull serves every card in a
# topic during a run, so an audit batch never re-pulls the same literature.
_ACQUIRED_TOPICS: set = set()


def _make_acquirer(max_pulls: int = 4, oa_only: bool = True):
    """An acquire-on-miss function for verify_evidence. Pulls (OA-first, capped)
    via the ask() acquisition path, refreshes the warm index in place, and reports
    how many new papers landed. Nightly/unattended use stays OA-only; the shadow
    tier is reached only when an operator has wired a provider AND passes oa_only
    False explicitly."""
    def _acquire(topic: str) -> dict:
        key = topic.lower().strip()
        if key in _ACQUIRED_TOPICS:
            return {"pulled": 0, "topic": topic, "note": "topic already acquired this run"}
        before = _INDEX.stats().get("papers", 0)
        try:
            from .pipeline import ask as _ask
            _ask(topic, top_n=max_pulls, oa_only=oa_only, verbose=False)
        except Exception as e:  # acquisition is best-effort; never break the verify
            # NOT marked acquired: a failed pull (transient network/error) can retry
            # on a later request instead of being suppressed for the server's lifetime.
            return {"pulled": 0, "topic": topic, "error": str(e)}
        _ACQUIRED_TOPICS.add(key)  # only after a successful pull
        _INDEX.reload()
        return {"pulled": max(0, _INDEX.stats().get("papers", 0) - before), "topic": topic}
    return _acquire


def handle_verify(body: dict) -> dict:
    """Pure-ish entry for POST /api/verify-evidence (also unit-tested directly)."""
    from . import verify
    claim = body.get("claim") or {}
    if not isinstance(claim, dict) or not (str(claim.get("text") or "")).strip():
        return {"error": "empty claim.text"}
    opts = body.get("options") or {}
    if not isinstance(opts, dict):
        return {"error": "options must be an object"}
    # Validate + clamp numeric options so malformed JSON yields a 400, not a 500.
    try:
        k = max(1, min(50, int(opts.get("k", 8))))
        min_conf = max(0.0, min(1.0, float(opts.get("min_confidence", 0.45))))
        importance = max(1, min(3, int(opts.get("importance", 1))))
    except (TypeError, ValueError):
        return {"error": "k/min_confidence/importance must be numeric"}
    return verify.verify_evidence(
        claim, index=_INDEX, citation=body.get("citation"),
        k=k, acquire_on_miss=bool(opts.get("acquire_on_miss", False)),
        min_confidence=min_conf, importance=importance,
        acquirer=_make_acquirer(oa_only=not opts.get("allow_shadow", False))
        if opts.get("acquire_on_miss") else None)


def _queue_question(question: str) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("a") as fh:
        fh.write(json.dumps({"question": question,
                             "ts": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")


def _read_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    return [json.loads(l) for l in QUEUE_PATH.read_text().splitlines() if l.strip()]


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalEvidence/0.1"

    # -- helpers -------------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        # NO cross-origin headers. The PWA is served from this same origin, so it
        # needs none; emitting `Access-Control-Allow-Origin: *` would let ANY web
        # page the user has open read their ledger at 127.0.0.1 cross-origin. The
        # browser blocks cross-origin reads of these responses precisely because
        # we don't advertise CORS. (There is no auth — keep this on localhost /
        # your tailnet, never the open internet.)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def log_message(self, *a):  # quieter
        pass

    # -- routing -------------------------------------------------------------

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            return self._api_get(path)
        return self._static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/ask":
            return self._ask()
        if path == "/api/verify-evidence":
            return self._verify()
        self._json({"error": "not found"}, 404)

    def _verify(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        out = handle_verify(body)
        return self._json(out, 400 if out.get("error") else 200)

    # -- static (the PWA) ----------------------------------------------------

    def _static(self, path: str):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        fp = (WEBAPP / rel).resolve()
        if WEBAPP not in fp.parents and fp != WEBAPP or not fp.is_file():
            # fall back to index.html for client-side routes
            fp = WEBAPP / "index.html"
        ctype = _CTYPE.get(fp.suffix, "application/octet-stream")
        self._send(200, fp.read_bytes(), ctype)

    # -- api -----------------------------------------------------------------

    def _api_get(self, path: str):
        if path == "/api/ledger":
            answers = [{"id": e["id"], "question": e["question"],
                        "confidence": e.get("confidence"), "ts": e.get("ts"),
                        "n_cited": (e.get("grounding") or {}).get("n_cited", 0)}
                       for e in _LEDGER.entries if e.get("answer")]
            answers.sort(key=lambda a: a["id"], reverse=True)
            return self._json({"answers": answers,
                               "stats": {**_LEDGER.stats(), **_INDEX.stats()}})
        if path.startswith("/api/answers/"):
            try:
                eid = int(path.rsplit("/", 1)[-1])
            except ValueError:
                return self._json({"error": "bad id"}, 400)
            e = _LEDGER.get(eid)
            if not e or not e.get("answer"):
                return self._json({"error": "no answer"}, 404)
            return self._json(e)
        if path == "/api/queue":
            return self._json({"queue": _read_queue()})
        self._json({"error": "not found"}, 404)

    def _ask(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        q = (body.get("question") or "").strip()
        if not q:
            return self._json({"error": "empty question"}, 400)

        # 0. Exact repeat (case-insensitive) -> serve it. find_similar
        #    deliberately drops identical-string matches (it's built to surface
        #    *different* prior questions), so an exact re-ask is handled here.
        qn = q.strip().lower()
        exact = next((e for e in _LEDGER.entries
                      if e.get("answer") and e["question"].strip().lower() == qn), None)
        if exact:
            return self._json({"status": "answered", "similarity": 1.0, "answer": exact})

        # Prior worked answers (different phrasings), ranked by similarity.
        cands = [(e, s) for e, s in _LEDGER.find_similar(q, threshold=0.50, top=4)
                 if e.get("answer")]

        # 1. Very close paraphrase -> serve the worked answer directly.
        if cands and cands[0][1] >= 0.82:
            entry, sim = cands[0]
            return self._json({"status": "answered", "similarity": round(sim, 2),
                               "answer": entry})

        # 2. Otherwise: live evidence (retrieval, no synthesis) + queue, and
        #    surface any RELATED worked answers as suggestions. We don't
        #    auto-serve a loosely-matched answer to a subtly different clinical
        #    question — we offer it and let the clinician decide.
        passages = _INDEX.search(q, k=8)
        _queue_question(q)
        ev = [{"slug": p.slug, "title": p.title, "doi": p.doi, "tier": p.tier,
               "text": " ".join(p.text.split())[:500]} for p in passages]
        related = [{"id": e["id"], "question": e["question"],
                    "confidence": e.get("confidence"), "similarity": round(s, 2)}
                   for e, s in cands[:3]]
        return self._json({"status": "queued", "evidence": ev, "related": related,
                           "message": "Full answer queued for the next deep run."})


def serve(port: int = 8765, host: str = "127.0.0.1", verbose: bool = True) -> None:
    _boot(verbose=verbose)
    httpd = HTTPServer((host, port), Handler)
    if verbose:
        print(f"\n  LocalEvidence serving on http://{host}:{port}")
        print(f"  (bind is {host} — reach it from your phone over your tailnet/LAN, "
              f"never the open internet)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
