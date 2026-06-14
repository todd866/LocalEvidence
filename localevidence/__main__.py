"""CLI: python3 -m localevidence ask "<clinical question>"

Run from the repo root. The `ask` loop discovers candidate literature, triages
it, acquires full text into the local library (which warms over time), indexes
at passage level, and writes an evidence pack to
projects/<slug>/runs/<run_id>/evidence-pack.md for synthesis.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="localevidence")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Build a grounded evidence pack for a clinical question (discover->acquire->index->pack)")
    ask.add_argument("question", nargs="+", help="The clinical question")
    ask.add_argument("--project", default=None, help="Project slug (default: derived from question)")
    ask.add_argument("--run-id", default=None, help="Run id (default: timestamp)")
    ask.add_argument("-q", "--extra-query", action="append", default=[],
                     help="Extra focused discovery query (repeatable)")
    ask.add_argument("--top-n", type=int, default=20, help="Max NEW papers to acquire")
    ask.add_argument("--max-per-query", type=int, default=100, help="Candidates per discovery query")
    ask.add_argument("-k", "--passages", type=int, default=12, help="Passages to retrieve")
    ask.add_argument("--relevance-floor", type=float, default=0.30, help="Drop candidates below this cosine relevance")
    ask.add_argument("--oa-only", action="store_true", help="Open-access providers only (skip the shadow tier)")
    ask.add_argument("--resume", action="store_true", help="Reuse stage checkpoints if present")
    ask.add_argument("--quiet", action="store_true", help="Less progress output")

    answer = sub.add_parser("answer", help="Persist a synthesised answer into its ledger entry")
    answer.add_argument("--project", default=None, help="Project slug (uses latest run)")
    answer.add_argument("--run", default=None, help="Explicit run directory")
    answer.add_argument("--file", default=None, help="Answer markdown file (default: <run>/answer.md)")
    answer.add_argument("--reasoning", default=None, help="Short synthesis rationale")
    answer.add_argument("--confidence", default=None, choices=["high", "moderate", "low"], help="Answer confidence")

    srv = sub.add_parser("serve", help="Run the local backend + PWA (reach it from your phone over LAN/tailnet)")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--host", default="127.0.0.1", help="Bind address (default localhost; use 0.0.0.0 only behind a private network)")

    gl = sub.add_parser("guidelines", help="Harvest web-published clinical guidelines into the local library")
    gl.add_argument("--source", default="rch", choices=["rch"], help="Guideline source to harvest")
    gl.add_argument("--limit", type=int, default=0, help="Harvest at most N guidelines")
    gl.add_argument("--pace", type=float, default=0.7, help="Seconds between fetches")
    gl.add_argument("--refresh", action="store_true", help="Re-fetch guidelines already held")
    gl.add_argument("--no-index", action="store_true", help="Skip indexing into the passage store")
    gl.add_argument("--quiet", action="store_true")

    load = sub.add_parser("load", help="Self-play a question bank to warm the system")
    load.add_argument("--bank", default=None, help="Path to question bank YAML (default: data/question_bank.yaml)")
    load.add_argument("--topic", default=None, help="Only run questions with this topic")
    load.add_argument("--limit", type=int, default=0, help="Run at most N questions")
    load.add_argument("--top-n", type=int, default=12, help="Max NEW papers per question")
    load.add_argument("--max-per-query", type=int, default=70, help="Candidates per discovery query")
    load.add_argument("-k", "--passages", type=int, default=12, help="Passages retrieved per question")
    load.add_argument("--oa-only", action="store_true", help="Legal OA sources only")
    load.add_argument("--force", action="store_true", help="Re-run questions already in the ledger")
    load.add_argument("--pace", type=float, default=2.0, help="Seconds between questions")
    load.add_argument("--quiet", action="store_true", help="Less progress output")

    qu = sub.add_parser("queue", help="Process the phone PWA's queued questions (ledger/queue.jsonl)")
    qu.add_argument("action", choices=["list", "run", "clear"], help="list / run / clear the queue")
    qu.add_argument("--top-n", type=int, default=15, help="Max NEW papers per question (run)")
    qu.add_argument("-k", "--passages", type=int, default=12, help="Passages retrieved per question (run)")
    qu.add_argument("--oa-only", action="store_true", help="Open-access providers only (run)")
    qu.add_argument("--keep", action="store_true", help="Don't clear the queue after run")
    qu.add_argument("--quiet", action="store_true", help="Less progress output")

    au = sub.add_parser("audit", help="Emit the provenance + grounding audit trail for a worked answer")
    au.add_argument("-e", "--entry", type=int, default=None, help="Ledger entry id (default: most recent answered)")
    au.add_argument("--project", default=None, help="Audit the latest answered run of a project")
    au.add_argument("--json", action="store_true", help="Machine-readable JSON")
    au.add_argument("--resolve", action="store_true", help="Live-check cited DOIs against doi.org (network)")

    il = sub.add_parser("index-library", help="Index an existing library's full-text papers into the passage index (use a corpus you already hold)")
    il.add_argument("--match", default=None, help="Only papers whose title matches this regex (topic subset)")
    il.add_argument("--source", default=None, help="Only papers with this catalog source")
    il.add_argument("--limit", type=int, default=0, help="Index at most N papers")
    il.add_argument("--batch", type=int, default=150, help="Papers per persisted chunk (resumable on a big run)")
    il.add_argument("--quiet", action="store_true")

    pk = sub.add_parser("pack", help="Distributable knowledge pack: shareable list + summaries + map (no corpus)")
    pk.add_argument("action", choices=["export", "harvest"], help="export a pack / harvest (rebuild) from one")
    pk.add_argument("dir", help="Pack directory (write for export, read for harvest)")
    pk.add_argument("--match", default=None, help="Export only papers whose title matches this regex (topic subset)")
    pk.add_argument("--clusters", type=int, default=None, help="Number of topic clusters (export; default ~sqrt(n))")
    pk.add_argument("--neighbours", type=int, default=5, help="Nearest-neighbour links per paper (export)")
    pk.add_argument("--oa-only", action="store_true", help="Open-access providers only (harvest)")
    pk.add_argument("--top", type=int, default=0, help="Harvest only the first N papers (harvest)")
    pk.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "ask":
        from .pipeline import ask as run_ask
        res = run_ask(
            " ".join(args.question),
            project=args.project,
            run_id=args.run_id,
            extra_queries=args.extra_query or None,
            top_n=args.top_n,
            max_per_query=args.max_per_query,
            k_passages=args.passages,
            oa_only=args.oa_only,
            relevance_floor=args.relevance_floor,
            resume=args.resume,
            verbose=not args.quiet,
        )
        print(f"\nEvidence pack written to:\n{res.evidence_pack_path}")
        return 0

    if args.command == "answer":
        from .pipeline import record_answer
        ans = None
        if args.file:
            from pathlib import Path
            ans = Path(args.file).read_text()
        record_answer(project=args.project, run_dir=args.run, answer=ans,
                      reasoning=args.reasoning, confidence=args.confidence)
        return 0

    if args.command == "serve":
        from .server import serve
        serve(port=args.port, host=args.host)
        return 0

    if args.command == "guidelines":
        from .guidelines import harvest_rch, index_guidelines
        if args.source == "rch":
            harvest_rch(limit=args.limit, pace_s=args.pace, refresh=args.refresh,
                        verbose=not args.quiet)
        if not args.no_index:
            index_guidelines(verbose=not args.quiet)
        return 0

    if args.command == "load":
        from .loader import load_bank, run_bank
        bank = load_bank(args.bank)
        run_bank(
            bank,
            topic=args.topic,
            limit=args.limit,
            top_n=args.top_n,
            max_per_query=args.max_per_query,
            k_passages=args.passages,
            oa_only=args.oa_only,
            force=args.force,
            pace_s=args.pace,
            verbose=not args.quiet,
        )
        return 0

    if args.command == "queue":
        from .queue import list_queue, run_queue, clear_queue
        if args.action == "list":
            list_queue()
        elif args.action == "clear":
            print(f"cleared {clear_queue()} queued question(s)")
        else:
            run_queue(top_n=args.top_n, k_passages=args.passages,
                      oa_only=args.oa_only, clear=not args.keep,
                      verbose=not args.quiet)
        return 0

    if args.command == "audit":
        from .audit import audit_cli
        return audit_cli(entry_id=args.entry, project=args.project,
                         as_json=args.json, resolve=args.resolve)

    if args.command == "index-library":
        from .ingest import index_library
        index_library(match=args.match, source=args.source, limit=args.limit,
                      batch=args.batch, verbose=not args.quiet)
        return 0

    if args.command == "pack":
        from .pack import export_pack, harvest_pack
        if args.action == "export":
            export_pack(args.dir, match=args.match, k_clusters=args.clusters,
                        neighbours=args.neighbours, verbose=not args.quiet)
        else:
            harvest_pack(args.dir, oa_only=args.oa_only, top=args.top,
                         verbose=not args.quiet)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
