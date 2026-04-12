"""
Live monitor for batch_v2.py runs (multi-signal architecture).
Usage: python3 monitor.py [output_file] [--file artist_list.txt] [--budget N]

Ground truth comes from:
  1. data/batch_progress.json   — live stats from batch_v2.py
  2. data/artists/*/state.json  — per-step completion
  3. data/artists/*/*.json      — signal data files
"""

import sys
import re
import time
import glob
import json
import os
import datetime
import argparse
from pathlib import Path
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich import box

TASK_DIR   = "/private/tmp/claude-501"
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data/artists"
PROGRESS_FILE = BASE_DIR / "data/batch_progress.json"

# ── Model pricing ─────────────────────────────────────────────────────────────

MODELS = {
    "claude-haiku-4-5-20251001": {
        "name": "Haiku 4.5",
        "input_per_m": 0.80,
        "output_per_m": 4.00,
    },
    "claude-sonnet-4-20250514": {
        "name": "Sonnet 4",
        "input_per_m": 3.00,
        "output_per_m": 15.00,
    },
}

MODEL_KEY          = "claude-haiku-4-5-20251001"
MODEL_INFO         = MODELS[MODEL_KEY]
MODEL_NAME         = MODEL_INFO["name"]
PRICE_INPUT_PER_M  = MODEL_INFO["input_per_m"]
PRICE_OUTPUT_PER_M = MODEL_INFO["output_per_m"]
AVG_INPUT_TOKENS   = 2500
AVG_OUTPUT_TOKENS  = 800
BUDGET             = 73.06

STATUS_PENDING   = "pending"
STATUS_ACTIVE    = "active"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED    = "failed"
STATUS_SKIPPED   = "skipped"

STATUS_ICON = {
    STATUS_PENDING:   ("·", "dim"),
    STATUS_ACTIVE:    ("▶", "bold yellow"),
    STATUS_SUCCEEDED: ("✓", "bold green"),
    STATUS_FAILED:    ("✗", "bold red"),
    STATUS_SKIPPED:   ("~", "dim cyan"),
}

PHASE_ICONS = {
    "interview_search":   "🔍 Search Interviews",
    "interview_scrape":   "📥 Scrape Interviews",
    "interview_extract":  "✂️  Extract Quotes",
    "review_search":      "🔍 Search Reviews",
    "review_scrape":      "📥 Scrape Reviews",
    "review_extract":     "✂️  Extract Critic",
    "influence_extract":  "🔗 Extract Influences",
    "tag_extract":        "🏷️  Extract Tags",
    "search_interviews":  "🔍 Search Interviews",
    "scrape_interviews":  "📥 Scrape Interviews",
    "extract_quotes":     "✂️  Extract Quotes",
    "search_reviews":     "🔍 Search Reviews",
    "scrape_reviews":     "📥 Scrape Reviews",
    "extract_critic":     "✂️  Extract Critic",
    "global_pipeline":    "⚙️  Global Pipeline",
}

# ── Project plan phases ──────────────────────────────────────────────────────

def detect_plan_phases():
    """Dynamically detect plan phase statuses from actual project state."""
    import subprocess

    # Phase 4: QA — check for qa_report.json with overall_pass
    qa_report = BASE_DIR / "data" / "qa_report.json"
    qa_done = False
    if qa_report.exists():
        try:
            qr = json.loads(qa_report.read_text())
            if qr.get("overall_pass"):
                qa_done = True
        except Exception:
            pass

    # Phase 1: Artist list — need 10K+ artists to consider it done
    artist_list = BASE_DIR / "data" / "artist_list_20k.txt"
    phase1_done = False
    if artist_list.exists():
        try:
            line_count = sum(1 for line in artist_list.read_text().splitlines() if line.strip())
            phase1_done = line_count >= 10000
        except Exception:
            pass

    # Check if generate_artist_list.py is actively running
    phase1_running = False
    try:
        pgrep = subprocess.run(["pgrep", "-f", "generate_artist_list"], capture_output=True, text=True, timeout=3)
        if pgrep.stdout.strip():
            phase1_running = True
    except Exception:
        pass

    phase1_active = phase1_running or (qa_done and not phase1_done)

    # Phase 5: Full ingestion — only counts if batch has >1000 artists (not QA runs)
    bp = BASE_DIR / "data" / "batch_progress.json"
    phase5_active = False
    phase5_done = False
    if bp.exists():
        try:
            bpd = json.loads(bp.read_text())
            total = bpd.get("total_artists", 0)
            completed = bpd.get("completed", 0)
            if total >= 1000:  # Only large runs count as Phase 5
                phase5_active = not bpd.get("finished")
                phase5_done = bool(bpd.get("finished")) and completed == total
        except Exception:
            pass

    # Phase 6: Global pipeline — check for embeddings/similarity matrix
    embeddings = BASE_DIR / "data" / "embeddings.npz"
    similarity = BASE_DIR / "data" / "similarity_matrix.npz"
    phase6_done = embeddings.exists() and similarity.exists()

    # Phase 7: Git push
    phase7_done = False

    # Build status list
    result = []
    phases = [
        ("Phase 1", "Artist List Generation"),
        ("Phase 2", "Code Changes"),
        ("Phase 3", "Multi-Signal Architecture"),
        ("Phase 4", "QA Gate"),
        ("Phase 5", "Full Ingestion (1K)"),
        ("Phase 6", "Global Pipeline + App Build"),
        ("Phase 7", "GitHub Push + Mount"),
    ]

    for label, name in phases:
        if label == "Phase 1":
            status = "done" if phase1_done else ("active" if phase1_active else "pending")
        elif label == "Phase 2":
            status = "done"
        elif label == "Phase 3":
            status = "done"
        elif label == "Phase 4":
            status = "done" if qa_done else "pending"
        elif label == "Phase 5":
            if phase5_done:
                status = "done"
            elif phase5_active:
                status = "active"
            elif phase1_done and qa_done and not phase5_done:
                status = "pending"
            else:
                status = "pending"
        elif label == "Phase 6":
            status = "done" if phase6_done else ("active" if phase5_done else "pending")
        elif label == "Phase 7":
            status = "done" if phase7_done else ("active" if phase6_done else "pending")
        else:
            status = "pending"
        result.append((label, name, status))

    return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_output():
    pattern = os.path.join(TASK_DIR, "**", "tasks", "*.output")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_artist_list(file_path=None):
    if file_path:
        p = Path(file_path)
        if p.exists():
            return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    for candidate in ["data/artist_list_20k.txt", "data/seed_artists.txt"]:
        p = BASE_DIR / candidate
        if p.exists():
            return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return []


def normalize(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def get_batch_health(output_file):
    now = time.time()
    checked_at = datetime.datetime.now().strftime("%-I:%M:%S %p")

    running = False
    process_count = 0
    try:
        import subprocess
        for script in ["batch_v2.py", "batch_full.py", "qa_mini_batch.py", "generate_artist_list.py"]:
            result = subprocess.run(
                ["pgrep", "-f", script],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                running = True
                process_count += len(result.stdout.strip().splitlines())
    except Exception:
        pass

    # Also check freshness of mb_artist_cache.json (Phase 1 indicator)
    mb_cache = BASE_DIR / "data" / "mb_artist_cache.json"
    if not running:
        for check_file in [PROGRESS_FILE, mb_cache]:
            if check_file.exists():
                try:
                    fmtime = os.path.getmtime(check_file)
                    if (now - fmtime) < 60:
                        running = True
                        process_count = max(process_count, 1)
                        break
                except Exception:
                    pass

    stale_mins = 0.0
    last_updated = "—"

    # Use the most recently updated file for staleness
    best_mtime = 0
    for check_file in [PROGRESS_FILE, mb_cache]:
        if check_file.exists():
            try:
                mtime = os.path.getmtime(check_file)
                if mtime > best_mtime:
                    best_mtime = mtime
            except Exception:
                pass
    if output_file:
        try:
            mtime = os.path.getmtime(output_file)
            if mtime > best_mtime:
                best_mtime = mtime
        except Exception:
            pass

    if best_mtime > 0:
        stale_mins = (now - best_mtime) / 60.0
        last_updated = datetime.datetime.fromtimestamp(best_mtime).strftime("%-I:%M:%S %p")

    if running and stale_mins < 5:
        status = "running"
    elif running and stale_mins >= 5:
        status = "stale"
    else:
        status = "dead"

    return dict(
        running=running,
        process_count=process_count,
        stale_mins=stale_mins,
        status=status,
        last_updated=last_updated,
        checked_at=checked_at,
    )


# ── Ground-truth reads ──────────────────────────────────────────────────────

def load_data_stats():
    """Read data/artists/* for cumulative ground-truth across ALL runs."""
    result = dict(
        status={},
        total_api_calls=0,
        total_quotes=0,
        total_passages=0,
        total_influences=0,
        total_tags=0,
        artists_valid=set(),
        artists_attempted=set(),
    )

    if not DATA_DIR.exists():
        return result

    for artist_dir in DATA_DIR.iterdir():
        if not artist_dir.is_dir():
            continue
        aid = artist_dir.name
        has_valid = False
        has_attempted = False

        state_path = artist_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                st = state.get("status", "pending")
                if st == "extract_complete":
                    has_valid = True
                elif st in ("in_progress", "failed"):
                    has_attempted = True
            except Exception:
                pass

        for fname in ("sources.json", "review_sources.json"):
            p = artist_dir / fname
            if p.exists():
                try:
                    sources = json.loads(p.read_text())
                    result["total_api_calls"] += sum(1 for s in sources if s.get("text"))
                except Exception:
                    pass

        q_path = artist_dir / "quotes.json"
        if q_path.exists():
            try:
                data = json.loads(q_path.read_text())
                result["total_quotes"] += len(data.get("quotes", []))
                has_attempted = True
                if data.get("corpus_meta", {}).get("corpus_valid"):
                    has_valid = True
            except Exception:
                pass

        c_path = artist_dir / "critic_quotes.json"
        if c_path.exists():
            try:
                data = json.loads(c_path.read_text())
                result["total_passages"] += len(data.get("quotes", []))
                has_attempted = True
                if data.get("corpus_meta", {}).get("corpus_valid"):
                    has_valid = True
            except Exception:
                pass

        inf_path = artist_dir / "influences.json"
        if inf_path.exists():
            try:
                data = json.loads(inf_path.read_text())
                result["total_influences"] += data.get("meta", {}).get("count", 0)
            except Exception:
                pass

        tag_path = artist_dir / "tags.json"
        if tag_path.exists():
            try:
                data = json.loads(tag_path.read_text())
                result["total_tags"] += data.get("meta", {}).get("count", 0)
            except Exception:
                pass

        if has_valid:
            result["status"][aid] = STATUS_SUCCEEDED
            result["artists_valid"].add(aid)
        elif has_attempted:
            result["status"][aid] = STATUS_FAILED
            result["artists_attempted"].add(aid)

    return result


def load_batch_progress():
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {}


# ── Output file parse ────────────────────────────────────────────────────────

def parse_live(lines, batch_progress):
    live = dict(
        current_artist="",
        current_artist_id="",
        current_num=0,
        total=0,
        current_phase="",
        current_source="",
        global_step="",
        failed_this_run=set(),
        recent=[],
        done=False,
        artists_done_cur=0,
        billed_cur=0,
    )

    if batch_progress:
        live["current_artist"] = batch_progress.get("in_progress", "")
        live["current_artist_id"] = normalize(live["current_artist"]) if live["current_artist"] else ""
        live["current_phase"] = batch_progress.get("current_step", "")
        live["total"] = batch_progress.get("total_artists", 0)
        _done = batch_progress.get("completed", 0) + batch_progress.get("failed", 0) + batch_progress.get("skipped_complete", 0)
        live["current_num"] = _done + (1 if live["current_artist"] else 0)
        live["artists_done_cur"] = _done
        if batch_progress.get("finished"):
            live["done"] = True

    after_sep = False
    for line in lines:
        s = line.strip()

        if re.match(r'^={50,}$', s):
            after_sep = True
            continue
        if re.match(r'^─{50,}$', s):
            after_sep = False
            continue

        if after_sep:
            m = re.match(r'^\[(\d+)/(\d+)\]\s+(.+)$', s)
            if m:
                if not batch_progress:
                    live["current_num"] = int(m.group(1))
                    live["total"] = int(m.group(2))
                    live["current_artist"] = m.group(3).split(" (")[0]
                    live["current_artist_id"] = normalize(live["current_artist"])
                live["current_phase"] = ""
                live["current_source"] = ""
            after_sep = False
            continue

        after_sep = False

        # Phase detection
        for pattern, phase in [
            ("[interview_search]", "interview_search"),
            ("[interview_scrape]", "interview_scrape"),
            ("[interview_extract]", "interview_extract"),
            ("[review_search]", "review_search"),
            ("[review_scrape]", "review_scrape"),
            ("[review_extract]", "review_extract"),
            ("[influence_extract]", "influence_extract"),
            ("[tag_extract]", "tag_extract"),
            ("PHASE 1: INTERVIEW", "interview_search"),
            ("PHASE 2: CRITIC", "review_search"),
            ("[SEARCH] Finding interview", "interview_search"),
            ("[SCRAPE INTERVIEWS]", "interview_scrape"),
            ("[EXTRACT QUOTES]", "interview_extract"),
            ("[SEARCH REVIEWS]", "review_search"),
            ("[SCRAPE REVIEWS]", "review_scrape"),
            ("[EXTRACT CRITIC", "review_extract"),
            ("GLOBAL DUAL-SIGNAL", "global_pipeline"),
        ]:
            if pattern in s:
                live["current_phase"] = phase
                break

        m = re.match(r'^\[(\d+)/(\d+)\]\s+(.+)$', s)
        if m:
            live["current_source"] = m.group(3)

        m = re.match(r'^Running (\S+\.py)\.\.\.', s)
        if m:
            live["global_step"] = m.group(1)

        if "BATCH COMPLETE" in s or "Log saved to:" in s:
            live["done"] = True

        if re.match(r'^\s+OK:', s):
            live["billed_cur"] += 1

        if s and not re.match(r'^[=─]{10,}$', s):
            live["recent"].append(s)

    live["recent"] = live["recent"][-22:]
    return live


def calc_projection(data_stats, live, current_file, batch_progress):
    proj = dict(
        cost_so_far=0.0,
        projected_total_cost=0.0,
        budget_pct=0.0,
        budget_status="green",
        calls_per_artist=0.0,
        mins_per_artist=0.0,
        mins_remaining=0.0,
        eta_str="—",
    )

    cost_per_call = (
        AVG_INPUT_TOKENS  / 1_000_000 * PRICE_INPUT_PER_M +
        AVG_OUTPUT_TOKENS / 1_000_000 * PRICE_OUTPUT_PER_M
    )

    # CUMULATIVE cost from actual data (all runs, all signals)
    proj["cost_so_far"] = data_stats["total_api_calls"] * cost_per_call

    # Timing
    if batch_progress and batch_progress.get("artist_times"):
        times = batch_progress["artist_times"]
        if times:
            proj["mins_per_artist"] = (sum(times) / len(times)) / 60.0

    if proj["mins_per_artist"] == 0:
        elapsed = 0.0
        if current_file:
            try:
                stat = os.stat(current_file)
                created = getattr(stat, "st_birthtime", stat.st_ctime)
                elapsed = (time.time() - created) / 60.0
            except Exception:
                pass
        done_cur = live["artists_done_cur"]
        if done_cur > 0 and elapsed > 0:
            proj["mins_per_artist"] = elapsed / done_cur
        else:
            proj["mins_per_artist"] = 2.0

    if live["artists_done_cur"] > 0:
        proj["calls_per_artist"] = live["billed_cur"] / live["artists_done_cur"]
    else:
        proj["calls_per_artist"] = 30.0

    total = live["total"] or 128
    cpa = proj["calls_per_artist"]
    proj["projected_total_calls"] = int(total * cpa)
    proj["projected_total_cost"] = proj["projected_total_calls"] * cost_per_call

    pct = proj["projected_total_cost"] / BUDGET * 100 if BUDGET > 0 else 0
    proj["budget_pct"] = pct
    proj["budget_status"] = "green" if pct < 55 else "yellow" if pct < 80 else "red"

    artists_done_total = len(data_stats["artists_valid"])
    remaining = max(0, total - artists_done_total)
    worker_count = max(1, batch_progress.get("worker_count", 1) if batch_progress else 1)
    proj["mins_remaining"] = (remaining * proj["mins_per_artist"]) / worker_count
    eta_dt = datetime.datetime.now() + datetime.timedelta(minutes=proj["mins_remaining"])
    proj["eta_str"] = eta_dt.strftime("%-I:%M %p")

    return proj


# ── Panels ────────────────────────────────────────────────────────────────────

def make_plan_panel(batch_progress):
    """Show the 7-phase project plan with dynamically detected status."""
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("phase", style="dim", width=9)
    t.add_column("name", width=28)
    t.add_column("status", justify="right", width=12)

    phases = detect_plan_phases()

    # Compute phase-specific progress percentages
    phase_pcts = {}

    # Phase 1: artist list progress from MB cache
    mb_cache = BASE_DIR / "data" / "mb_artist_cache.json"
    if mb_cache.exists():
        try:
            mc = json.loads(mb_cache.read_text())
            phase_pcts["Phase 1"] = min(100, int(mc.get("total", 0) / 200))  # 20K target
        except Exception:
            pass
    artist_list = BASE_DIR / "data" / "artist_list_20k.txt"
    if artist_list.exists() and artist_list.stat().st_size > 1000:
        phase_pcts["Phase 1"] = 100

    def _bp_done(bp):
        return bp.get("completed", 0) + bp.get("failed", 0) + bp.get("skipped_complete", 0)

    # Phase 4: QA progress
    if batch_progress and batch_progress.get("total_artists", 0) <= 100:
        done = _bp_done(batch_progress)
        total = batch_progress.get("total_artists", 0)
        if total > 0:
            phase_pcts["Phase 4"] = int(done / total * 100)

    # Phase 5: ingestion progress
    if batch_progress and batch_progress.get("total_artists", 0) > 100:
        done = _bp_done(batch_progress)
        total = batch_progress.get("total_artists", 0)
        if total > 0:
            phase_pcts["Phase 5"] = int(done / total * 100)

    for label, name, status in phases:
        pct = phase_pcts.get(label, 0)
        if status == "done":
            t.add_row(f"[green]{label}[/]", f"[green]{name}[/]", "[bold green]✓ DONE[/]")
        elif status == "active":
            pct_str = f"{pct}%" if pct > 0 else "…"
            t.add_row(f"[bold yellow]{label}[/]", f"[bold yellow]{name}[/]", f"[bold yellow]▶ {pct_str}[/]")
        else:
            t.add_row(f"[dim]{label}[/]", f"[dim]{name}[/]", "[dim]· pending[/]")

    # Next up
    active_idx = next((i for i, (_, _, s) in enumerate(phases) if s == "active"), -1)
    if active_idx >= 0 and active_idx + 1 < len(phases):
        next_label, next_name, _ = phases[active_idx + 1]
        t.add_row("", "", "")
        t.add_row("[dim]Next →[/]", f"[dim italic]{next_name}[/]", "")

    return Panel(t, title="[bold]Project Plan[/]", border_style="magenta")


def make_credits_panel(data_stats, proj, live):
    billed     = data_stats["total_api_calls"]
    quotes     = data_stats["total_quotes"]
    passages   = data_stats["total_passages"]
    influences = data_stats["total_influences"]
    tags       = data_stats["total_tags"]

    cost_so_far  = proj["cost_so_far"]
    proj_total   = proj["projected_total_cost"]
    budget_left  = max(0.0, BUDGET - cost_so_far)
    bstatus      = proj["budget_status"]
    cpa          = proj["calls_per_artist"]
    mpa          = proj["mins_per_artist"]
    mins_rem     = proj["mins_remaining"]
    eta_str      = proj["eta_str"]

    bc = {"green": "green", "yellow": "yellow", "red": "red"}[bstatus]

    spent_pct  = min(100.0, cost_so_far / BUDGET * 100) if BUDGET > 0 else 0
    proj_pct   = min(100.0, proj_total / BUDGET * 100) if BUDGET > 0 else 0
    bar_w      = 18
    spent_fill = max(1, int(bar_w * spent_pct / 100)) if cost_so_far > 0 else 0
    proj_fill  = max(spent_fill, int(bar_w * proj_pct / 100))
    bar_spent  = "█" * spent_fill
    bar_proj   = "▒" * max(0, proj_fill - spent_fill)
    bar_empty  = "░" * max(0, bar_w - proj_fill)
    bar = (f"[bold yellow]{bar_spent}[/]"
           f"[{bc} dim]{bar_proj}[/]"
           f"[dim]{bar_empty}[/]")

    if bstatus == "green":
        verdict = f"[bold green]● OK[/]  ~${proj_total:.0f} of ${BUDGET:.0f}"
    elif bstatus == "yellow":
        verdict = f"[bold yellow]● TIGHT[/]  ~${proj_total:.0f}/{BUDGET:.0f}"
    else:
        verdict = f"[bold red]● RISK[/]  ~${proj_total:.0f}/{BUDGET:.0f}"

    time_rem = (f"{mins_rem/60:.1f} hrs" if mins_rem > 90
                else f"{mins_rem:.0f} min" if mins_rem > 0 else "—")

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("k", style="dim", width=17)
    t.add_column("v", justify="right")

    t.add_row("[bold]CUMULATIVE SPEND[/]", f"[bold yellow]${cost_so_far:.2f}[/]  [dim]/ ${BUDGET:.0f}[/]")
    t.add_row("",                     Text.from_markup(f"{bar}  [bold yellow]{spent_pct:.0f}%[/]"))
    t.add_row("[dim]Budget left[/]",  f"[{'red' if budget_left < 5 else 'bold green'}]${budget_left:.2f}[/]")
    t.add_row("",                     "")

    t.add_row("[bold]PROJECTED[/]",   Text.from_markup(verdict))
    t.add_row("[dim]Model[/]",        f"[dim]{MODEL_NAME}[/]")
    t.add_row("",                     "")

    t.add_row("[bold]ETA[/]",         f"[bold cyan]{eta_str}[/]")
    t.add_row("Time left",            f"[bold]{time_rem}[/]")
    t.add_row("Min/artist",           f"[dim]{mpa:.1f}  ({cpa:.1f} calls)[/]")
    t.add_row("",                     "")

    t.add_row("[bold]SIGNALS[/]",     f"[bold]{billed:,} API calls[/]")
    t.add_row("Quotes",               f"[green]{quotes:,}[/]")
    t.add_row("Passages",             f"[cyan]{passages:,}[/]")
    t.add_row("Influences",           f"[magenta]{influences:,}[/]")
    t.add_row("Tags",                 f"[yellow]{tags:,}[/]")

    return Panel(t, title=f"[bold]API Credits  /  ${BUDGET:.0f} Budget[/]",
                 border_style=bc)


def make_dashboard(data_stats, live, proj, health, all_artists, output_file, file_size, batch_progress):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=6),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left",    ratio=3),
        Layout(name="middle",  ratio=4),
        Layout(name="credits", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="plan",  size=14),
        Layout(name="stack"),
    )
    layout["middle"].split_column(
        Layout(name="current", size=10),
        Layout(name="log"),
    )

    # ── Artist statuses ──────────────────────────────────────────────────
    artist_status = {}
    artist_phases = {}

    if len(all_artists) <= 500:
        for artist in all_artists:
            aid = normalize(artist)
            if aid in data_stats["status"]:
                artist_status[aid] = data_stats["status"][aid]
            else:
                artist_status[aid] = STATUS_PENDING
            artist_phases[aid] = ""
    else:
        for aid, st in data_stats["status"].items():
            artist_status[aid] = st

    cur_aid = live["current_artist_id"]
    if cur_aid:
        artist_status[cur_aid] = STATUS_ACTIVE
        artist_phases[cur_aid] = live["current_phase"]

    for aid in live.get("failed_this_run", set()):
        if artist_status.get(aid) not in (STATUS_SUCCEEDED, STATUS_ACTIVE):
            artist_status[aid] = STATUS_FAILED

    # ── Header: phase + overall progress bars ──────────────────────────
    phases = detect_plan_phases()
    n_phases = len(phases)
    done_phases = sum(1 for _, _, s in phases if s == "done")
    active_phase = next(((l, n) for l, n, s in phases if s == "active"), None)
    overall_pct = int(done_phases / n_phases * 100)

    # Phase-level progress
    phase_pct = 0
    phase_label = ""
    if active_phase:
        phase_label = f"{active_phase[0]}: {active_phase[1]}"
        # Compute phase-specific %
        if "Artist List" in active_phase[1]:
            mb_cache = BASE_DIR / "data" / "mb_artist_cache.json"
            if mb_cache.exists():
                try:
                    mc = json.loads(mb_cache.read_text())
                    phase_pct = min(99, int(mc.get("total", 0) / 200))  # 20K target
                except Exception:
                    pass
            al = BASE_DIR / "data" / "artist_list_20k.txt"
            if al.exists():
                try:
                    lc = sum(1 for l in al.read_text().splitlines() if l.strip())
                    phase_pct = min(99, int(lc / 200))
                except Exception:
                    pass
        elif "Ingestion" in active_phase[1]:
            bp_total_a = batch_progress.get("total_artists", 0)
            bp_done_a = batch_progress.get("completed", 0) + batch_progress.get("skipped_complete", 0) + batch_progress.get("failed", 0)
            if bp_total_a > 0:
                phase_pct = int(bp_done_a / bp_total_a * 100)
        elif "QA" in active_phase[1]:
            bp_total_a = batch_progress.get("total_artists", 0)
            bp_done_a = batch_progress.get("completed", 0) + batch_progress.get("skipped_complete", 0) + batch_progress.get("failed", 0)
            if bp_total_a > 0:
                phase_pct = int(bp_done_a / bp_total_a * 100)
    else:
        phase_label = "No active phase"

    n_succeeded = len(data_stats["artists_valid"])
    n_attempted = len(data_stats["artists_attempted"])
    n_active    = 1 if cur_aid else 0

    bar_w = 40

    # Phase bar
    p_filled = int(bar_w * phase_pct / 100)
    phase_bar = "█" * p_filled + "░" * (bar_w - p_filled)

    # Overall bar
    o_filled = int(bar_w * overall_pct / 100)
    overall_bar = "█" * o_filled + "░" * (bar_w - o_filled)

    procs = health.get("process_count", 0)
    hstatus = health.get("status", "dead")
    if hstatus == "running":
        run_status = "[bold green]RUNNING[/]"
    elif hstatus == "stale":
        run_status = "[bold yellow]STALE[/]"
    else:
        run_status = "[dim]IDLE[/]"
    proc_note = f"  [dim]({procs} proc)[/]" if procs > 0 else ""

    step_label = PHASE_ICONS.get(live["current_phase"], live["current_phase"]) or ""
    artist_note = f"  [dim]→ {live['current_artist']} {step_label}[/]" if live["current_artist"] else ""

    header = Text.from_markup(
        f"  [bold cyan]Basilect Engine[/]  {run_status}{proc_note}\n"
        f"  [bold]{phase_label}[/]  [yellow]{phase_bar}[/]  [bold]{phase_pct}%[/]\n"
        f"  [dim]Overall[/]  [green]{overall_bar}[/]  [bold]{overall_pct}%[/]  "
        f"[dim]({done_phases}/{n_phases} phases)[/]"
        f"{artist_note}"
    )
    layout["header"].update(Panel(header))

    # ── Plan panel ────────────────────────────────────────────────────────
    layout["plan"].update(make_plan_panel(batch_progress))

    # ── Artist Stack ──────────────────────────────────────────────────────
    stack_text = Text()

    if len(all_artists) > 200:
        total = batch_progress.get("total_artists", 0) or live["total"] or len(all_artists) or 128
        n_pending = max(0, total - n_succeeded - n_attempted - n_active)
        stack_text.append(f"  {total:,} total artists\n\n", style="bold")
        stack_text.append(f"  ✓ {n_succeeded:,} valid\n", style="green")
        stack_text.append(f"  ✗ {n_attempted:,} partial\n", style="red")
        stack_text.append(f"  ▶ {n_active} active\n", style="yellow")
        stack_text.append(f"  · {n_pending:,} pending\n", style="dim")

        stack_text.append(f"\n  Recent:\n", style="bold")
        try:
            recent_artists = sorted(
                DATA_DIR.iterdir(),
                key=lambda d: d.stat().st_mtime if d.is_dir() else 0,
                reverse=True
            )[:12]
            for d in recent_artists:
                if d.is_dir():
                    aid = d.name
                    st = data_stats["status"].get(aid, STATUS_PENDING)
                    icon, style = STATUS_ICON.get(st, STATUS_ICON[STATUS_PENDING])
                    stack_text.append(f"  {icon} {aid}\n", style=style)
        except Exception:
            pass
    else:
        for artist in all_artists:
            aid = normalize(artist)
            s = artist_status.get(aid, STATUS_PENDING)
            icon, style = STATUS_ICON[s]
            phase_label = PHASE_ICONS.get(artist_phases.get(aid, ""), "")

            if s == STATUS_ACTIVE:
                stack_text.append(f" {icon} ", style=style)
                stack_text.append(artist, style="bold yellow")
                if phase_label:
                    stack_text.append(f"  {phase_label}", style="dim yellow")
                stack_text.append("\n")
            elif s == STATUS_SUCCEEDED:
                stack_text.append(f" {icon} {artist}\n", style="green")
            elif s == STATUS_FAILED:
                stack_text.append(f" {icon} {artist}\n", style="red")
            elif s == STATUS_SKIPPED:
                stack_text.append(f" {icon} {artist}\n", style="dim cyan")
            else:
                stack_text.append(f" {icon} {artist}\n", style="dim")

    layout["stack"].update(Panel(stack_text, title="[bold]Artists[/]", border_style="cyan"))

    # ── Now Processing (all workers) ──────────────────────────────────────
    ct = Table(box=box.SIMPLE, show_header=False, padding=(0, 0))
    ct.add_column("w",      style="dim",       width=4)
    ct.add_column("artist", style="bold cyan", width=22)
    ct.add_column("step",   style="yellow",    width=22)
    ct.add_column("prog",   style="dim",       width=8)

    # Read all per-worker shards for live status
    import glob as _glob
    worker_shards = sorted(_glob.glob(str(BASE_DIR / "data" / "batch_progress_*.json")))
    if worker_shards:
        for idx, shard_path in enumerate(worker_shards, 1):
            try:
                wd = json.loads(Path(shard_path).read_text())
                w_artist = wd.get("in_progress") or "—"
                w_step   = wd.get("current_step") or "—"
                w_done   = wd.get("completed", 0) + wd.get("skipped_complete", 0) + wd.get("failed", 0)
                w_total  = wd.get("total_artists", 0)
                # Get step progress from state.json
                step_prog = ""
                if w_artist and w_artist != "—":
                    w_id = re.sub(r'[^\w\s\-]', '', w_artist.lower()).strip()
                    w_id = re.sub(r'\s+', '_', w_id).strip('_')
                    state_f = DATA_DIR / w_id / "state.json"
                    if state_f.exists():
                        try:
                            st = json.loads(state_f.read_text())
                            steps = st.get("steps", {})
                            n_done = sum(1 for s in steps.values() if s.get("status") in ("done", "skipped"))
                            step_prog = f"{n_done}/8"
                        except Exception:
                            pass
                step_icon = PHASE_ICONS.get(w_step, w_step)
                ct.add_row(
                    f"W{idx}",
                    w_artist[:22],
                    step_icon[:22],
                    f"[dim]{w_done}/{w_total}[/]" if w_total else step_prog,
                )
            except Exception:
                ct.add_row(f"W{idx}", "[dim]—[/]", "[dim]—[/]", "")
    else:
        # Fallback: single worker view
        cur       = live["current_artist"] or "—"
        cur_phase = PHASE_ICONS.get(live["current_phase"], live["current_phase"] or "—")
        ct.add_row("W1", cur[:22], cur_phase[:22], f"[dim]{live['current_num']}/{live['total']}[/]")

    if live["global_step"]:
        ct.add_row("", "[dim]pipeline[/]", f"[magenta]{live['global_step']}[/]", "")

    layout["current"].update(Panel(ct, title="[bold]Now Processing — 4 Workers[/]", border_style="yellow"))

    # ── Live Log ──────────────────────────────────────────────────────────
    log_text = Text()
    for line in live["recent"]:
        if "quote(s) extracted" in line or "passage(s) extracted" in line:
            log_text.append(line + "\n", style="green")
        elif "influence" in line.lower():
            log_text.append(line + "\n", style="magenta")
        elif "tag" in line.lower() and ("extracted" in line or "[" in line):
            log_text.append(line + "\n", style="yellow")
        elif line.startswith("FAIL"):
            log_text.append(line + "\n", style="dim red")
        elif "ERROR:" in line:
            log_text.append(line + "\n", style="red")
        elif "SUCCESS:" in line or "COMPLETE" in line:
            log_text.append(line + "\n", style="bold green")
        elif "FAILED:" in line or "PARTIAL" in line:
            log_text.append(line + "\n", style="bold red")
        elif "PHASE" in line or "Ingesting" in line:
            log_text.append(line + "\n", style="bold yellow")
        elif line.startswith("Added:") or ("Saved" in line and "source" in line):
            log_text.append(line + "\n", style="cyan")
        elif line.startswith("[") and "]" in line:
            log_text.append(line + "\n", style="bold dim")
        else:
            log_text.append(line + "\n", style="dim")
    layout["log"].update(Panel(log_text, title="[bold]Live Log[/]", border_style="blue"))

    # ── Credits ───────────────────────────────────────────────────────────
    layout["credits"].update(make_credits_panel(data_stats, proj, live))

    # ── Footer ────────────────────────────────────────────────────────────
    kb = file_size / 1024
    st = health["status"]
    sm = health["stale_mins"]

    if st == "running":
        health_icon  = "● RUNNING"
        health_color = "bold green"
        health_note  = f"output {sm:.0f}m ago" if sm > 1 else "live"
    elif st == "stale":
        health_icon  = "● STALE"
        health_color = "bold yellow"
        health_note  = f"no output for {sm:.0f}m — may be hung"
    else:
        health_icon  = "● DEAD"
        health_color = "bold red"
        health_note  = f"process not found  |  last output {sm:.0f}m ago"

    footer_text = Text.from_markup(
        f"[{health_color}]{health_icon}[/]  [{health_color}]{health_note}[/]  "
        f"[dim]|  last output: {health['last_updated']}  "
        f"|  checked: {health['checked_at']}  "
        f"|  {kb:.1f} KB  |  Ctrl+C to exit[/]"
    )
    layout["footer"].update(Panel(footer_text))

    return layout


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global MODEL_KEY, MODEL_INFO, MODEL_NAME, PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, BUDGET

    parser = argparse.ArgumentParser(description="Live monitor for batch ingestion runs")
    parser.add_argument("output_file", nargs="?", help="Output file to monitor")
    parser.add_argument("--file", help="Artist list file (default: auto-detect)")
    parser.add_argument("--budget", type=float, default=73.06, help="Budget cap in dollars")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                       default=MODEL_KEY, help="Model for cost estimation")
    args = parser.parse_args()

    MODEL_KEY = args.model
    MODEL_INFO = MODELS[MODEL_KEY]
    MODEL_NAME = MODEL_INFO["name"]
    PRICE_INPUT_PER_M = MODEL_INFO["input_per_m"]
    PRICE_OUTPUT_PER_M = MODEL_INFO["output_per_m"]
    BUDGET = args.budget

    output_file = args.output_file or find_latest_output()

    if not output_file or not Path(output_file).exists():
        if not PROGRESS_FILE.exists():
            print("No output file or batch_progress.json found.")
            print("Start a batch run first, or pass output file path as argument.")
            sys.exit(1)
        output_file = None

    all_artists = load_artist_list(args.file)
    if not all_artists:
        print("Warning: could not load artist list")

    console = Console()
    tick = 0

    with Live(console=console, refresh_per_second=2, screen=True) as live_display:
        data_stats = load_data_stats()
        health = get_batch_health(output_file)
        while True:
            try:
                batch_progress = load_batch_progress()

                lines = []
                file_size = 0
                if output_file and Path(output_file).exists():
                    content = Path(output_file).read_text(errors="replace")
                    lines = content.splitlines()
                    file_size = Path(output_file).stat().st_size

                live_state = parse_live(lines, batch_progress)
                proj = calc_projection(data_stats, live_state, output_file, batch_progress)
                health = get_batch_health(output_file)

                tick += 1
                if tick % 10 == 0:
                    data_stats = load_data_stats()

                live_display.update(
                    make_dashboard(data_stats, live_state, proj, health,
                                 all_artists, output_file, file_size, batch_progress)
                )

                # Keep refreshing even when batch is done — dashboard stays up
                time.sleep(2 if live_state["done"] else 0.5)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
