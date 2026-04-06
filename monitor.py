"""
Live monitor for batch_full.py runs.
Usage: python3 monitor.py [output_file]
       python3 monitor.py          # auto-finds latest task output

Ground truth for completion + billing comes from data/artists/ (accurate across
all runs). The output file is used only for the live "what's happening right now"
overlay — current artist, current phase, live log lines.
"""

import sys
import re
import time
import glob
import json
import os
import datetime
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
SEED_FILE  = BASE_DIR / "data/seed_artists.txt"
DATA_DIR   = BASE_DIR / "data/artists"

# claude-sonnet-4-20250514 pricing (per million tokens)
MODEL_NAME         = "claude-sonnet-4-20250514"
PRICE_INPUT_PER_M  = 3.00
PRICE_OUTPUT_PER_M = 15.00
AVG_INPUT_TOKENS   = 2500
AVG_OUTPUT_TOKENS  = 900
BUDGET             = 100.00

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
    "search_interviews": "🔍 Search Interviews",
    "scrape_interviews": "📥 Scrape Interviews",
    "extract_quotes":    "✂️  Extract Quotes",
    "search_reviews":    "🔍 Search Reviews",
    "scrape_reviews":    "📥 Scrape Reviews",
    "extract_critic":    "✂️  Extract Critic",
    "global_pipeline":   "⚙️  Global Pipeline",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_output():
    pattern = os.path.join(TASK_DIR, "**", "tasks", "*.output")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_seed_artists():
    if SEED_FILE.exists():
        return [l.strip() for l in SEED_FILE.read_text().splitlines() if l.strip()]
    return []


def normalize(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def get_batch_health(output_file):
    """
    Returns a dict describing the health of the batch process:
      - running:        True if batch_full.py process exists
      - stale_mins:     minutes since output file last changed
      - status:         'running' | 'stale' | 'dead'
      - last_updated:   human-readable time of last output line
      - checked_at:     current time string
    """
    now = time.time()
    checked_at = datetime.datetime.now().strftime("%-I:%M:%S %p")

    # Check for live process
    running = False
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", "batch_full.py"],
            capture_output=True, text=True
        )
        running = result.returncode == 0
    except Exception:
        pass

    # Check output file freshness
    stale_mins = 0.0
    last_updated = "—"
    try:
        mtime = os.path.getmtime(output_file)
        stale_mins = (now - mtime) / 60.0
        last_updated = datetime.datetime.fromtimestamp(mtime).strftime("%-I:%M:%S %p")
    except Exception:
        pass

    if running and stale_mins < 5:
        status = "running"
    elif running and stale_mins >= 5:
        status = "stale"
    else:
        status = "dead"

    return dict(
        running=running,
        stale_mins=stale_mins,
        status=status,
        last_updated=last_updated,
        checked_at=checked_at,
    )


# ── Ground-truth reads from data/ ────────────────────────────────────────────

def load_data_stats():
    """
    Read data/artists/* for ground-truth across ALL runs:
      - status per artist: SUCCEEDED (valid corpus), FAILED (attempted, not valid), PENDING
      - total API calls:   sources with text = one extract call each
      - total quotes / passages extracted
    """
    result = dict(
        status={},           # artist_id -> STATUS_*
        total_api_calls=0,
        total_quotes=0,
        total_passages=0,
        artists_valid=set(),
        artists_attempted=set(),
    )

    if not DATA_DIR.exists():
        return result

    for artist_dir in DATA_DIR.iterdir():
        if not artist_dir.is_dir():
            continue
        aid = artist_dir.name
        has_valid    = False
        has_attempted = False

        # Interview sources → API calls (sources with scraped text)
        for fname in ("sources.json", "review_sources.json"):
            p = artist_dir / fname
            if p.exists():
                try:
                    sources = json.loads(p.read_text())
                    result["total_api_calls"] += sum(1 for s in sources if s.get("text"))
                except Exception:
                    pass

        # Quotes corpus
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

        # Critic corpus
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

        if has_valid:
            result["status"][aid] = STATUS_SUCCEEDED
            result["artists_valid"].add(aid)
        elif has_attempted:
            result["status"][aid] = STATUS_FAILED
            result["artists_attempted"].add(aid)

    return result


# ── Current-run output file parse (live overlay only) ────────────────────────

def parse_live(lines):
    """
    Parse the current run's output file for:
      - which artist is actively being processed right now
      - current phase + source
      - which artists were resolved in this run (for FAILED marking)
      - recent log lines for the live log panel
      - whether this run is done
    """
    live = dict(
        current_artist="",
        current_artist_id="",
        current_num=0,
        total=128,
        current_phase="",
        current_source="",
        global_step="",
        failed_this_run=set(),
        recent=[],
        done=False,
        elapsed_min=0.0,
        artists_done_cur=0,
        billed_cur=0,
    )

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
                live["current_num"] = int(m.group(1))
                live["total"]       = int(m.group(2))
                live["current_artist"]    = m.group(3)
                live["current_artist_id"] = normalize(m.group(3))
                live["current_phase"]  = ""
                live["current_source"] = ""
            after_sep = False
            continue

        after_sep = False

        if "PHASE 1: INTERVIEW QUOTES" in s:
            live["current_phase"] = "search_interviews"
        elif "PHASE 2: CRITIC DISCOURSE" in s:
            live["current_phase"] = "search_reviews"
        elif "[SEARCH] Finding interview" in s:
            live["current_phase"] = "search_interviews"
        elif "[SCRAPE INTERVIEWS]" in s:
            live["current_phase"] = "scrape_interviews"
        elif "[EXTRACT QUOTES]" in s:
            live["current_phase"] = "extract_quotes"
        elif "[SEARCH REVIEWS]" in s:
            live["current_phase"] = "search_reviews"
        elif "[SCRAPE REVIEWS]" in s:
            live["current_phase"] = "scrape_reviews"
        elif "[EXTRACT CRITIC DISCOURSE]" in s:
            live["current_phase"] = "extract_critic"
        elif "GLOBAL DUAL-SIGNAL PIPELINE" in s:
            live["current_phase"] = "global_pipeline"

        m = re.match(r'^\[(\d+)/(\d+)\]\s+(.+)$', s)
        if m:
            live["current_source"] = m.group(3)

        m = re.match(r'^Running (\S+\.py)\.\.\.', s)
        if m:
            live["global_step"] = m.group(1)

        m = re.match(r'^FAILED: (\S+) has no valid corpora', s)
        if m:
            live["failed_this_run"].add(m.group(1))

        if "Log saved to:" in s:
            live["done"] = True

        if re.match(r'^\s+OK:', s):
            live["billed_cur"] += 1

        if re.match(r'^(?:SUCCESS:|FAILED:|Already has valid corpus)', s):
            live["artists_done_cur"] += 1

        if s and not re.match(r'^[=─]{10,}$', s):
            live["recent"].append(s)

    live["recent"] = live["recent"][-22:]
    return live


def calc_projection(data_stats, live, current_file):
    """Combine ground-truth data stats + current run timing for projections."""
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

    # Cost so far = actual sources processed across all runs (from data/)
    proj["cost_so_far"] = data_stats["total_api_calls"] * cost_per_call

    # Rate from current run
    try:
        stat    = os.stat(current_file)
        created = getattr(stat, "st_birthtime", stat.st_ctime)
        elapsed = (time.time() - created) / 60.0
    except Exception:
        elapsed = 0.0

    done_cur   = live["artists_done_cur"]
    billed_cur = live["billed_cur"]

    if done_cur > 0:
        proj["calls_per_artist"] = billed_cur / done_cur
        proj["mins_per_artist"]  = elapsed / done_cur
    else:
        proj["calls_per_artist"] = 13.0   # fallback from historical avg
        proj["mins_per_artist"]  = 3.0

    total = live["total"] or 128
    cpa   = proj["calls_per_artist"]
    proj["projected_total_calls"] = int(total * cpa)
    proj["projected_total_cost"]  = proj["projected_total_calls"] * cost_per_call

    pct = proj["projected_total_cost"] / BUDGET * 100
    proj["budget_pct"] = pct
    proj["budget_status"] = "green" if pct < 55 else "yellow" if pct < 80 else "red"

    # ETA from current run rate
    artists_done_total = len(data_stats["artists_valid"])
    remaining = max(0, total - artists_done_total)
    proj["mins_remaining"] = remaining * proj["mins_per_artist"]
    eta_dt = datetime.datetime.now() + datetime.timedelta(minutes=proj["mins_remaining"])
    proj["eta_str"] = eta_dt.strftime("%-I:%M %p")

    return proj


# ── Panels ────────────────────────────────────────────────────────────────────

def make_credits_panel(data_stats, proj, live):
    billed    = data_stats["total_api_calls"]
    extracted = data_stats["total_quotes"] + data_stats["total_passages"]
    quotes    = data_stats["total_quotes"]
    passages  = data_stats["total_passages"]

    cost_so_far  = proj["cost_so_far"]
    proj_total   = proj["projected_total_cost"]
    budget_left  = max(0.0, BUDGET - cost_so_far)
    bstatus      = proj["budget_status"]
    cpa          = proj["calls_per_artist"]
    mpa          = proj["mins_per_artist"]
    mins_rem     = proj["mins_remaining"]
    eta_str      = proj["eta_str"]

    bc = {"green": "green", "yellow": "yellow", "red": "red"}[bstatus]

    # ── SPENT bar (actual spend vs budget) ──────────────────────────────
    spent_pct  = min(100.0, cost_so_far / BUDGET * 100)
    proj_pct   = min(100.0, proj_total  / BUDGET * 100)
    bar_w      = 18
    spent_fill = max(1, int(bar_w * spent_pct / 100)) if cost_so_far > 0 else 0
    proj_fill  = max(spent_fill, int(bar_w * proj_pct / 100))
    # bar: spent=solid, projected=light, remainder=empty
    bar_spent  = "█" * spent_fill
    bar_proj   = "▒" * max(0, proj_fill - spent_fill)
    bar_empty  = "░" * max(0, bar_w - proj_fill)
    bar = (f"[bold yellow]{bar_spent}[/]"
           f"[{bc} dim]{bar_proj}[/]"
           f"[dim]{bar_empty}[/]")

    # ── Budget verdict ───────────────────────────────────────────────────
    if bstatus == "green":
        verdict = f"[bold green]● COVERED[/]  proj. ${proj_total:.0f} of ${BUDGET:.0f}"
    elif bstatus == "yellow":
        verdict = f"[bold yellow]● TIGHT[/]  proj. ${proj_total:.0f} of ${BUDGET:.0f}"
    else:
        verdict = f"[bold red]● AT RISK[/]  proj. ${proj_total:.0f} of ${BUDGET:.0f}"

    time_rem = (f"{mins_rem/60:.1f} hrs" if mins_rem > 90
                else f"{mins_rem:.0f} min" if mins_rem > 0 else "—")

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("k", style="dim", width=17)
    t.add_column("v", justify="right")

    # ── Spent ────────────────────────────────────────────────────────────
    t.add_row("[bold]SPENT[/]",       f"[bold yellow]${cost_so_far:.2f}[/]  [dim]/ ${BUDGET:.0f}[/]")
    t.add_row("",                     Text.from_markup(
                                          f"{bar}  [bold yellow]{spent_pct:.0f}%[/]"))
    t.add_row("[dim]Budget left[/]",  f"[{'red' if budget_left < 20 else 'bold green'}]${budget_left:.2f}[/]")
    t.add_row("",                     "")

    # ── Projection ───────────────────────────────────────────────────────
    t.add_row("[bold]PROJECTED[/]",   Text.from_markup(verdict))
    t.add_row("Total cost",           f"[{bc}]~${proj_total:.2f}[/]")
    t.add_row("Total API calls",      f"[dim]{proj['projected_total_calls']:,}[/]")
    t.add_row("",                     "")

    # ── Timing ───────────────────────────────────────────────────────────
    t.add_row("[bold]ETA[/]",         f"[bold cyan]{eta_str}[/]")
    t.add_row("Time left",            f"[bold]{time_rem}[/]")
    t.add_row("Min/artist",           f"[dim]{mpa:.1f}  ({cpa:.1f} calls)[/]")
    t.add_row("",                     "")

    # ── Usage breakdown ──────────────────────────────────────────────────
    t.add_row("[bold]USAGE[/]",       f"[bold]{billed:,} API calls[/]")
    t.add_row("Quotes",               f"[green]{quotes:,}[/]")
    t.add_row("Passages",             f"[cyan]{passages:,}[/]")
    t.add_row("Total items",          f"[bold cyan]{extracted:,}[/]")
    t.add_row("",                     "")
    t.add_row("[dim]console.anthropic.com[/]", "")

    return Panel(t, title=f"[bold]API Credits  /  ${BUDGET:.0f} Budget[/]",
                 border_style=bc)


def make_dashboard(data_stats, live, proj, health, all_artists, output_file, file_size):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="stack",   ratio=3),
        Layout(name="middle",  ratio=4),
        Layout(name="credits", ratio=2),
    )
    layout["middle"].split_column(
        Layout(name="current", size=10),
        Layout(name="log"),
    )

    # ── Compute overall status for each artist ─────────────────────────────
    # Priority: ACTIVE (current run) > SUCCEEDED (data/) > FAILED (current run)
    #           > SKIPPED (already valid, skipped by batch) > PENDING
    artist_status = {}
    artist_phases = {}
    for artist in all_artists:
        aid = normalize(artist)
        if aid in data_stats["status"]:
            artist_status[aid] = data_stats["status"][aid]   # SUCCEEDED from data
        else:
            artist_status[aid] = STATUS_PENDING
        artist_phases[aid] = ""

    # Overlay current run state
    cur_aid = live["current_artist_id"]
    if cur_aid:
        artist_status[cur_aid] = STATUS_ACTIVE
        artist_phases[cur_aid] = live["current_phase"]

    for aid in live["failed_this_run"]:
        if artist_status.get(aid) not in (STATUS_SUCCEEDED, STATUS_ACTIVE):
            artist_status[aid] = STATUS_FAILED

    # ── Header ────────────────────────────────────────────────────────────
    total       = live["total"] or len(all_artists)
    n_succeeded = len(data_stats["artists_valid"])
    n_attempted = len(data_stats["artists_attempted"])
    n_active    = 1 if cur_aid and artist_status.get(cur_aid) == STATUS_ACTIVE else 0
    n_touched   = n_succeeded + n_attempted + n_active
    n_pending   = max(0, total - n_touched)
    pct         = (n_touched / total * 100) if total else 0

    bar_w  = 46
    filled = int(bar_w * pct / 100)
    bar    = "█" * filled + "░" * (bar_w - filled)
    run_status = "[bold green]COMPLETE[/]" if live["done"] else "[bold yellow]RUNNING[/]"

    header = Text.from_markup(
        f"  [bold cyan]Basilect Engine[/]  Dual-Signal Batch Ingest  {run_status}\n"
        f"  [green]{bar}[/]  [bold]{pct:.0f}%[/]  "
        f"[dim]{n_touched}/{total} touched  "
        f"([green]✓{n_succeeded} valid[/]  "
        f"[red]✗{n_attempted} partial[/]  "
        f"[yellow]▶{n_active} active[/]  "
        f"·{n_pending} pending)[/]"
    )
    layout["header"].update(Panel(header))

    # ── Artist Stack ──────────────────────────────────────────────────────
    stack_text = Text()
    for artist in all_artists:
        aid   = normalize(artist)
        s     = artist_status.get(aid, STATUS_PENDING)
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

    layout["stack"].update(Panel(stack_text, title="[bold]All Artists[/]", border_style="cyan"))

    # ── Now Processing ────────────────────────────────────────────────────
    cur       = live["current_artist"] or "—"
    cur_phase = PHASE_ICONS.get(live["current_phase"], live["current_phase"] or "—")
    cur_src   = live["current_source"] or ""

    ct = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    ct.add_column("k", style="dim", width=12)
    ct.add_column("v")
    ct.add_row("Artist",   f"[bold cyan]{cur}[/]  [dim]({live['current_num']}/{live['total']})[/]")
    ct.add_row("Phase",    f"[yellow]{cur_phase}[/]")
    if cur_src:
        ct.add_row("Source", f"[dim]{cur_src}[/]")
    if live["global_step"]:
        ct.add_row("Pipeline", f"[magenta]{live['global_step']}[/]")
    layout["current"].update(Panel(ct, title="[bold]Now Processing[/]", border_style="yellow"))

    # ── Live Log ──────────────────────────────────────────────────────────
    log_text = Text()
    for line in live["recent"]:
        if "quote(s) extracted" in line or "passage(s) extracted" in line:
            log_text.append(line + "\n", style="green")
        elif line.startswith("FAIL"):
            log_text.append(line + "\n", style="dim red")
        elif "ERROR:" in line:
            log_text.append(line + "\n", style="red")
        elif "SUCCESS:" in line:
            log_text.append(line + "\n", style="bold green")
        elif "FAILED:" in line:
            log_text.append(line + "\n", style="bold red")
        elif "PHASE" in line or "Ingesting" in line:
            log_text.append(line + "\n", style="bold yellow")
        elif line.startswith("Added:") or ("Saved" in line and "source" in line):
            log_text.append(line + "\n", style="cyan")
        elif line.startswith("[SEARCH]") or line.startswith("[SCRAPE") or line.startswith("[EXTRACT"):
            log_text.append(line + "\n", style="bold dim")
        else:
            log_text.append(line + "\n", style="dim")
    layout["log"].update(Panel(log_text, title="[bold]Live Log[/]", border_style="blue"))

    # ── Credits ───────────────────────────────────────────────────────────
    layout["credits"].update(make_credits_panel(data_stats, proj, live))

    # ── Footer: health + timestamp ────────────────────────────────────────
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


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else find_latest_output()

    if not output_file or not Path(output_file).exists():
        print("No output file found. Pass path as argument or start a batch run first.")
        sys.exit(1)

    all_artists = load_seed_artists()
    if not all_artists:
        print(f"Warning: could not load {SEED_FILE}")

    console = Console()
    tick    = 0

    with Live(console=console, refresh_per_second=2, screen=True) as live_display:
        data_stats = load_data_stats()
        health     = get_batch_health(output_file)
        while True:
            try:
                content    = Path(output_file).read_text(errors="replace")
                lines      = content.splitlines()
                file_size  = Path(output_file).stat().st_size
                live_state = parse_live(lines)
                proj       = calc_projection(data_stats, live_state, output_file)
                health     = get_batch_health(output_file)

                # Reload data stats every 10 ticks (~5s) to catch new completions
                tick += 1
                if tick % 10 == 0:
                    data_stats = load_data_stats()

                live_display.update(
                    make_dashboard(data_stats, live_state, proj, health, all_artists, output_file, file_size)
                )

                if live_state["done"]:
                    data_stats = load_data_stats()
                    proj   = calc_projection(data_stats, live_state, output_file)
                    health = get_batch_health(output_file)
                    live_display.update(
                        make_dashboard(data_stats, live_state, proj, health, all_artists, output_file, file_size)
                    )
                    time.sleep(4)
                    break

                time.sleep(0.5)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
