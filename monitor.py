"""
Live monitor for batch_full.py runs.
Usage: python3 monitor.py [output_file]
       python3 monitor.py          # auto-finds latest task output
"""

import sys
import re
import time
import glob
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

TASK_DIR = "/private/tmp/claude-501"
SEED_FILE = "data/seed_artists.txt"

# claude-sonnet-4-20250514 pricing (per million tokens)
MODEL_NAME         = "claude-sonnet-4-20250514"
PRICE_INPUT_PER_M  = 3.00
PRICE_OUTPUT_PER_M = 15.00
AVG_INPUT_TOKENS   = 2500   # article text + system prompt
AVG_OUTPUT_TOKENS  = 900    # JSON quotes/passages
BUDGET             = 100.00

STATUS_PENDING    = "pending"
STATUS_ACTIVE     = "active"
STATUS_SUCCEEDED  = "succeeded"
STATUS_FAILED     = "failed"
STATUS_SKIPPED    = "skipped"

STATUS_ICON = {
    STATUS_PENDING:   ("·",  "dim"),
    STATUS_ACTIVE:    ("▶",  "bold yellow"),
    STATUS_SUCCEEDED: ("✓",  "bold green"),
    STATUS_FAILED:    ("✗",  "bold red"),
    STATUS_SKIPPED:   ("~",  "dim cyan"),
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


def find_latest_output():
    pattern = os.path.join(TASK_DIR, "**", "tasks", "*.output")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_all_outputs():
    pattern = os.path.join(TASK_DIR, "**", "tasks", "*.output")
    return glob.glob(pattern, recursive=True)


def load_seed_artists():
    path = Path(SEED_FILE)
    if path.exists():
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return []


def normalize(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def count_api_usage_all_files(current_file=None):
    """
    Scan ALL task output files (all previous runs + current) and tally:
      - billed_ok:        calls that succeeded (API responded, tokens used)
      - billed_error:     calls where API responded but JSON parse failed (still billed)
      - rejected:         calls rejected for low credits (not billed)
      - quotes_extracted: total quote/passage items extracted
    """
    totals = dict(billed_ok=0, billed_error=0, rejected=0, quotes_extracted=0)

    all_files = find_all_outputs()
    # Deduplicate by inode to avoid double-counting symlinks etc.
    seen = set()
    files_to_scan = []
    for f in all_files:
        try:
            ino = os.stat(f).st_ino
            if ino not in seen:
                seen.add(ino)
                files_to_scan.append(f)
        except OSError:
            pass

    for fpath in files_to_scan:
        try:
            text = Path(fpath).read_text(errors="replace")
        except OSError:
            continue

        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r'^OK:\s+\d+', stripped):
                totals["billed_ok"] += 1
                m = re.search(r'(\d+)\s+(?:quote|passage)', stripped)
                if m:
                    totals["quotes_extracted"] += int(m.group(1))
            elif "Failed to parse JSON" in stripped:
                totals["billed_error"] += 1
            elif "credit balance is too low" in stripped or "API call failed" in stripped:
                totals["rejected"] += 1

    return totals


def calc_projection(current_file):
    """
    From the current run's output file, derive:
      - elapsed minutes
      - artists completed
      - billed calls in this run
      - calls_per_artist average
      - mins_per_artist average
      - total artists
    Returns a dict of projection stats.
    """
    proj = dict(
        elapsed_min=0.0,
        artists_done_cur=0,
        billed_cur=0,
        calls_per_artist=0.0,
        mins_per_artist=0.0,
        total_artists=128,
        projected_total_calls=0,
        projected_total_cost=0.0,
        cost_so_far=0.0,
        cost_remaining=0.0,
        budget_pct=0.0,
        budget_status="green",   # green / yellow / red
        eta=None,
        eta_str="—",
        mins_remaining=0.0,
    )

    try:
        stat = os.stat(current_file)
        created = getattr(stat, "st_birthtime", stat.st_ctime)
        now = time.time()
        proj["elapsed_min"] = (now - created) / 60.0

        text = Path(current_file).read_text(errors="replace")

        # Artist completed count (context-aware batch lines)
        after_sep = False
        highest_num = 0
        total = 128
        for line in text.splitlines():
            s = line.strip()
            if re.match(r'^={50,}$', s):
                after_sep = True
                continue
            if after_sep:
                m = re.match(r'^\[(\d+)/(\d+)\]\s+', s)
                if m:
                    highest_num = max(highest_num, int(m.group(1)))
                    total = int(m.group(2))
                after_sep = False
        proj["total_artists"] = total

        done_cur = len(re.findall(
            r'(?:SUCCESS:|FAILED:|Already has valid corpus)', text))
        billed_cur = len(re.findall(r'^\s+OK:', text, re.MULTILINE))

        proj["artists_done_cur"] = done_cur
        proj["billed_cur"] = billed_cur

        if done_cur > 0:
            proj["calls_per_artist"] = billed_cur / done_cur
            proj["mins_per_artist"]  = proj["elapsed_min"] / done_cur

        # All-time billed calls for cost-so-far
        all_files  = find_all_outputs()
        all_billed = 0
        for f in all_files:
            try:
                t = Path(f).read_text(errors="replace")
                all_billed += len(re.findall(r'^\s+OK:', t, re.MULTILINE))
                all_billed += len(re.findall(r'Failed to parse JSON', t))
            except OSError:
                pass
        cost_per_call = (
            AVG_INPUT_TOKENS  / 1_000_000 * PRICE_INPUT_PER_M +
            AVG_OUTPUT_TOKENS / 1_000_000 * PRICE_OUTPUT_PER_M
        )
        proj["cost_so_far"] = all_billed * cost_per_call

        # Projected total
        cpa = proj["calls_per_artist"] if proj["calls_per_artist"] > 0 else 13.0
        proj["projected_total_calls"] = int(total * cpa)
        proj["projected_total_cost"]  = proj["projected_total_calls"] * cost_per_call
        proj["cost_remaining"]        = max(0.0, proj["projected_total_cost"] - proj["cost_so_far"])

        pct = proj["projected_total_cost"] / BUDGET * 100
        proj["budget_pct"] = pct
        if pct < 55:
            proj["budget_status"] = "green"
        elif pct < 80:
            proj["budget_status"] = "yellow"
        else:
            proj["budget_status"] = "red"

        # ETA
        mpa = proj["mins_per_artist"] if proj["mins_per_artist"] > 0 else 3.0
        remaining_artists = total - done_cur
        proj["mins_remaining"] = remaining_artists * mpa
        eta_dt = datetime.datetime.now() + datetime.timedelta(minutes=proj["mins_remaining"])
        proj["eta"] = eta_dt
        proj["eta_str"] = eta_dt.strftime("%-I:%M %p")

    except Exception:
        pass

    return proj


def parse_output(lines, all_artists):
    status = {normalize(a): STATUS_PENDING for a in all_artists}
    phases = {normalize(a): "" for a in all_artists}

    state = {
        "total": len(all_artists),
        "current_num": 0,
        "current_artist": "",
        "current_artist_id": "",
        "current_phase": "",
        "current_source": "",
        "status": status,
        "phases": phases,
        "succeeded": [],
        "failed": [],
        "skipped": [],
        "recent": [],
        "global_pipeline": False,
        "global_step": "",
        "done": False,
    }

    after_sep = False

    for line in lines:
        stripped = line.strip()

        if re.match(r'^={50,}$', stripped):
            after_sep = True
            continue
        if re.match(r'^─{50,}$', stripped):
            after_sep = False
            continue

        # Batch-level artist line: only right after ===
        if after_sep:
            m = re.match(r'^\[(\d+)/(\d+)\]\s+(.+)$', stripped)
            if m:
                state["current_num"] = int(m.group(1))
                state["total"] = int(m.group(2))
                state["current_artist"] = m.group(3)
                state["current_artist_id"] = normalize(m.group(3))
                state["current_phase"] = ""
                state["current_source"] = ""
                if state["current_artist_id"] in state["status"]:
                    state["status"][state["current_artist_id"]] = STATUS_ACTIVE
                after_sep = False
                continue

        after_sep = False

        m = re.match(r'^Ingesting \(FULL DUAL PIPELINE\): (.+)$', stripped)
        if m:
            aid = normalize(m.group(1))
            state["current_artist_id"] = aid
            if aid in state["status"]:
                state["status"][aid] = STATUS_ACTIVE

        if re.match(r'^Already has valid corpus.*skipping', stripped) and state["current_artist_id"]:
            aid = state["current_artist_id"]
            state["status"][aid] = STATUS_SKIPPED
            if aid not in state["skipped"]:
                state["skipped"].append(aid)

        # Phase
        if "PHASE 1: INTERVIEW QUOTES" in stripped:
            state["current_phase"] = "search_interviews"
        elif "PHASE 2: CRITIC DISCOURSE" in stripped:
            state["current_phase"] = "search_reviews"
        elif "[SEARCH] Finding interview" in stripped:
            state["current_phase"] = "search_interviews"
        elif "[SCRAPE INTERVIEWS]" in stripped:
            state["current_phase"] = "scrape_interviews"
        elif "[EXTRACT QUOTES]" in stripped:
            state["current_phase"] = "extract_quotes"
        elif "[SEARCH REVIEWS]" in stripped:
            state["current_phase"] = "search_reviews"
        elif "[SCRAPE REVIEWS]" in stripped:
            state["current_phase"] = "scrape_reviews"
        elif "[EXTRACT CRITIC DISCOURSE]" in stripped:
            state["current_phase"] = "extract_critic"
        elif "GLOBAL DUAL-SIGNAL PIPELINE" in stripped:
            state["current_phase"] = "global_pipeline"
            state["global_pipeline"] = True

        # Source-level line inside extract (not after ===)
        m = re.match(r'^\[(\d+)/(\d+)\]\s+(.+)$', stripped)
        if m:
            state["current_source"] = m.group(3)

        m = re.match(r'^Running (\S+\.py)\.\.\.', stripped)
        if m:
            state["global_step"] = m.group(1)

        m = re.match(r'^SUCCESS: (\S+) has at least one valid corpus', stripped)
        if m:
            aid = m.group(1)
            state["status"][aid] = STATUS_SUCCEEDED
            if aid not in state["succeeded"]:
                state["succeeded"].append(aid)
            state["current_phase"] = ""

        m = re.match(r'^FAILED: (\S+) has no valid corpora', stripped)
        if m:
            aid = m.group(1)
            state["status"][aid] = STATUS_FAILED
            if aid not in state["failed"]:
                state["failed"].append(aid)
            state["current_phase"] = ""

        if re.match(r'^  \+ (.+)', stripped):
            aid = re.match(r'^  \+ (.+)', stripped).group(1)
            state["status"][aid] = STATUS_SUCCEEDED
            if aid not in state["succeeded"]:
                state["succeeded"].append(aid)
        if re.match(r'^  - (.+)', stripped):
            aid = re.match(r'^  - (.+)', stripped).group(1)
            state["status"][aid] = STATUS_FAILED
            if aid not in state["failed"]:
                state["failed"].append(aid)
        if re.match(r'^  ~ (.+)', stripped):
            aid = re.match(r'^  ~ (.+)', stripped).group(1)
            state["status"][aid] = STATUS_SKIPPED
            if aid not in state["skipped"]:
                state["skipped"].append(aid)

        if "Log saved to:" in stripped:
            state["done"] = True

        if stripped and not re.match(r'^={50,}$', stripped) and not re.match(r'^─{50,}$', stripped):
            state["recent"].append(stripped)

    state["recent"] = state["recent"][-20:]

    if state["current_artist_id"] and state["current_phase"]:
        state["phases"][state["current_artist_id"]] = state["current_phase"]

    return state


def make_credits_panel(usage, proj):
    billed    = usage["billed_ok"] + usage["billed_error"]
    rejected  = usage["rejected"]
    extracted = usage["quotes_extracted"]

    cost_so_far     = proj["cost_so_far"]
    proj_total      = proj["projected_total_cost"]
    cost_remaining  = proj["cost_remaining"]
    budget_left     = max(0.0, BUDGET - cost_so_far)
    budget_pct      = proj["budget_pct"]
    bstatus         = proj["budget_status"]
    eta_str         = proj["eta_str"]
    mins_rem        = proj["mins_remaining"]
    cpa             = proj["calls_per_artist"]
    mpa             = proj["mins_per_artist"]

    # Budget status indicator
    if bstatus == "green":
        status_icon  = "●"
        status_color = "bold green"
        status_label = f"COVERED  ({budget_pct:.0f}% of ${BUDGET:.0f})"
    elif bstatus == "yellow":
        status_icon  = "●"
        status_color = "bold yellow"
        status_label = f"TIGHT  ({budget_pct:.0f}% of ${BUDGET:.0f})"
    else:
        status_icon  = "●"
        status_color = "bold red"
        status_label = f"AT RISK  ({budget_pct:.0f}% of ${BUDGET:.0f})"

    # Budget bar
    bar_w  = 16
    filled = min(bar_w, int(bar_w * budget_pct / 100))
    if bstatus == "green":
        bar = f"[green]{'█' * filled}[/][dim]{'░' * (bar_w - filled)}[/]"
    elif bstatus == "yellow":
        bar = f"[yellow]{'█' * filled}[/][dim]{'░' * (bar_w - filled)}[/]"
    else:
        bar = f"[red]{'█' * filled}[/][dim]{'░' * (bar_w - filled)}[/]"

    # Time remaining string
    if mins_rem > 90:
        time_rem_str = f"{mins_rem/60:.1f} hrs"
    elif mins_rem > 0:
        time_rem_str = f"{mins_rem:.0f} min"
    else:
        time_rem_str = "—"

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("k", style="dim", width=17)
    t.add_column("v", justify="right")

    t.add_row(f"[{status_color}]{status_icon}[/] Budget status",
              f"[{status_color}]{status_label}[/]")
    t.add_row("",  Text.from_markup(bar))
    t.add_row("",                "")
    t.add_row("Spent (est.)",     f"[bold]~${cost_so_far:.2f}[/]")
    t.add_row("Budget remaining", f"[{'red' if budget_left < 20 else 'green'}]~${budget_left:.2f}[/]")
    t.add_row("",                "")
    t.add_row("Proj. total cost", f"[bold {'red' if bstatus=='red' else 'yellow' if bstatus=='yellow' else 'green'}]~${proj_total:.2f}[/]")
    t.add_row("Proj. calls total",f"[dim]{proj['projected_total_calls']:,}[/]")
    t.add_row("Calls/artist",     f"[dim]{cpa:.1f}[/]")
    t.add_row("",                "")
    t.add_row("Time remaining",   f"[bold]{time_rem_str}[/]")
    t.add_row("ETA",              f"[bold cyan]{eta_str}[/]")
    t.add_row("Min/artist",       f"[dim]{mpa:.1f}[/]")
    t.add_row("",                "")
    t.add_row("─" * 17,          "─" * 10)
    t.add_row("",                "")
    t.add_row("Billed calls",     f"[bold]{billed}[/]")
    t.add_row("  ✓ OK",           f"[green]{usage['billed_ok']}[/]")
    t.add_row("  ⚠ Parse err",    f"[yellow]{usage['billed_error']}[/]")
    t.add_row("Rejected (free)",  f"[dim]{rejected}[/]")
    t.add_row("Items extracted",  f"[bold cyan]{extracted:,}[/]")
    t.add_row("",                "")
    t.add_row("[dim]Exact balance:[/]",  "")
    t.add_row("[dim]console.anthropic.com[/]", "")

    border = {"green": "green", "yellow": "yellow", "red": "red"}[bstatus]
    return Panel(t, title=f"[bold]API Credits  /  ${BUDGET:.0f} Budget[/]", border_style=border)


def make_dashboard(state, all_artists, usage, proj, output_file, file_size):
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

    # ── Header ──────────────────────────────────────────────────────────
    total = state["total"] or len(all_artists)
    done_count = len(state["succeeded"]) + len(state["failed"]) + len(state["skipped"])
    pct = (done_count / total * 100) if total else 0
    bar_w = 46
    filled = int(bar_w * pct / 100)
    bar = "█" * filled + "░" * (bar_w - filled)
    active = 1 if (state["current_artist_id"] and
                   state["status"].get(state["current_artist_id"]) == STATUS_ACTIVE) else 0
    pending = total - done_count - active
    status_str = "[bold green]COMPLETE[/]" if state["done"] else "[bold yellow]RUNNING[/]"

    header = Text.from_markup(
        f"  [bold cyan]Basilect Engine[/]  Dual-Signal Batch Ingest  {status_str}\n"
        f"  [green]{bar}[/]  [bold]{pct:.0f}%[/]  "
        f"[dim]{done_count}/{total}  "
        f"([green]✓{len(state['succeeded'])}[/] "
        f"[red]✗{len(state['failed'])}[/] "
        f"[cyan]~{len(state['skipped'])}[/] "
        f"[yellow]▶{active}[/] "
        f"·{pending} pending)[/]"
    )
    layout["header"].update(Panel(header))

    # ── Artist Stack ─────────────────────────────────────────────────────
    stack_text = Text()
    for artist in all_artists:
        aid = normalize(artist)
        s = state["status"].get(aid, STATUS_PENDING)
        icon, style = STATUS_ICON[s]
        phase = state["phases"].get(aid, "")
        phase_label = PHASE_ICONS.get(phase, "")

        if s == STATUS_ACTIVE:
            stack_text.append(f" {icon} ", style=style)
            stack_text.append(f"{artist}", style="bold yellow")
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

    # ── Current Artist ────────────────────────────────────────────────────
    cur        = state["current_artist"] or "—"
    cur_phase  = PHASE_ICONS.get(state["current_phase"], state["current_phase"] or "—")
    cur_src    = state["current_source"] or ""
    global_step = state["global_step"] or ""

    cur_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    cur_table.add_column("k", style="dim", width=12)
    cur_table.add_column("v")
    cur_table.add_row("Artist", f"[bold cyan]{cur}[/]  [dim]({state['current_num']}/{state['total']})[/]")
    cur_table.add_row("Phase",  f"[yellow]{cur_phase}[/]")
    if cur_src:
        cur_table.add_row("Source", f"[dim]{cur_src}[/]")
    if global_step:
        cur_table.add_row("Pipeline", f"[magenta]{global_step}[/]")
    layout["current"].update(Panel(cur_table, title="[bold]Now Processing[/]", border_style="yellow"))

    # ── Live Log ──────────────────────────────────────────────────────────
    log_text = Text()
    for line in state["recent"]:
        if re.match(r'^OK\s+\d+', line) or "quote(s) extracted" in line or "passage(s) extracted" in line:
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
    layout["credits"].update(make_credits_panel(usage, proj))

    # ── Footer ────────────────────────────────────────────────────────────
    kb = file_size / 1024
    layout["footer"].update(
        Panel(f"[dim]{output_file}  |  {kb:.1f} KB  |  Ctrl+C to exit[/]", style="dim")
    )

    return layout


def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else find_latest_output()

    if not output_file or not Path(output_file).exists():
        print("No output file found. Pass path as argument or start a batch run first.")
        sys.exit(1)

    all_artists = load_seed_artists()
    if not all_artists:
        print(f"Warning: could not load {SEED_FILE}, artist stack will be empty")

    console = Console()
    tick = 0

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        usage = count_api_usage_all_files(output_file)
        proj  = calc_projection(output_file)
        while True:
            try:
                content   = Path(output_file).read_text(errors="replace")
                lines     = content.splitlines()
                file_size = Path(output_file).stat().st_size
                state     = parse_output(lines, all_artists)

                # Rescan usage + projection every 10 ticks (~5s)
                tick += 1
                if tick % 10 == 0:
                    usage = count_api_usage_all_files(output_file)
                    proj  = calc_projection(output_file)

                live.update(make_dashboard(state, all_artists, usage, proj, output_file, file_size))

                if state["done"]:
                    usage = count_api_usage_all_files(output_file)
                    proj  = calc_projection(output_file)
                    live.update(make_dashboard(state, all_artists, usage, proj, output_file, file_size))
                    time.sleep(4)
                    break

                time.sleep(0.5)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
