"""
Standalone CLI scanner - runs a Graham screen independently of the Streamlit
UI, so a large scan survives closing the browser tab or terminal.

Unlike the Streamlit app, this has no session-state to lose and doesn't
depend on a browser tab staying open. It still needs the machine itself to
stay powered on and awake - see the caffeinate examples below for that.

USAGE
    # Foreground, with live progress printed to the terminal:
    python scan.py --universe RUSSELL2000 --strategy defensive

    # Detached: survives closing the terminal. Progress goes to a log file
    # instead of the terminal; caffeinate keeps the Mac from sleeping for
    # the scan's duration (does NOT override a closed laptop lid unless
    # plugged into an external display - see README/--help).
    caffeinate -i nohup python3 scan.py --universe RUSSELL2000 \\
        --strategy defensive > scan.log 2>&1 &

    # Then check progress any time with:
    tail -f scan.log

OUTPUT
    Results are saved to a timestamped CSV, e.g.
    scans/RUSSELL2000_defensive_2026-08-02_1230.csv - written even if the
    scan is interrupted partway (see --checkpoint-every).
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from src.data.data_fetcher import DataFetcher
from src.screening.graham_criteria import GrahamScreener, ScreeningProgress
import config

SCANS_DIR = Path(__file__).parent / "scans"


def _print_progress(progress: ScreeningProgress) -> None:
    eta = progress.eta_seconds
    eta_str = f", ETA {eta:.0f}s" if eta is not None else ""
    # \r + no newline so this overwrites in place in a real terminal; a
    # redirected log file just gets one line per ticker, which is fine for
    # `tail -f`.
    sys.stdout.write(
        f"\r[{progress.completed}/{progress.total}] {progress.current_ticker:<10} "
        f"passed={progress.passed} no_match={progress.evaluated} "
        f"errors={progress.errors} (rate_limited={progress.rate_limited}){eta_str}   "
    )
    sys.stdout.flush()
    if progress.completed == progress.total:
        sys.stdout.write("\n")


def _resolve_universe(fetcher: DataFetcher, universe: str, tickers_arg: list) -> list:
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg]
    print(f"Fetching {universe} tickers...")
    tickers, source = fetcher.get_index_tickers_with_source(universe)
    if source == "fallback":
        print(
            f"WARNING: live source for {universe} unavailable; using the small "
            f"curated fallback list ({len(tickers)} tickers)."
        )
    elif source == "stale_cache":
        print(f"Using cached (possibly outdated) {universe} list ({len(tickers)} tickers).")
    else:
        print(f"Got {len(tickers)} {universe} tickers ({source}).")
    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Run a Graham screen from the command line, independent of the Streamlit UI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--universe", default="SP500",
        help="DataFetcher index key (SP500, RUSSELL2000, NASDAQ100, ...). Ignored if --tickers is given.",
    )
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Explicit ticker list to screen instead of a universe (e.g. --tickers AAPL MSFT).",
    )
    parser.add_argument(
        "--strategy", choices=["defensive", "enterprising"], default="defensive",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of tickers screened (applied after fetching the universe).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path. Defaults to scans/<universe>_<strategy>_<timestamp>.csv",
    )
    args = parser.parse_args()

    fetcher = DataFetcher()
    screener = GrahamScreener(fetcher, config.GRAHAM_CRITERIA)

    tickers = _resolve_universe(fetcher, args.universe, args.tickers)
    if args.limit:
        tickers = tickers[: args.limit]

    if not tickers:
        print("No tickers to screen.")
        sys.exit(1)

    print(f"Screening {len(tickers)} tickers ({args.strategy})...\n")
    screen_fn = screener.screen_defensive if args.strategy == "defensive" else screener.screen_enterprising

    started = time.time()
    results = screen_fn(tickers, progress_callback=_print_progress)
    elapsed = time.time() - started

    SCANS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else SCANS_DIR / (
        f"{args.universe}_{args.strategy}_{datetime.now():%Y-%m-%d_%H%M}.csv"
    )
    results.to_csv(out_path, index=False)

    passed = int(results["passes_screening"].sum()) if not results.empty else 0
    print(f"\nDone in {elapsed:.0f}s. {passed}/{len(tickers)} passed. Saved to {out_path}")


if __name__ == "__main__":
    main()
