import argparse
import os

import pandas as pd
import yfinance as yf


def safe_name(ticker: str) -> str:
    """Convert a ticker string to a filesystem-safe name.
    Args:
        ticker: Ticker symbol possibly containing characters not allowed in
            filenames, e.g. "AIR.PA" or "BRK/A".

    Returns:
        A string safe to use as a filename component.
    """
    return ticker.replace(".", "_").replace("/", "_")

def read_tickers(args) -> list[str]:
    """Return a list of tickers from CLI args or a tickers file.

    The function prefers tickers provided via the CLI (`args.tickers`). If no
    CLI tickers are provided, it reads `args.tickers_file` and returns the
    non-empty, non-comment lines as tickers.

    Args:
        args: Parsed argument namespace with the following attributes:
            - `tickers`: Optional list of ticker strings provided via CLI.
            - `tickers_file`: Path to a file containing one ticker per line.

    Returns:
        A list of ticker strings.

    Raises:
        FileNotFoundError: If the `tickers_file` does not exist when
            `tickers` is not provided.
    """
    if (args.tickers):
        return args.tickers

    with open(args.tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return tickers

def download_close(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Download daily close prices for a ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker name.
        start: Start date in "YYYY-MM-DD" format (inclusive). If None, uses
            Yahoo default.
        end: End date in "YYYY-MM-DD" format (exclusive in yfinance). If None,
            downloads up to latest.

    Returns:
        A DataFrame with columns ["Date", "Close"] sorted by Date.

    Raises:
        ValueError: If no data is returned for the given ticker/date range.
    """
    df = yf.download(ticker, 
                     start=start, 
                     end=end, 
                     interval="1d", 
                     progress=False, 
                     multi_level_index=False,
                     auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")

    df = df.reset_index()
    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], utc = False).dt.tz_localize(None)
    out = out.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return out

def main():
    """CLI entrypoint: download close prices and write Parquet files.

    The function parses CLI arguments, obtains the list of tickers via
    `read_tickers`, downloads each ticker's close prices with
    `download_close`, and writes the result to `<out_dir>/<safe_name(ticker)>.parquet`.
    """
    p = argparse.ArgumentParser(description="Download daily Close prices and save to Parquet.")
    p.add_argument("--tickers", nargs="*", default=None, help="Tickers via CLI, e.g. AIR.PA SAP.DE")
    p.add_argument("--tickers-file", default="configs/tickers.txt", help="File with 1 Ticker per line.")
    p.add_argument("--start", default="2015-01-01", help="Start date: YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    p.add_argument("--out-dir", default="data/raw", help="Output directory for Parquet files.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if a Parquet file already exists.")
    args = p.parse_args()

    tickers = read_tickers(args)
    os.makedirs(args.out_dir, exist_ok=True)

    succeeded: list[str] = []
    failed: list[str] = []

    for t in tickers:
        path = os.path.join(args.out_dir, f"{safe_name(t)}.parquet")
        if os.path.exists(path) and not args.force:
            print(f"[SKIP] {t}: file already exists at {path}. Use --force to re-download.")
            succeeded.append(t)
            continue
        try:
            close_df = download_close(t, args.start, args.end)
            close_df.to_parquet(path, index=False)
            print(f"[OK] {t}: {len(close_df)} rows -> {path}")
            succeeded.append(t)
        except Exception as exc:
            print(f"[ERROR] {t}: download failed – {exc}")
            failed.append(t)

    print(f"\nSummary: {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")


if __name__ == "__main__":
    main()