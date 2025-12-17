import os
import pandas as pd
import yfinance as yf

class data_handler:
    CACHE_DIR = "_cache_prices"
    os.makedirs(CACHE_DIR, exist_ok=True)

    @staticmethod
    def _normalize_symbol(ticker: str) -> str:
        # Yahoo uses META instead of FB (kept for compatibility)
        if ticker.upper() == "FB":
            return "META"
        return ticker.upper()

    @classmethod
    def main(cls, tickers, start_date, end_date, freq="daily"):
        """
        Returns:
          dict[ticker] -> list of dicts with keys: Date (YYYY-MM-DD), Adj_Close, Volume

        freq:
          - 'daily'  : daily bars
          - 'weekly' : resampled to end-of-week (W-FRI) using last Adj Close and sum Volume
        """
        out = {}
        for t in tickers:
            sym = cls._normalize_symbol(t)
            cache_path = os.path.join(cls.CACHE_DIR, f"{t.upper()}_{start_date}_{end_date}_{freq}.csv")

            if os.path.exists(cache_path):
                print("Found in cache!!!")
                df = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
            else:
                df = yf.download(
                    sym,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False
                )
                if df is None or df.empty:
                    raise RuntimeError(f"No data downloaded for {t} (Yahoo symbol used: {sym})")

                # Keep only what we need
                price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
                vol = df["Volume"] if "Volume" in df.columns else 0

                df = pd.DataFrame({"Adj_Close": price, "Volume": vol}).dropna(subset=["Adj_Close"])
                df.index.name = "Date"

                if str(freq).lower().startswith("week"):
                    # End-of-week prices (Friday). Volume is summed across the week.
                    df_week = pd.DataFrame()
                    df_week["Adj_Close"] = df["Adj_Close"].resample("W-FRI").last()
                    df_week["Volume"] = df["Volume"].resample("W-FRI").sum()
                    df = df_week.dropna(subset=["Adj_Close"])

                df.to_csv(cache_path, index_label="Date")

            rows = []
            for dt, row in df.iterrows():
                rows.append({
                    "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "Adj_Close": float(row["Adj_Close"]),
                    "Volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0
                })

            out[t.upper()] = rows

        return out
