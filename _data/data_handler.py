import pandas as pd
import yfinance as yf

class data_handler:
    """Compatibility layer for older course notebooks.

    Usage:
        from data_handler import data_handler
        raw_data = data_handler.main(tickers, start_date, end_date)

    Output format matches the old repo:
        dict[ticker] -> list of dicts with keys 'Date', 'Adj_Close', 'Volume'
    """

    @staticmethod
    def _norm_ticker(t: str) -> str:
        # Yahoo Finance: FB is now META
        return "META" if t.upper() == "FB" else t.upper()

    @staticmethod
    def main(tickers, start_date, end_date):
        norm = [data_handler._norm_ticker(t) for t in tickers]

        df = yf.download(
            norm,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        out = {}
        for orig in tickers:
            sym = data_handler._norm_ticker(orig)

            # MultiIndex columns when multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                sub = df.xs(sym, axis=1, level=1, drop_level=True).copy()
            else:
                sub = df.copy()

            price_col = "Adj Close" if "Adj Close" in sub.columns else "Close"

            rows = []
            for dt, adj in sub[price_col].dropna().items():
                vol = sub.loc[dt, "Volume"] if "Volume" in sub.columns else 0
                rows.append({
                    "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "Adj_Close": float(adj),
                    "Volume": int(vol) if pd.notna(vol) else 0
                })

            out[orig] = rows

        return out

# Also provide module-level main for convenience
def main(tickers, start_date, end_date):
    return data_handler.main(tickers, start_date, end_date)
