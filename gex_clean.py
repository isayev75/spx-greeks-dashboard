import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL        = "^SPX"
N_EXPIRATIONS = 5
RISK_FREE_RATE = 0.045
STRIKE_RANGE  = 100
STRIKE_STEP   = 2.5
PLOT_WIDTH    = 850
PLOT_HEIGHT   = 4800

COLORSCALE = [
    [0.00, "#211340"],
    [0.20, "#EA00FF"],
    [0.35, "#FF00DD"],
    [0.50, "#D1C1AE"],
    [0.65, "#3552FC"],
    [0.80, "#101A6B"],
    [1.00, "#192040"],
]

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def bs_gamma(S, K, T, r, sigma):
    if min(S, K, T, sigma) <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S, K, T, r, sigma, option_type="call"):
    if min(S, K, T, sigma) <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    decay = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    carry = (-r * K * np.exp(-r * T) * norm.cdf(d2)  if option_type == "call"
             else r * K * np.exp(-r * T) * norm.cdf(-d2))
    return (decay + carry) / 365.0


def bs_vanna(S, K, T, r, sigma):
    if min(S, K, T, sigma) <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return (-norm.pdf(d1) * d2 / sigma) / 100.0


def bs_charm(S, K, T, r, sigma, option_type="call"):
    if min(S, K, T, sigma) <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    base = -norm.pdf(d1) * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    if option_type == "put":
        base -= r * np.exp(-r * T)
    return base / 365.0


# ---------------------------------------------------------------------------
# Exposure computation
# ---------------------------------------------------------------------------

def _time_to_expiry(expiration):
    return max((datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days / 365.0, 1 / 365.0)


def _iter_rows(df, spot, T, r):
    """Yield (strike, oi, iv) for valid rows."""
    for _, row in df.iterrows():
        strike = row.get("strike", 0)
        oi     = row.get("openInterest", 0)
        iv     = row.get("impliedVolatility", 0)
        if any(pd.isna(x) or x <= 0 for x in (strike, oi, iv)) or iv > 5:
            continue
        yield strike, oi, iv


def compute_exposure(calls, puts, spot, expiration):
    T = _time_to_expiry(expiration)
    r = RISK_FREE_RATE
    records = []

    for df, opt in ((calls, "call"), (puts, "put")):
        for strike, oi, iv in _iter_rows(df, spot, T, r):
            sign_gex   =  1 if opt == "call" else -1
            sign_vanna = -1 if opt == "call" else  1

            records.append({
                "strike": strike,
                "gex":   sign_gex   * oi * bs_gamma(spot, strike, T, r, iv) * spot**2 * 100,
                "tex":              -oi * bs_theta(spot, strike, T, r, iv, opt) * spot * 100,
                "vanna": sign_vanna * oi * bs_vanna(spot, strike, T, r, iv) * spot * 100,
                "charm":           -oi * bs_charm(spot, strike, T, r, iv, opt) * spot * 100,
            })

    if not records:
        return {k: pd.Series(dtype=float) for k in ("gex", "tex", "vanna", "charm")}

    df = pd.DataFrame(records).groupby("strike").sum()
    return {col: df[col] for col in ("gex", "tex", "vanna", "charm")}


def build_matrix(series_by_exp, spot):
    """Snap exposures onto a regular strike grid. Returns (z_matrix, strikes, exps)."""
    strike_min, strike_max = spot - STRIKE_RANGE, spot + STRIKE_RANGE

    all_strikes = {s for series in series_by_exp.values() for s in series.index}
    grid = sorted({
        round(s / STRIKE_STEP) * STRIKE_STEP
        for s in all_strikes
        if strike_min <= s <= strike_max
    })
    exps = list(series_by_exp.keys())

    z = np.zeros((len(grid), len(exps)))
    for col, exp in enumerate(exps):
        series = series_by_exp.get(exp, pd.Series(dtype=float))
        for row, strike in enumerate(grid):
            if strike in series.index:
                z[row, col] = series[strike]
            else:
                diffs = (series.index - strike).map(abs)
                if len(diffs) and diffs.min() <= STRIKE_STEP * 0.6:
                    z[row, col] = series.iloc[diffs.argmin()]

    return z, grid, exps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _fmt(val):
    if val == 0:    return "$0"
    mag = abs(val)
    if mag >= 1e9:  return f"${val/1e9:.2f}B"
    if mag >= 1e6:  return f"${val/1e6:.2f}M"
    if mag >= 1e3:  return f"${val/1e3:.1f}K"
    return f"${val:.0f}"


def _text_matrix(z):
    return np.vectorize(_fmt)(z)


def _spot_annotation(spot, strikes, row):
    nearest = min(strikes, key=lambda s: abs(s - spot))
    ref = "" if row == 1 else str(row)
    return dict(
        x=-0.5, y=nearest,
        text=f"<b>▶ {spot:,.0f}</b>",
        showarrow=False,
        font=dict(color="#00FFFF", size=9, family="monospace"),
        xref=f"x{ref}", yref=f"y{ref}",
        xanchor="right", yanchor="middle",
        bgcolor="rgba(0,0,0,0.85)",
        bordercolor="#00FFFF", borderwidth=1, borderpad=3,
    )


PANELS = [
    ("gex",   "Gamma Exposure (GEX)", 0.89),
    ("tex",   "Theta Exposure (TEX)", 0.63),
    ("vanna", "Vanna Exposure",       0.37),
    ("charm", "Charm Exposure (CEX)", 0.11),
]


def build_dashboard(spot, matrices, symbol):
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[f"<b>{symbol} — {label}</b>" for _, label, _ in PANELS],
        vertical_spacing=0.03,
        specs=[[{"type": "heatmap"}]] * 4,
    )

    axis_style = dict(showgrid=False, showline=True, linecolor="#333333", linewidth=1)
    annotations = []

    for row, (key, label, cb_y) in enumerate(PANELS, start=1):
        z, strikes, exps = matrices[key]

        fig.add_trace(go.Heatmap(
            z=z, x=exps, y=[float(s) for s in strikes],
            text=_text_matrix(z), texttemplate="%{text}",
            textfont={"size": 10, "color": "white", "family": "monospace"},
            colorscale=COLORSCALE, zmid=0, showscale=True,
            colorbar=dict(
                title=dict(text=f"<b>{label}</b>",
                           font=dict(color="white", size=11, family="monospace")),
                tickfont=dict(color="white", size=9, family="monospace"),
                len=0.22, thickness=20, x=1.02, y=cb_y,
            ),
            hovertemplate=f"Strike %{{y}} | Exp %{{x}} | {label} %{{z:+.1f}}M<extra></extra>",
        ), row=row, col=1)

        annotations.append(_spot_annotation(spot, strikes, row))

        fig.update_xaxes(**axis_style, type="category",
                         tickfont=dict(color="white", size=10, family="monospace"),
                         title=dict(text="Expiration Date",
                                    font=dict(color="#666666", size=11, family="monospace")),
                         row=row, col=1)
        fig.update_yaxes(**axis_style, dtick=5,
                         tickfont=dict(color="white", size=8, family="monospace"),
                         title=dict(text="Strike",
                                    font=dict(color="#666666", size=11, family="monospace")),
                         row=row, col=1)

    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="white", family="monospace"),
        title=dict(
            text=(f"<b>{symbol} — Greek Exposures</b><br>"
                  f"<span style='font-size:12px'>"
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M')} — Spot: ${spot:,.2f}</span>"),
            x=0.5, xanchor="center",
            font=dict(size=16, color="white", family="monospace"),
        ),
        height=PLOT_HEIGHT, width=PLOT_WIDTH,
        showlegend=False,
        margin=dict(l=50, r=120, t=100, b=40),
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ticker = yf.Ticker(SYMBOL)
    spot   = ticker.history(period="2d")["Close"].iloc[-1]

    raw = {k: {} for k in ("gex", "tex", "vanna", "charm")}

    for exp in ticker.options[:N_EXPIRATIONS]:
        try:
            chain = ticker.option_chain(exp)
            result = compute_exposure(chain.calls, chain.puts, spot, exp)
            for k, series in result.items():
                if not series.empty:
                    raw[k][exp] = series
        except Exception as e:
            print(f"[warn] {exp}: {e}")

    matrices = {k: build_matrix(raw[k], spot) for k in raw}
    fig = build_dashboard(spot, matrices, SYMBOL)
    fig.show(renderer="browser")


if __name__ == "__main__":
    main()
