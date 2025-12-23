#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import ConfusionMatrixDisplay, f1_score


@dataclass(frozen=True)
class Columns:
    datetime: str = "datetime"
    country: str = "country"
    price: str = "price"
    nuclear: str = "nuclear"

    pumped_gen: str = "_hydro_pumped_storage_actual_aggregated_"
    pumped_consume: str = "_hydro_pumped_storage_actual_consumption_"

    solar: str = "_solar_actual_aggregated_"
    wind_onshore: str = "_wind_onshore_actual_aggregated_"
    biomass: str = "_biomass_actual_aggregated_"
    other_renewable: str = "_other_renewable_actual_aggregated_"
    hydro_run: str = "_hydro_run_of_river_and_poundage_actual_aggregated_"
    hydro_reservoir: str = "_hydro_water_reservoir_actual_aggregated_"

    gas: str = "_fossil_gas_actual_aggregated_"
    hard_coal: str = "_fossil_hard_coal_actual_aggregated_"
    lignite: str = "_fossil_brown_coal_lignite_actual_aggregated_"
    oil: str = "_fossil_oil_actual_aggregated_"
    coal_gas: str = "_fossil_coal_derived_gas_actual_aggregated_"
    other: str = "_other_actual_aggregated_"


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if len(df.columns) > 0 and isinstance(df.columns[0], str) and df.columns[0].startswith("\ufeff"):
        df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("\ufeff")})
    return df


def _add_time_columns(df: pd.DataFrame, cols: Columns, market_tz: str) -> pd.DataFrame:
    out = df.copy()
    out["datetime_utc"] = pd.to_datetime(out[cols.datetime], utc=True, errors="raise")
    out["datetime_local"] = out["datetime_utc"].dt.tz_convert(market_tz)
    out = out.sort_values("datetime_utc").reset_index(drop=True)

    dow = out["datetime_local"].dt.dayofweek
    out["week_start_local"] = (out["datetime_local"] - pd.to_timedelta(dow, unit="D")).dt.normalize()
    out["date_local"] = out["datetime_local"].dt.date
    out["hour_local"] = out["datetime_local"].dt.hour
    out["week_of_year"] = out["datetime_local"].dt.isocalendar().week.astype("int64")
    out["month"] = out["datetime_local"].dt.month.astype("int64")
    return out


def _rolling_p99(series: pd.Series, window_hours: int) -> pd.Series:
    if window_hours <= 0:
        raise ValueError("window_hours must be > 0")
    min_periods = max(1, window_hours // 2)
    return series.rolling(window_hours, min_periods=min_periods).quantile(0.99).shift(1)


def _add_hourly_features(df: pd.DataFrame, cols: Columns, spike_window_weeks: int, min_mw: float) -> pd.DataFrame:
    out = df.copy()

    out["price"] = out[cols.price].astype("float64")

    pumped_gen = out[cols.pumped_gen].fillna(0.0).astype("float64")
    pumped_consume = out[cols.pumped_consume].fillna(0.0).astype("float64")
    out["pumped_gen_mw"] = pumped_gen
    out["pumped_consume_mw"] = pumped_consume
    out["pumped_net_mw"] = pumped_gen - pumped_consume

    gen_on = pumped_gen > min_mw
    pump_on = pumped_consume > min_mw
    out["pumped_mode"] = "idle"
    out.loc[gen_on & ~pump_on, "pumped_mode"] = "generate"
    out.loc[pump_on & ~gen_on, "pumped_mode"] = "pump"
    out.loc[gen_on & pump_on, "pumped_mode"] = "both"

    out["neg_flag"] = out["price"] < 0
    out["spike_threshold"] = _rolling_p99(out["price"], window_hours=spike_window_weeks * 7 * 24)
    out["spike_flag"] = out["price"] > out["spike_threshold"]

    # Generation mix (PL aggregated set). Missing columns are treated as 0.
    def s(name: str) -> pd.Series:
        if name in out.columns:
            return out[name].fillna(0.0).astype("float64")
        return pd.Series(0.0, index=out.index, dtype="float64")

    renewable_gen = s(cols.solar) + s(cols.wind_onshore) + s(cols.hydro_run) + s(cols.hydro_reservoir) + s(cols.biomass) + s(cols.other_renewable)
    fossil_gen = s(cols.gas) + s(cols.hard_coal) + s(cols.lignite) + s(cols.oil) + s(cols.coal_gas)
    nuclear_gen = s(cols.nuclear)
    other_gen = s(cols.other)
    pumped_gen_for_total = s(cols.pumped_gen)

    out["renewable_gen_mw"] = renewable_gen
    out["fossil_gen_mw"] = fossil_gen
    out["nuclear_gen_mw"] = nuclear_gen
    out["other_gen_mw"] = other_gen
    out["total_gen_mw"] = renewable_gen + fossil_gen + nuclear_gen + other_gen + pumped_gen_for_total

    denom = out["total_gen_mw"].where(out["total_gen_mw"] > 0)
    out["renewable_share"] = out["renewable_gen_mw"] / denom
    out["gas_share"] = s(cols.gas) / denom
    out["coal_share"] = (s(cols.hard_coal) + s(cols.lignite)) / denom
    out["nuclear_share"] = out["nuclear_gen_mw"] / denom

    out["wind_ramp"] = s(cols.wind_onshore).diff().abs()
    out["solar_ramp"] = s(cols.solar).diff().abs()
    out["renewable_ramp"] = out["renewable_gen_mw"].diff().abs()

    return out


def _weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("week_start_local", sort=True)

    def q(x: pd.Series, quantile: float) -> float:
        x = x.dropna().astype("float64")
        if x.empty:
            return float("nan")
        return float(x.quantile(quantile))

    price = df["price"]
    weekly = pd.DataFrame(
        {
            "n_hours": g.size(),
            "price_mean": g["price"].mean(),
            "price_std": g["price"].std(),
            "price_p10": g["price"].apply(lambda s: q(s, 0.10)),
            "price_p50": g["price"].apply(lambda s: q(s, 0.50)),
            "price_p90": g["price"].apply(lambda s: q(s, 0.90)),
            "neg_hours": g["neg_flag"].sum(),
            "spike_hours": g["spike_flag"].sum(),
            "peak_price": g["price"].max(),
            "pumped_gen_hours": g["pumped_gen_mw"].apply(lambda s: int((s > 0).sum())),
            "pumped_pump_hours": g["pumped_consume_mw"].apply(lambda s: int((s > 0).sum())),
            "pumped_net_mean": g["pumped_net_mw"].mean(),
            "pumped_net_p10": g["pumped_net_mw"].apply(lambda s: q(s, 0.10)),
            "pumped_net_p90": g["pumped_net_mw"].apply(lambda s: q(s, 0.90)),
            "renewable_share_mean": g["renewable_share"].mean(),
            "gas_share_mean": g["gas_share"].mean(),
            "coal_share_mean": g["coal_share"].mean(),
            "nuclear_share_mean": g["nuclear_share"].mean(),
            "wind_ramp_p90": g["wind_ramp"].apply(lambda s: q(s, 0.90)),
            "solar_ramp_p90": g["solar_ramp"].apply(lambda s: q(s, 0.90)),
            "renewable_ramp_p90": g["renewable_ramp"].apply(lambda s: q(s, 0.90)),
        }
    ).reset_index()

    # Peak hour (local) for reporting.
    idx = df.loc[price.notna()].groupby("week_start_local")["price"].idxmax()
    peak_dt = df.loc[idx, ["week_start_local", "datetime_local"]].set_index("week_start_local")["datetime_local"]
    weekly["peak_hour_local"] = weekly["week_start_local"].map(peak_dt)

    # Seasonal features (week_of_year/month) based on week_start.
    wsl = pd.to_datetime(weekly["week_start_local"])
    weekly["week_of_year"] = wsl.dt.isocalendar().week.astype("int64")
    weekly["month"] = wsl.dt.month.astype("int64")

    weekly["vol_range"] = weekly["price_p90"] - weekly["price_p10"]
    return weekly


def _prepare_supervised(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.sort_values("week_start_local").reset_index(drop=True).copy()
    w["target_week_start_local"] = w["week_start_local"].shift(-1)
    w["next_week_price_mean"] = w["price_mean"].shift(-1)
    w["delta"] = w["next_week_price_mean"] - w["price_mean"]
    return w.iloc[:-1].reset_index(drop=True)


def _direction_label(delta: float, theta: float) -> str:
    if not np.isfinite(delta) or not np.isfinite(theta) or theta <= 0:
        return "Flat"
    if delta > theta:
        return "Up"
    if delta < -theta:
        return "Down"
    return "Flat"


def _walk_forward_backtest(
    supervised: pd.DataFrame,
    feature_cols: list[str],
    train_window_weeks: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    x_all = supervised[feature_cols]
    y_all = supervised["next_week_price_mean"].astype("float64")

    for i in range(train_window_weeks, len(supervised)):
        train = supervised.iloc[i - train_window_weeks : i]
        test = supervised.iloc[i : i + 1]

        x_train = train[feature_cols]
        y_train = train["next_week_price_mean"].astype("float64")
        x_test = test[feature_cols]

        # Direction labels for training based on historical delta std.
        delta_hist = train["delta"].astype("float64")
        theta = float(0.25 * delta_hist.std(ddof=0))

        y_dir_train = delta_hist.apply(lambda d: _direction_label(float(d), theta))
        clf = HistGradientBoostingClassifier(random_state=42)
        clf.fit(x_train, y_dir_train)

        # Quantile regressors for interval.
        q_models = {
            0.1: HistGradientBoostingRegressor(loss="quantile", quantile=0.1, random_state=42),
            0.5: HistGradientBoostingRegressor(loss="quantile", quantile=0.5, random_state=42),
            0.9: HistGradientBoostingRegressor(loss="quantile", quantile=0.9, random_state=42),
        }
        for m in q_models.values():
            m.fit(x_train, y_train)

        q10 = float(q_models[0.1].predict(x_test)[0])
        q50 = float(q_models[0.5].predict(x_test)[0])
        q90 = float(q_models[0.9].predict(x_test)[0])

        delta_hat = q50 - float(test["price_mean"].iloc[0])
        dir_hat = _direction_label(delta_hat, theta)

        # Actuals (for this row, the target is week_start_local + 1 week).
        actual = float(test["next_week_price_mean"].iloc[0])
        delta_actual = float(test["delta"].iloc[0])
        dir_actual = _direction_label(delta_actual, theta)

        rows.append(
            {
                "week_start_local": test["week_start_local"].iloc[0],
                "target_week_start_local": test["target_week_start_local"].iloc[0],
                "price_mean": float(test["price_mean"].iloc[0]),
                "next_week_price_mean": actual,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "theta": theta,
                "dir_pred": dir_hat,
                "dir_true": dir_actual,
                "covered_10_90": bool((actual >= q10) and (actual <= q90)),
            }
        )

    return pd.DataFrame(rows)


def _plot_interval_backtest(bt: pd.DataFrame, out_path: Path) -> None:
    if bt.empty:
        return
    bt = bt.loc[bt["target_week_start_local"].notna()].copy()
    x = pd.to_datetime(bt["target_week_start_local"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(x, bt["q10"], bt["q90"], color="tab:blue", alpha=0.2, label="pred [P10,P90]")
    ax.plot(x, bt["q50"], color="tab:blue", lw=1.5, label="pred P50")
    ax.plot(x, bt["next_week_price_mean"], color="black", lw=1.2, label="actual")
    ax.set_title("Walk-forward: next-week mean price interval")
    ax.set_ylabel("€/MWh")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_coverage(bt: pd.DataFrame, out_path: Path) -> None:
    if bt.empty:
        return
    bt = bt.loc[bt["target_week_start_local"].notna()].copy()
    cov = bt["covered_10_90"].astype("int64")
    cum = cov.cumsum() / (np.arange(len(cov)) + 1)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(pd.to_datetime(bt["target_week_start_local"]), cum, lw=1.5, color="tab:green")
    ax.axhline(0.8, color="black", lw=0.8, alpha=0.4)
    ax.set_ylim(0, 1)
    ax.set_title("Cumulative coverage of [P10,P90]")
    ax.set_ylabel("coverage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_confusion(bt: pd.DataFrame, out_path: Path) -> float:
    if bt.empty:
        return float("nan")
    labels = ["Down", "Flat", "Up"]
    y_true = bt["dir_true"].astype(str)
    y_pred = bt["dir_pred"].astype(str)
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro"))
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, colorbar=False, normalize=None)
    ax.set_title(f"Direction (macro-F1={macro:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return macro


def _plot_pred_error(bt: pd.DataFrame, out_path: Path) -> None:
    if bt.empty:
        return
    bt = bt.loc[bt["target_week_start_local"].notna()].copy()
    x = pd.to_datetime(bt["target_week_start_local"])
    err = (bt["next_week_price_mean"] - bt["q50"]).astype("float64")
    miss = ~bt["covered_10_90"].astype(bool)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(x, err, lw=1.2, color="tab:blue", label="actual - pred(P50)")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    if miss.any():
        ax.scatter(x[miss], err[miss], s=25, color="tab:red", label="interval miss")
    ax.set_title("Backtest residuals (next-week mean)")
    ax.set_ylabel("€/MWh")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_final_report_template_md(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = """# Final report template (Markdown)

This file is a **format/style template** for the comprehensive PDF report.
The comprehensive PDF report is generated by `python generate_report.py` and should follow
the same section structure.

## 1. Overview

- Market: {country}
- Timezone: {market_tz}
- Data coverage: {data_start_local} → {data_end_local}
- Report generated at: {generated_at_local}

## 2. Key takeaways (auto summary)

- Latest week price: mean={latest_price_mean}, P10={latest_price_p10}, P90={latest_price_p90}
- Risk hours: neg={latest_neg_hours}, spike={latest_spike_hours}
- Pumped storage: pump_hours={latest_pump_hours}, gen_hours={latest_gen_hours}, net_mean={latest_pumped_net_mean}
- Backtest: macro_f1={macro_f1}, coverage_10_90={coverage_10_90}

## 3. Latest week deep dive

Include:
- Hourly price (this week vs previous week)
- Price distribution for the latest week
- Generation mix (stacked)
- Pumped-net time series

## 4. Backtest summary

Include:
- Interval backtest (P10–P90 band + actual)
- Coverage over time
- Direction confusion matrix
- Residuals over time (actual - P50), marking interval misses

## 5. Weekly review (narrative)

For the recent N weeks (configurable), include:
- One-line conclusion (interval coverage + direction hit/miss)
- Possible drivers (risk hours, ramps, pumped behavior changes)
- Risk note + next-week watchlist
"""
    out_path.write_text(template, encoding="utf-8")


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(title, fontsize=18, y=0.97)
    ax = fig.add_subplot(111)
    ax.axis("off")

    y = 0.92
    for line in lines:
        for part in wrap(line, width=95, break_long_words=False, break_on_hyphens=False):
            ax.text(0.03, y, part, fontsize=11, va="top", ha="left")
            y -= 0.035
            if y < 0.06:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(11.7, 8.3))
                fig.suptitle(title, fontsize=18, y=0.97)
                ax = fig.add_subplot(111)
                ax.axis("off")
                y = 0.92
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_image_grid_page(pdf: PdfPages, title: str, items: list[tuple[str, Path]]) -> None:
    existing = [(label, path) for label, path in items if path.exists()]
    if not existing:
        return

    n = len(existing)
    rows = 2 if n > 2 else 1
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(11.7, 8.3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for (label, path), ax in zip(existing, axes.ravel(), strict=False):
        img = imread(path)
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=18, y=0.98)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _write_final_report_pdf(
    out_path: Path,
    *,
    df: pd.DataFrame,
    weekly: pd.DataFrame,
    bt: pd.DataFrame,
    macro_f1: float,
    country: str,
    market_tz: str,
    spike_window_weeks: int,
    train_window_weeks: int,
    min_week_hours: int,
    review_weeks: int,
    backtest_dir: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    latest = _latest_full_week_start(weekly)
    w = weekly.set_index("week_start_local").sort_index()
    latest_row = w.loc[latest]
    prev = w.index[w.index.get_loc(latest) - 1] if w.index.get_loc(latest) > 0 else None
    prev_row = w.loc[prev] if prev is not None else None
    delta = float(latest_row["price_mean"] - prev_row["price_mean"]) if prev_row is not None else float("nan")

    start_local = df["datetime_local"].min()
    end_local = df["datetime_local"].max()
    generated_at = pd.Timestamp.now(tz=market_tz)

    coverage = float(bt["covered_10_90"].mean()) if len(bt) else float("nan")
    mae = float((bt["next_week_price_mean"] - bt["q50"]).abs().mean()) if len(bt) else float("nan")
    miss_rate = float((~bt["covered_10_90"].astype(bool)).mean()) if len(bt) else float("nan")

    sub_latest = df.loc[df["week_start_local"] == latest].copy()
    p_pump = sub_latest.loc[sub_latest["pumped_mode"] == "pump", "price"].mean()
    p_gen = sub_latest.loc[sub_latest["pumped_mode"] == "generate", "price"].mean()
    spread = float(p_gen - p_pump) if pd.notna(p_pump) and pd.notna(p_gen) else float("nan")

    cover_lines = [
        f"- Market: {country}",
        f"- Timezone: {market_tz}",
        f"- Data coverage: {start_local} → {end_local}",
        f"- Generated at: {generated_at}",
        "",
        "Latest week snapshot:",
        f"- Week start (local): {latest}",
        f"- Price mean: {latest_row['price_mean']:.2f} €/MWh (WoW {delta:+.2f})",
        f"- Price P10–P90: {latest_row['price_p10']:.2f} → {latest_row['price_p90']:.2f} €/MWh",
        f"- Risk hours: neg={int(latest_row['neg_hours'])}, spike={int(latest_row['spike_hours'])}",
        f"- Pumped: pump_hours={int(latest_row['pumped_pump_hours'])}, gen_hours={int(latest_row['pumped_gen_hours'])}, net_mean={latest_row['pumped_net_mean']:.2f} MW",
        f"- Pumped spread proxy (generate - pump): {spread:.2f} €/MWh",
        "",
        "Backtest snapshot (walk-forward, weekly):",
        f"- Train window: {train_window_weeks} weeks | min_week_hours={min_week_hours}",
        f"- Spike window: {spike_window_weeks} weeks",
        f"- Direction macro-F1: {macro_f1:.3f}",
        f"- Interval coverage (P10–P90): {coverage:.3f} | miss rate≈{miss_rate:.0%}",
        f"- MAE(|actual - P50|): {mae:.2f} €/MWh",
    ]

    with PdfPages(out_path) as pdf:
        _add_text_page(pdf, title="Final report", lines=cover_lines)

        method_lines = [
            "Pipeline:",
            "1) Parse datetime to UTC and convert to market timezone.",
            "2) Build hourly features: risk flags (neg/spike), pumped-net and modes, mix shares, ramps.",
            "3) Aggregate to weekly table (Monday 00:00 local): price stats + risk hours + pumped behavior + mix.",
            "4) Supervised targets: next_week_price_mean and delta.",
            "5) Walk-forward backtest (weekly):",
            "   - Direction (Up/Down/Flat) with adaptive threshold theta=0.25*std(delta_history).",
            "   - Interval (P10/P50/P90) via quantile regression (HistGradientBoostingRegressor, loss=quantile).",
            "6) Outputs: weekly report PDF, review narrative, backtest figures, and this final report PDF.",
        ]
        _add_text_page(pdf, title="Methods and pipeline", lines=method_lines)

        _plot_week_report_page1(df=df, weekly=weekly, out_pdf=pdf)
        _plot_week_report_page2(weekly=weekly, bt=bt, macro_f1=macro_f1, out_pdf=pdf)

        _plot_image_grid_page(
            pdf,
            title="Backtest diagnostics (figures)",
            items=[
                ("Direction confusion", backtest_dir / "direction_confusion.png"),
                ("Residuals over time", backtest_dir / "pred_error_over_time.png"),
                ("Coverage over time", backtest_dir / "coverage_over_time.png"),
                ("Interval backtest", backtest_dir / "interval_backtest.png"),
            ],
        )

        narrative_lines = [
            f"- A narrative review is generated at: reports/weekly_review.md (recent {review_weeks} weeks).",
            "- Use it as a structured log: conclusion → possible drivers → risk note → next-week watchlist.",
            "",
            "Notes:",
            "- Interval coverage is sensitive to tail events and regime changes.",
            "- If miss rate is high, increase train window, enrich features, or adjust thresholds.",
        ]
        _add_text_page(pdf, title="Review notes", lines=narrative_lines)


def _write_weekly_review(
    weekly: pd.DataFrame,
    bt: pd.DataFrame,
    macro_f1: float,
    out_path: Path,
    lookback: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bt.empty:
        out_path.write_text("# Weekly review\n\nNo backtest rows.\n", encoding="utf-8")
        return

    w = weekly.set_index("week_start_local").sort_index()
    bt = bt.loc[bt["target_week_start_local"].notna()].copy()
    bt["interval_width"] = (bt["q90"] - bt["q10"]).astype("float64")
    bt["abs_err"] = (bt["next_week_price_mean"] - bt["q50"]).abs().astype("float64")
    bt["miss_side"] = np.where(
        bt["next_week_price_mean"] > bt["q90"],
        "above",
        np.where(bt["next_week_price_mean"] < bt["q10"], "below", "covered"),
    )

    ctx_cols = [
        "price_mean",
        "price_p10",
        "price_p90",
        "vol_range",
        "neg_hours",
        "spike_hours",
        "pumped_gen_hours",
        "pumped_pump_hours",
        "pumped_net_mean",
        "renewable_share_mean",
        "gas_share_mean",
        "coal_share_mean",
        "nuclear_share_mean",
        "wind_ramp_p90",
        "solar_ramp_p90",
        "renewable_ramp_p90",
    ]
    ctx_cols = [c for c in ctx_cols if c in w.columns]
    ctx = w[ctx_cols]

    bt["target_week_start_local"] = pd.to_datetime(bt["target_week_start_local"])
    bt["week_start_local"] = pd.to_datetime(bt["week_start_local"])
    bt = bt.join(ctx, on="target_week_start_local", rsuffix="_target")
    bt = bt.join(ctx.add_suffix("_curr"), on="week_start_local")

    spike_p90 = float(ctx["spike_hours"].quantile(0.90)) if "spike_hours" in ctx.columns else float("nan")
    neg_p90 = float(ctx["neg_hours"].quantile(0.90)) if "neg_hours" in ctx.columns else float("nan")
    ramp_p90 = float(ctx["renewable_ramp_p90"].quantile(0.90)) if "renewable_ramp_p90" in ctx.columns else float("nan")

    def fmt(x: object, digits: int = 2) -> str:
        if x is None:
            return "NA"
        try:
            if isinstance(x, (float, np.floating)) and not np.isfinite(x):
                return "NA"
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)

    n = len(bt)
    coverage = float(bt["covered_10_90"].mean())
    mae = float(bt["abs_err"].mean())
    avg_w = float(bt["interval_width"].mean())

    worst = bt.sort_values("abs_err", ascending=False).head(5)
    big_err_th = float(bt["abs_err"].quantile(0.80))
    miss_rate = float((~bt["covered_10_90"].astype(bool)).mean())

    lines: list[str] = []
    lines.append("# Weekly review（自动复盘）\n\n")
    lines.append("## 总览（滚动回测）\n\n")
    lines.append(f"- 样本数：{n}\n")
    lines.append(f"- 方向预测 Macro-F1：{macro_f1:.3f}\n")
    lines.append(f"- 区间覆盖率（P10–P90）：{coverage:.3f}\n")
    lines.append(f"- 平均区间宽度（q90-q10）：{avg_w:.2f} €/MWh\n")
    lines.append(f"- MAE（|actual - q50|）：{mae:.2f} €/MWh\n\n")
    lines.append("## 交易口径解读（怎么讲）\n\n")
    lines.append("- 方向预测：用于“下周偏多/偏空/震荡”的仓位倾向，不追求逐小时点位。\n")
    lines.append("- 区间预测（P10–P90）：用于风险暴露管理；覆盖率偏低通常意味着“尾部事件/制度性缺口/极端天气/机组故障”等非平稳冲击。\n")
    lines.append(f"- 当前回测 miss 率≈{miss_rate:.0%}，重点复盘“未覆盖周 + 大偏差周”。\n\n")

    lines.append("## 最大误差周（Top 5）\n\n")
    for _, r in worst.iterrows():
        t = r["target_week_start_local"]
        lines.append(
            f"- {t}: actual={fmt(r['next_week_price_mean'])}, pred=[{fmt(r['q10'])},{fmt(r['q50'])},{fmt(r['q90'])}], miss={r['miss_side']}, abs_err={fmt(r['abs_err'])}\n"
        )
    lines.append("\n")

    lookback = max(1, int(lookback))
    lines.append(f"## 近 {lookback} 周逐周复盘（预测下周均价）\n\n")
    recent = bt.sort_values("target_week_start_local").tail(lookback)
    for _, r in recent.iterrows():
        t = r["target_week_start_local"]
        covered = bool(r["covered_10_90"])
        is_big = bool(pd.notna(r["abs_err"]) and float(r["abs_err"]) >= big_err_th)
        head = f"### 目标周：{t}" + ("（重点复盘）" if (not covered or is_big) else "")
        lines.append(head + "\n\n")

        actual = float(r["next_week_price_mean"])
        p50 = float(r["q50"])
        err = float(actual - p50)
        direction_hit = (str(r["dir_pred"]) == str(r["dir_true"]))

        # 一句话结论（交易口径）
        cov_str = "覆盖" if covered else f"未覆盖（{r['miss_side']}）"
        dir_str = "命中" if direction_hit else "未命中"
        lines.append(
            f"- 结论：区间{cov_str}，方向{dir_str}；下周均价实际={fmt(actual)} vs 预测P50={fmt(p50)}（误差 {fmt(err)} €/MWh）。\n"
        )
        lines.append(
            f"- 预测：[{fmt(r['q10'])}, {fmt(r['q90'])}]（P50={fmt(r['q50'])}）｜实际：{fmt(r['next_week_price_mean'])}｜θ={fmt(r['theta'], 3)}｜pred={r['dir_pred']} / true={r['dir_true']}\n"
        )

        ctx_bullets: list[str] = []
        if "neg_hours" in r and pd.notna(r["neg_hours"]):
            ctx_bullets.append(f"负价小时={int(r['neg_hours'])}")
        if "spike_hours" in r and pd.notna(r["spike_hours"]):
            ctx_bullets.append(f"spike小时={int(r['spike_hours'])}")
        if "pumped_pump_hours" in r and pd.notna(r["pumped_pump_hours"]):
            ctx_bullets.append(f"抽蓄吸纳小时={int(r['pumped_pump_hours'])}")
        if "pumped_gen_hours" in r and pd.notna(r["pumped_gen_hours"]):
            ctx_bullets.append(f"抽蓄发电小时={int(r['pumped_gen_hours'])}")
        if "pumped_net_mean" in r and pd.notna(r["pumped_net_mean"]):
            ctx_bullets.append(f"抽蓄净出力均值={fmt(r['pumped_net_mean'])} MW")
        if "vol_range" in r and pd.notna(r["vol_range"]):
            ctx_bullets.append(f"波动区间(P90-P10)={fmt(r['vol_range'])} €/MWh")
        if ctx_bullets:
            lines.append("- 事实周画像（发生了什么）： " + "；".join(ctx_bullets) + "\n")

        # 归因与风险提示（模板化）
        drivers: list[str] = []
        risks: list[str] = []

        if "spike_hours" in r and pd.notna(r["spike_hours"]) and np.isfinite(spike_p90) and float(r["spike_hours"]) >= spike_p90:
            drivers.append("尖峰小时放大（系统紧平衡/边际成本抬升）")
            risks.append("尖峰尾部风险偏高：考虑收紧上沿/提高对冲比例")
        if "neg_hours" in r and pd.notna(r["neg_hours"]) and np.isfinite(neg_p90) and float(r["neg_hours"]) >= neg_p90:
            drivers.append("负价小时增多（可再生过剩/需求偏弱）")
            risks.append("下行尾部风险偏高：关注负价扩散与跨日延续性")
        if "renewable_ramp_p90" in r and pd.notna(r["renewable_ramp_p90"]) and np.isfinite(ramp_p90) and float(r["renewable_ramp_p90"]) >= ramp_p90:
            drivers.append("可再生爬坡压力加大（调节需求上升，波动放大）")
            risks.append("波动风险偏高：区间应更宽或加强风险缓冲")

        # 抽蓄“行为是否反常”（高价吸纳/低价发电）
        if "price_p10" in r and "price_p90" in r and pd.notna(r["price_p10"]) and pd.notna(r["price_p90"]):
            # Use realized target-week price percentiles (week-level) as coarse proxy.
            pass

        if "pumped_pump_hours_curr" in r and "pumped_pump_hours" in r and pd.notna(r["pumped_pump_hours_curr"]) and pd.notna(r["pumped_pump_hours"]):
            d = int(r["pumped_pump_hours"]) - int(r["pumped_pump_hours_curr"])
            if abs(d) >= 24:
                drivers.append(f"抽蓄吸纳小时显著切换（环比 {d:+d}h）")
        if "pumped_gen_hours_curr" in r and "pumped_gen_hours" in r and pd.notna(r["pumped_gen_hours_curr"]) and pd.notna(r["pumped_gen_hours"]):
            d = int(r["pumped_gen_hours"]) - int(r["pumped_gen_hours_curr"])
            if abs(d) >= 24:
                drivers.append(f"抽蓄发电小时显著切换（环比 {d:+d}h）")

        if drivers:
            lines.append("- 可能驱动（交易语言）： " + "；".join(drivers) + "\n")

        if (not covered) or is_big:
            # 强化“复盘—动作”
            if not risks:
                risks.append("出现尾部偏差：下周建议以区间/风险预算为主，而非点预测驱动")
            lines.append("- 风险提示： " + "；".join(risks) + "\n")

        # 下周关注点（周报可直接引用的要点）
        watch: list[str] = []
        if "spike_hours" in r and pd.notna(r["spike_hours"]) and int(r["spike_hours"]) > 0:
            watch.append("关注高峰时段的尖峰复现（晚高峰/早高峰）")
        if "neg_hours" in r and pd.notna(r["neg_hours"]) and int(r["neg_hours"]) > 0:
            watch.append("关注负价是否跨日持续、是否从周末扩散到工作日")
        if "pumped_net_mean" in r and pd.notna(r["pumped_net_mean"]):
            watch.append("关注抽蓄净出力是否从吸纳主导转向发电主导（套利窗口变化）")
        if watch:
            lines.append("- 下周关注点（可直接放周报）： " + "；".join(watch) + "\n")
        lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")

def _latest_full_week_start(weekly: pd.DataFrame) -> pd.Timestamp:
    candidates = weekly.loc[weekly["n_hours"] >= 160, "week_start_local"]
    if not candidates.empty:
        return candidates.iloc[-1]
    return weekly["week_start_local"].iloc[-1]


def _plot_week_report_page1(df: pd.DataFrame, weekly: pd.DataFrame, out_pdf: PdfPages) -> None:
    latest = _latest_full_week_start(weekly)
    w = weekly.set_index("week_start_local").sort_index()
    prev = w.index[w.index.get_loc(latest) - 1] if w.index.get_loc(latest) > 0 else None

    sub_latest = df.loc[df["week_start_local"] == latest].copy()
    sub_prev = df.loc[df["week_start_local"] == prev].copy() if prev is not None else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))  # A4 landscape-ish

    ax = axes[0, 0]
    ax.plot(sub_latest["datetime_local"], sub_latest["price"], lw=1.0, label="this week")
    if not sub_prev.empty:
        ax.plot(sub_prev["datetime_local"], sub_prev["price"], lw=1.0, alpha=0.7, label="prev week")
    ax.set_title("Hourly price: this vs prev week")
    ax.set_ylabel("€/MWh")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[0, 1]
    ax.hist(sub_latest["price"].dropna(), bins=40, color="tab:blue", alpha=0.8)
    p10 = float(sub_latest["price"].quantile(0.10))
    p90 = float(sub_latest["price"].quantile(0.90))
    ax.axvline(p10, color="black", lw=0.8, alpha=0.7)
    ax.axvline(p90, color="black", lw=0.8, alpha=0.7)
    ax.set_title("This week price distribution (P10/P90)")
    ax.set_xlabel("€/MWh")

    ax = axes[1, 0]
    # Generation mix stack (renewable/fossil/nuclear/other/pumped)
    t = sub_latest["datetime_local"]
    mix = np.vstack(
        [
            sub_latest["renewable_gen_mw"].to_numpy(),
            sub_latest["fossil_gen_mw"].to_numpy(),
            sub_latest["nuclear_gen_mw"].to_numpy(),
            sub_latest["other_gen_mw"].to_numpy(),
            sub_latest["pumped_gen_mw"].to_numpy(),
        ]
    )
    labels = ["renewable", "fossil", "nuclear", "other", "pumped_gen"]
    ax.stackplot(t, mix, labels=labels, alpha=0.85)
    ax.set_title("This week generation mix (MW)")
    ax.set_ylabel("MW")
    ax.legend(loc="upper left", ncols=3, frameon=False)

    ax = axes[1, 1]
    ax.plot(sub_latest["datetime_local"], sub_latest["pumped_net_mw"], lw=1.0, color="tab:orange")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("This week pumped_net (MW) [gen-consume]")
    ax.set_ylabel("MW")

    fig.suptitle(f"Weekly market report (PL) — week_start={latest}")
    fig.tight_layout()
    out_pdf.savefig(fig)
    plt.close(fig)


def _plot_week_report_page2(weekly: pd.DataFrame, bt: pd.DataFrame, macro_f1: float, out_pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))

    ax = axes[0, 0]
    x = weekly["week_start_local"]
    ax.plot(x, weekly["price_mean"], lw=1.2, color="tab:blue")
    ax.fill_between(x, weekly["price_p10"], weekly["price_p90"], color="tab:blue", alpha=0.15)
    ax.set_title("Weekly price mean and P10–P90 (realized)")
    ax.set_ylabel("€/MWh")

    ax = axes[0, 1]
    ax.bar(x, weekly["neg_hours"], color="tab:purple", alpha=0.8, label="neg_hours")
    ax.bar(x, weekly["spike_hours"], bottom=weekly["neg_hours"], color="tab:red", alpha=0.6, label="spike_hours")
    ax.set_title("Weekly risk hours (negative + spike)")
    ax.set_ylabel("hours")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1, 0]
    if not bt.empty:
        xt = pd.to_datetime(bt["target_week_start_local"])
        ax.fill_between(xt, bt["q10"], bt["q90"], color="tab:green", alpha=0.2, label="pred [P10,P90]")
        ax.plot(xt, bt["next_week_price_mean"], color="black", lw=1.0, label="actual")
        ax.set_title("Backtest: interval vs actual (next-week mean)")
        ax.set_ylabel("€/MWh")
        ax.legend(loc="upper left", frameon=False)
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    if not bt.empty:
        cov = bt["covered_10_90"].astype("int64")
        cum = cov.cumsum() / (np.arange(len(cov)) + 1)
        ax.plot(pd.to_datetime(bt["target_week_start_local"]), cum, lw=1.5, color="tab:green")
        ax.axhline(0.8, color="black", lw=0.8, alpha=0.4)
        ax.set_ylim(0, 1)
        ax.set_title(f"Coverage (cum), macro-F1={macro_f1:.3f}")
        ax.set_ylabel("coverage")
    else:
        ax.set_axis_off()

    fig.tight_layout()
    out_pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/raw/entsoe_data_2024_2025.csv"))
    parser.add_argument("--country", default="PL")
    parser.add_argument("--market-tz", default="Europe/Warsaw")
    parser.add_argument("--spike-window-weeks", type=int, default=8)
    parser.add_argument("--min-mw", type=float, default=1e-6)

    parser.add_argument("--train-window-weeks", type=int, default=52)
    parser.add_argument("--review-weeks", type=int, default=8)
    parser.add_argument("--min-week-hours", type=int, default=160)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--backtest-dir", type=Path, default=Path("backtest_results"))
    parser.add_argument("--final-template", type=Path, default=Path("reports/final_report_template.md"))
    parser.add_argument("--final-pdf", type=Path, default=Path("reports/final_report.pdf"))
    args = parser.parse_args(argv)

    cols = Columns()
    df = _read_csv(args.csv)

    required = [cols.datetime, cols.country, cols.price, cols.pumped_gen, cols.pumped_consume]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df.loc[df[cols.country] == args.country].copy()
    if df.empty:
        raise SystemExit(f"No rows found for country={args.country}")

    df = _add_time_columns(df, cols=cols, market_tz=args.market_tz)
    df = _add_hourly_features(df, cols=cols, spike_window_weeks=args.spike_window_weeks, min_mw=args.min_mw)

    weekly_all = _weekly_features(df)
    weekly = weekly_all.loc[weekly_all["n_hours"] >= int(args.min_week_hours)].copy()
    supervised = _prepare_supervised(weekly)

    feature_cols = [
        "price_mean",
        "price_std",
        "price_p10",
        "price_p90",
        "vol_range",
        "neg_hours",
        "spike_hours",
        "pumped_gen_hours",
        "pumped_pump_hours",
        "pumped_net_mean",
        "pumped_net_p10",
        "pumped_net_p90",
        "renewable_share_mean",
        "gas_share_mean",
        "coal_share_mean",
        "nuclear_share_mean",
        "wind_ramp_p90",
        "solar_ramp_p90",
        "renewable_ramp_p90",
        "week_of_year",
        "month",
    ]
    feature_cols = [c for c in feature_cols if c in supervised.columns]

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.backtest_dir.mkdir(parents=True, exist_ok=True)

    weekly_all.to_csv(args.reports_dir / "weekly_features_pl.csv", index=False)

    bt = _walk_forward_backtest(supervised, feature_cols=feature_cols, train_window_weeks=args.train_window_weeks)
    bt.to_csv(args.backtest_dir / "walk_forward_predictions.csv", index=False)

    _plot_interval_backtest(bt, args.backtest_dir / "interval_backtest.png")
    macro_f1 = _plot_confusion(bt, args.backtest_dir / "direction_confusion.png")
    _plot_coverage(bt, args.backtest_dir / "coverage_over_time.png")
    _plot_pred_error(bt, args.backtest_dir / "pred_error_over_time.png")

    pdf_path = args.reports_dir / "weekly_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _plot_week_report_page1(df=df, weekly=weekly, out_pdf=pdf)
        _plot_week_report_page2(weekly=weekly, bt=bt, macro_f1=macro_f1, out_pdf=pdf)

    coverage = float(bt["covered_10_90"].mean()) if len(bt) else float("nan")
    summary = (
        "# Weekly report generated\n\n"
        f"- country: {args.country}\n"
        f"- market_tz: {args.market_tz}\n"
        f"- pdf: {pdf_path}\n"
        f"- backtest_rows: {len(bt)}\n"
        f"- macro_f1: {macro_f1:.3f}\n"
        f"- coverage_10_90: {coverage:.3f}\n"
    )
    (args.reports_dir / "summary.md").write_text(summary, encoding="utf-8")
    _write_weekly_review(
        weekly=weekly,
        bt=bt,
        macro_f1=macro_f1,
        out_path=args.reports_dir / "weekly_review.md",
        lookback=args.review_weeks,
    )

    _write_final_report_template_md(args.final_template)
    _write_final_report_pdf(
        args.final_pdf,
        df=df,
        weekly=weekly,
        bt=bt,
        macro_f1=macro_f1,
        country=args.country,
        market_tz=args.market_tz,
        spike_window_weeks=args.spike_window_weeks,
        train_window_weeks=args.train_window_weeks,
        min_week_hours=args.min_week_hours,
        review_weeks=args.review_weeks,
        backtest_dir=args.backtest_dir,
    )

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {args.final_template}")
    print(f"Wrote: {args.final_pdf}")
    print(f"Wrote: {args.backtest_dir}")
    return 0
