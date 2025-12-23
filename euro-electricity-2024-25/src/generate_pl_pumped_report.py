#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class Columns:
    datetime: str = "datetime"
    country: str = "country"
    price: str = "price"
    pumped_gen: str = "_hydro_pumped_storage_actual_aggregated_"
    pumped_consume: str = "_hydro_pumped_storage_actual_consumption_"


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if len(df.columns) > 0 and isinstance(df.columns[0], str) and df.columns[0].startswith("\ufeff"):
        df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("\ufeff")})
    return df


def _ensure_required_columns(df: pd.DataFrame, cols: Columns) -> None:
    missing = [c for c in [cols.datetime, cols.country, cols.price, cols.pumped_gen, cols.pumped_consume] if c not in df]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def _add_time_columns(df: pd.DataFrame, cols: Columns, market_tz: str) -> pd.DataFrame:
    out = df.copy()
    out["datetime_utc"] = pd.to_datetime(out[cols.datetime], utc=True, errors="raise")
    out["datetime_local"] = out["datetime_utc"].dt.tz_convert(market_tz)
    out = out.sort_values("datetime_utc").reset_index(drop=True)

    # Week starts Monday 00:00 in market local time.
    dow = out["datetime_local"].dt.dayofweek
    out["week_start_local"] = (out["datetime_local"] - pd.to_timedelta(dow, unit="D")).dt.normalize()
    out["date_local"] = out["datetime_local"].dt.date
    out["hour_local"] = out["datetime_local"].dt.hour
    return out


def _add_pumped_features(df: pd.DataFrame, cols: Columns, min_mw: float) -> pd.DataFrame:
    out = df.copy()
    pumped_gen = out[cols.pumped_gen].fillna(0.0)
    pumped_consume = out[cols.pumped_consume].fillna(0.0)

    out["pumped_gen_mw"] = pumped_gen
    out["pumped_consume_mw"] = pumped_consume
    out["pumped_net_mw"] = pumped_gen - pumped_consume

    gen_on = pumped_gen > min_mw
    pump_on = pumped_consume > min_mw
    out["pumped_mode"] = "idle"
    out.loc[gen_on & ~pump_on, "pumped_mode"] = "generate"
    out.loc[pump_on & ~gen_on, "pumped_mode"] = "pump"
    out.loc[gen_on & pump_on, "pumped_mode"] = "both"
    return out


def _add_risk_flags(df: pd.DataFrame, cols: Columns, spike_window_weeks: int) -> pd.DataFrame:
    out = df.copy()
    window_hours = int(spike_window_weeks * 7 * 24)
    price = out[cols.price].astype("float64")

    # Rolling 0.99 quantile threshold based on past hours only (shifted by 1).
    if window_hours <= 0:
        raise SystemExit("--spike-window-weeks must be > 0")
    thresh = price.rolling(window_hours, min_periods=max(1, window_hours // 2)).quantile(0.99).shift(1)

    out["neg_flag"] = price < 0
    out["spike_threshold"] = thresh
    out["spike_flag"] = price > thresh
    return out


def _weekly_agg(df: pd.DataFrame, cols: Columns) -> pd.DataFrame:
    g = df.groupby("week_start_local", sort=True)

    def q(x: pd.Series, quantile: float) -> float:
        return float(x.quantile(quantile))

    weekly = pd.DataFrame(
        {
            "n_hours": g.size(),
            "price_mean": g[cols.price].mean(),
            "price_std": g[cols.price].std(),
            "price_p10": g[cols.price].apply(lambda s: q(s, 0.10)),
            "price_p50": g[cols.price].apply(lambda s: q(s, 0.50)),
            "price_p90": g[cols.price].apply(lambda s: q(s, 0.90)),
            "neg_hours": g["neg_flag"].sum(),
            "spike_hours": g["spike_flag"].sum(),
            "pumped_gen_hours": g["pumped_gen_mw"].apply(lambda s: int((s > 0).sum())),
            "pumped_pump_hours": g["pumped_consume_mw"].apply(lambda s: int((s > 0).sum())),
            "pumped_both_hours": g["pumped_mode"].apply(lambda s: int((s == "both").sum())),
            "pumped_net_mean": g["pumped_net_mw"].mean(),
        }
    ).reset_index()

    return weekly


def _latest_full_week_start(weekly: pd.DataFrame) -> pd.Timestamp:
    # Prefer the latest week with near-full hours; tolerate DST/missing by using a soft threshold.
    candidates = weekly.loc[weekly["n_hours"] >= 160, "week_start_local"]
    if not candidates.empty:
        return candidates.iloc[-1]
    return weekly["week_start_local"].iloc[-1]


def _write_markdown_summary(
    out_dir: Path,
    df_pl: pd.DataFrame,
    weekly: pd.DataFrame,
    cols: Columns,
    market_tz: str,
    spike_window_weeks: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_week_start = _latest_full_week_start(weekly)
    w = weekly.set_index("week_start_local").sort_index()
    latest = w.loc[latest_week_start]
    prev = w.iloc[-2] if len(w) >= 2 else None

    in_latest = df_pl["week_start_local"] == latest_week_start
    df_w = df_pl.loc[in_latest].copy()
    price = df_w[cols.price].astype("float64")

    price_p10 = float(price.quantile(0.10))
    price_p90 = float(price.quantile(0.90))
    pump_high = int(((df_w["pumped_mode"] == "pump") & (price > price_p90)).sum())
    gen_low = int(((df_w["pumped_mode"] == "generate") & (price < price_p10)).sum())

    p_pump = df_w.loc[df_w["pumped_mode"] == "pump", cols.price].mean()
    p_gen = df_w.loc[df_w["pumped_mode"] == "generate", cols.price].mean()
    spread = float(p_gen - p_pump) if pd.notna(p_pump) and pd.notna(p_gen) else float("nan")

    corr_net_price = float(df_pl["pumped_net_mw"].corr(df_pl[cols.price]))
    corr_consume_price = float(df_pl["pumped_consume_mw"].corr(df_pl[cols.price]))
    corr_gen_price = float(df_pl["pumped_gen_mw"].corr(df_pl[cols.price]))

    start_local = df_pl["datetime_local"].min()
    end_local = df_pl["datetime_local"].max()

    delta_line = ""
    if prev is not None and pd.notna(prev["price_mean"]):
        delta = float(latest["price_mean"] - prev["price_mean"])
        delta_line = f"- 环比（均价）：{delta:+.2f} €/MWh\n"

    md = (
        f"# PL 抽蓄套利 / 风险提示周报（自动生成）\n\n"
        f"- 市场：PL\n"
        f"- 时区：{market_tz}\n"
        f"- 覆盖范围：{start_local} → {end_local}\n"
        f"- Spike 定义：过去 {spike_window_weeks} 周小时价 99 分位阈值（滚动、仅用历史，shift=1）\n\n"
        f"## 最新周（week_start_local={latest_week_start}）概览\n\n"
        f"- 均价：{latest['price_mean']:.2f} €/MWh\n"
        f"{delta_line}"
        f"- 波动区间（P10–P90）：{latest['price_p10']:.2f} → {latest['price_p90']:.2f} €/MWh\n"
        f"- 负价小时：{int(latest['neg_hours'])}\n"
        f"- Spike 小时：{int(latest['spike_hours'])}\n"
        f"- 抽蓄发电小时：{int(latest['pumped_gen_hours'])}\n"
        f"- 抽蓄吸纳小时：{int(latest['pumped_pump_hours'])}\n"
        f"- 抽蓄净出力均值：{latest['pumped_net_mean']:.2f} MW\n\n"
        f"## 套利直觉验证（条件价格）\n\n"
        f"- 吸纳（pump）时均价：{(p_pump if pd.notna(p_pump) else float('nan')):.2f} €/MWh\n"
        f"- 发电（generate）时均价：{(p_gen if pd.notna(p_gen) else float('nan')):.2f} €/MWh\n"
        f"- 简单价差（generate - pump）：{spread:.2f} €/MWh\n\n"
        f"## 风险提示（最新周）\n\n"
        f"- 高价吸纳小时（pump & price>P90）：{pump_high}\n"
        f"- 低价发电小时（generate & price<P10）：{gen_low}\n\n"
        f"## 行为与价格相关性（全样本）\n\n"
        f"- corr(pumped_net, price)：{corr_net_price:.3f}\n"
        f"- corr(pumped_consume, price)：{corr_consume_price:.3f}\n"
        f"- corr(pumped_gen, price)：{corr_gen_price:.3f}\n"
    )

    (out_dir / "pl_pumped_weekly_summary.md").write_text(md, encoding="utf-8")


def _plot_last_days(df: pd.DataFrame, cols: Columns, out_dir: Path, last_days: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    if df.empty:
        return

    cutoff = df["datetime_utc"].max() - pd.Timedelta(days=last_days)
    sub = df.loc[df["datetime_utc"] >= cutoff].copy()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(sub["datetime_local"], sub[cols.price], lw=1.0, color="tab:blue", label="price (€/MWh)")
    ax1.set_ylabel("price (€/MWh)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(sub["datetime_local"], sub["pumped_net_mw"], lw=1.0, color="tab:orange", label="pumped_net (MW)")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax2.set_ylabel("pumped_net (MW)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(f"PL last {last_days} days: price vs pumped_net")
    fig.tight_layout()
    fig.savefig(out_dir / "pl_last_days_price_vs_pumped_net.png", dpi=160)
    plt.close(fig)


def _plot_price_by_mode(df: pd.DataFrame, cols: Columns, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["pump", "idle", "generate"]
    data = [df.loc[df["pumped_mode"] == m, cols.price].dropna().astype("float64").values for m in modes]
    if sum(len(x) for x in data) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=modes, showfliers=False)
    ax.set_title("PL price distribution by pumped mode")
    ax.set_ylabel("price (€/MWh)")
    fig.tight_layout()
    fig.savefig(out_dir / "pl_price_by_pumped_mode_box.png", dpi=160)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, cols: Columns, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[[cols.price, "pumped_net_mw"]].dropna()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["pumped_net_mw"], sub[cols.price], s=3, alpha=0.15, edgecolors="none")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("PL price vs pumped_net scatter")
    ax.set_xlabel("pumped_net (MW) [gen - consume]")
    ax.set_ylabel("price (€/MWh)")
    fig.tight_layout()
    fig.savefig(out_dir / "pl_scatter_price_vs_pumped_net.png", dpi=160)
    plt.close(fig)


def _plot_weekly_dashboard(weekly: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    w = weekly.sort_values("week_start_local").copy()
    if w.empty:
        return

    x = w["week_start_local"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(x, w["price_mean"], color="tab:blue", lw=1.5, label="mean")
    ax.fill_between(x, w["price_p10"], w["price_p90"], color="tab:blue", alpha=0.15, label="P10–P90")
    ax.set_ylabel("€/MWh")
    ax.set_title("PL weekly price level & dispersion")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    ax.bar(x, w["neg_hours"], color="tab:purple", alpha=0.8, label="neg_hours")
    ax.bar(x, w["spike_hours"], bottom=w["neg_hours"], color="tab:red", alpha=0.6, label="spike_hours")
    ax.set_ylabel("hours")
    ax.set_title("Weekly risk hours (negative + spike)")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[2]
    ax.plot(x, w["pumped_gen_hours"], color="tab:orange", lw=1.5, label="gen_hours")
    ax.plot(x, w["pumped_pump_hours"], color="tab:green", lw=1.5, label="pump_hours")
    ax.set_ylabel("hours")
    ax.set_title("Weekly pumped-storage activity (hours)")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[3]
    ax.plot(x, w["pumped_net_mean"], color="tab:gray", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_ylabel("MW")
    ax.set_title("Weekly mean pumped_net (MW)")

    fig.tight_layout()
    fig.savefig(out_dir / "pl_weekly_dashboard.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/raw/entsoe_data_2024_2025.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/pl_pumped"))
    parser.add_argument("--country", default="PL")
    parser.add_argument("--market-tz", default="Europe/Warsaw")
    parser.add_argument("--spike-window-weeks", type=int, default=8)
    parser.add_argument("--last-days", type=int, default=14)
    parser.add_argument("--min-mw", type=float, default=1e-6)
    args = parser.parse_args()

    cols = Columns()
    df = _read_csv(args.csv)
    _ensure_required_columns(df, cols)

    df_pl = df.loc[df[cols.country] == args.country].copy()
    if df_pl.empty:
        raise SystemExit(f"No rows found for country={args.country}")

    df_pl = _add_time_columns(df_pl, cols=cols, market_tz=args.market_tz)
    df_pl = _add_pumped_features(df_pl, cols=cols, min_mw=args.min_mw)
    df_pl = _add_risk_flags(df_pl, cols=cols, spike_window_weeks=args.spike_window_weeks)

    weekly = _weekly_agg(df_pl, cols=cols)

    args.out.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out / "pl_weekly_features.csv", index=False)

    _write_markdown_summary(
        out_dir=args.out,
        df_pl=df_pl,
        weekly=weekly,
        cols=cols,
        market_tz=args.market_tz,
        spike_window_weeks=args.spike_window_weeks,
    )

    _plot_last_days(df_pl, cols=cols, out_dir=args.out, last_days=args.last_days)
    _plot_price_by_mode(df_pl, cols=cols, out_dir=args.out)
    _plot_scatter(df_pl, cols=cols, out_dir=args.out)
    _plot_weekly_dashboard(weekly, out_dir=args.out)

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
