"""
grid_visualizer.py
==================
Визуализация результатов оптимизации лонговой сетки.

Строит:
  1. Equity curve лучшей конфигурации
  2. Схему сетки (уровни ордеров и TP)
  3. Гистограмму PnL по сделкам
  4. Распределение длительности сделок
  5. Heatmap: n_orders vs tp_pct (средний score)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from typing import Optional

from grid_backtest import GridParams, BacktestResult, run_grid_backtest


# ─────────────────────────────────────────────────────────────────────────────
# Equity + детали лучшей конфигурации
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_result(
    result:    BacktestResult,
    params:    dict,
    df:        pd.DataFrame,
    symbol:    str = "UNKNOWN",
    save_path: str = "grid_best_equity.png",
):
    """Полный отчёт по лучшей конфигурации."""

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    reinvest_flag = getattr(result, "reinvest", False)
    reinvest_label = "  ♻ REINVEST=ON" if reinvest_flag else ""
    title = (f"Grid Long — {symbol}  |  "
             f"N={params['n_orders']}  TP={params['tp_pct']:.2f}%  "
             f"Steps={params['step_min']:.2f}..{params['step_max']:.2f}%  "
             f"Sizes={params['size_min']:.0f}..{params['size_max']:.0f}$"
             f"{reinvest_label}")

    # ── 1. PnL curve (широкий, верхний ряд) ─────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    if result.equity_curve:
        initial = result.initial_capital

        ts_raw = [e[0] for e in result.equity_curve]
        eq_raw = [e[1] for e in result.equity_curve]

        # Даунсемплинг с сохранением минимумов просадок:
        # делим на MAX_PTS окон, в каждом берём min (худший момент)
        MAX_PTS = 4000
        n_pts = len(eq_raw)
        if n_pts > MAX_PTS:
            import math
            chunk = math.ceil(n_pts / MAX_PTS)
            ts_ds, eq_ds = [], []
            for k in range(0, n_pts, chunk):
                segment = eq_raw[k:k + chunk]
                # берём индекс минимума в окне — сохраняем худшую просадку
                min_idx = k + segment.index(min(segment))
                ts_ds.append(ts_raw[min_idx])
                eq_ds.append(min(segment))
            ts, eq = ts_ds, eq_ds
        else:
            ts, eq = ts_raw, eq_raw

        pnl      = [e - initial for e in eq]
        peak_eq  = np.maximum.accumulate(eq)
        peak_pnl = [p - initial for p in peak_eq]

        ax_eq.plot(ts, pnl,      color="#00c49a", linewidth=1.0, label="PnL (bar_low)")
        ax_eq.plot(ts, peak_pnl, color="#888888", linewidth=0.8, linestyle="--", label="Peak PnL")
        ax_eq.fill_between(ts, pnl, peak_pnl, alpha=0.25, color="#ff4444", label="Drawdown")
        ax_eq.axhline(0, color="white", linewidth=0.7, linestyle=":")

        # Зона самой длинной сделки
        if result.trade_log:
            longest   = max(result.trade_log, key=lambda t: t["duration"])
            open_bar  = longest["open_bar"]
            close_bar = longest["close_bar"]
            if "timestamp" in df.columns:
                ts_open  = df["timestamp"].iloc[open_bar]
                ts_close = df["timestamp"].iloc[min(close_bar, len(df) - 1)]
            else:
                ts_open  = open_bar
                ts_close = close_bar
            dur_h = longest["duration"] / 60
            label = (f"Макс. сделка {longest['duration']:.0f} мин ({dur_h:.1f} ч)"
                     + ("  [forced]" if longest.get("forced") else ""))
            ax_eq.axvspan(ts_open, ts_close,
                          alpha=0.18, color="#ff3333", label=label, zorder=0)
            ax_eq.axvline(ts_open,  color="#ff6666", linewidth=1.0, linestyle=":", alpha=0.7)
            ax_eq.axvline(ts_close, color="#ff6666", linewidth=1.0, linestyle=":", alpha=0.7)

        # Жёлтая линия — момент максимальной просадки
        max_dd_bar = getattr(result, "max_dd_bar", None)
        if max_dd_bar is not None and "timestamp" in df.columns:
            ts_maxdd = df["timestamp"].iloc[min(max_dd_bar, len(df) - 1)]
            ax_eq.axvline(ts_maxdd, color="#ffdd00", linewidth=1.5,
                          linestyle="--", alpha=0.9,
                          label=f"Макс. DD {result.max_drawdown:.0f}$ (момент)")

    ax_eq.set_title(title, fontsize=11, color="white", pad=8)
    ax_eq.set_ylabel("PnL ($)", color="white")
    ax_eq.tick_params(colors="white")
    ax_eq.set_facecolor("#1a1a2e")
    ax_eq.grid(color="#333366", linewidth=0.4)
    ax_eq.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e", labelcolor="white")

    # ── 2. Схема сетки ────────────────────────────────────────────────────────
    ax_grid = fig.add_subplot(gs[1, 0])
    n = params["n_orders"]
    gp = GridParams(n_orders=n, steps=params["steps"], sizes=params["sizes"], tp_pct=params["tp_pct"])
    ref = 100.0  # нормированная цена
    levels = []
    cum = 0.0
    for s in gp.steps:
        cum += s
        levels.append(ref * (1 - cum / 100))

    avg_price = sum(levels[i] * gp.sizes[i] for i in range(n)) / sum(gp.sizes)
    tp_price  = avg_price * (1 + gp.tp_pct / 100)

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, n))
    for i, (lv, sz) in enumerate(zip(levels, gp.sizes)):
        ax_grid.barh(lv, sz, height=0.3, color=colors[i], alpha=0.8)
        ax_grid.axhline(lv, color=colors[i], linewidth=0.6, linestyle=":", alpha=0.5)

    ax_grid.axhline(ref,      color="#4cc9f0", linewidth=1.5, label=f"Entry {ref:.0f}")
    ax_grid.axhline(avg_price, color="#f4a261", linewidth=1.2, linestyle="--",
                    label=f"AvgEntry {avg_price:.2f}")
    ax_grid.axhline(tp_price,  color="#00c49a", linewidth=1.5, linestyle="-",
                    label=f"TP {tp_price:.2f}")
    ax_grid.set_title("Схема сетки (нормир.)", color="white", fontsize=9)
    ax_grid.set_xlabel("Размер ордера ($)", color="white")
    ax_grid.set_ylabel("Цена", color="white")
    ax_grid.tick_params(colors="white")
    ax_grid.set_facecolor("#1a1a2e")
    ax_grid.grid(color="#333366", linewidth=0.3)
    ax_grid.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

    # ── 3. Шаги ордеров ───────────────────────────────────────────────────────
    ax_steps = fig.add_subplot(gs[1, 1])
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n))
    ax_steps.bar(range(1, n+1), gp.steps, color=bar_colors, alpha=0.85)
    ax_steps.set_title("Шаги ордеров (%)", color="white", fontsize=9)
    ax_steps.set_xlabel("Ордер №", color="white")
    ax_steps.set_ylabel("Шаг (%)", color="white")
    ax_steps.tick_params(colors="white")
    ax_steps.set_facecolor("#1a1a2e")
    ax_steps.grid(color="#333366", linewidth=0.3, axis="y")

    # ── 4. Размеры ордеров ────────────────────────────────────────────────────
    ax_sizes = fig.add_subplot(gs[1, 2])
    bar_colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, n))
    ax_sizes.bar(range(1, n+1), gp.sizes, color=bar_colors2, alpha=0.85)
    ax_sizes.set_title("Размеры ордеров ($)", color="white", fontsize=9)
    ax_sizes.set_xlabel("Ордер №", color="white")
    ax_sizes.set_ylabel("Размер ($)", color="white")
    ax_sizes.tick_params(colors="white")
    ax_sizes.set_facecolor("#1a1a2e")
    ax_sizes.grid(color="#333366", linewidth=0.3, axis="y")

    # ── 5. PnL по сделкам ────────────────────────────────────────────────────
    ax_trades = fig.add_subplot(gs[2, :2])
    pnls = [t["pnl"] for t in result.trade_log]
    if pnls:
        forced_idx = [i for i, t in enumerate(result.trade_log) if t.get("forced")]
        bar_c = []
        for i, (t, p) in enumerate(zip(result.trade_log, pnls)):
            if t.get("forced"):
                bar_c.append("#ff9900")   # оранжевый — принудительное закрытие
            elif p > 0:
                bar_c.append("#00c49a")
            else:
                bar_c.append("#ff4444")
        ax_trades.bar(range(len(pnls)), pnls, color=bar_c, width=0.8, alpha=0.85)
        ax_trades.axhline(0, color="white", linewidth=0.5)
        title = f"PnL по сделкам (всего {len(pnls)})"
        if forced_idx:
            title += "  ⚡ оранжевый = незакрытая на конец периода"
        ax_trades.set_title(title, color="white", fontsize=9)
        ax_trades.set_xlabel("Сделка №", color="white")
        ax_trades.set_ylabel("PnL ($)", color="white")
        ax_trades.tick_params(colors="white")
        ax_trades.set_facecolor("#1a1a2e")
        ax_trades.grid(color="#333366", linewidth=0.3, axis="y")

    # ── 6. Длительность сделок ───────────────────────────────────────────────
    ax_dur = fig.add_subplot(gs[2, 2])
    durations = [t["duration"] for t in result.trade_log]
    if durations:
        ax_dur.hist(durations, bins=30, color="#4cc9f0", alpha=0.8, edgecolor="#0f0f1a")
        ax_dur.axvline(np.mean(durations), color="#f4a261", linewidth=1.5,
                       linestyle="--", label=f"Ср. {np.mean(durations):.0f} мин")
        ax_dur.set_title("Длительность сделок (мин)", color="white", fontsize=9)
        ax_dur.set_xlabel("Минут", color="white")
        ax_dur.set_ylabel("Кол-во", color="white")
        ax_dur.tick_params(colors="white")
        ax_dur.set_facecolor("#1a1a2e")
        ax_dur.grid(color="#333366", linewidth=0.3)
        ax_dur.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    reinvest_str = "  ♻ REINVEST=ON" if getattr(result, "reinvest", False) else ""
    plt.suptitle(
        f"Отчёт: Grid Long Strategy{reinvest_str}  |  "
        f"PnL={result.total_pnl:+.2f}$  MaxDD={result.max_drawdown:.2f}$  "
        f"WR={result.win_rate:.1f}%  PF={result.profit_factor:.2f}  "
        f"Sharpe={result.sharpe:.2f}  Score={result.score:.4f}",
        fontsize=12, color="white", y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    print(f"  График сохранён: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap: n_orders vs tp_pct
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    df_results: pd.DataFrame,
    save_path:  str = "grid_heatmap.png",
):
    """Heatmap корреляции n_orders и tp_pct со score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")

    valid = df_results[df_results["score"] > -999].copy()
    if valid.empty:
        print("  Нет данных для heatmap.")
        return

    # Бинируем
    valid["n_bin"]  = pd.cut(valid["n_orders"], bins=8)
    valid["tp_bin"] = pd.cut(valid["tp_pct"],   bins=8)

    for ax, (metric, cmap, label) in zip(axes, [
        ("score",    "YlGn",   "Score"),
        ("total_pnl","YlOrRd", "Total PnL ($)"),
    ]):
        pivot = valid.pivot_table(values=metric, index="n_bin", columns="tp_bin", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right",
                           fontsize=7, color="white")
        ax.set_yticklabels([str(i) for i in pivot.index], fontsize=7, color="white")
        ax.set_title(f"{label} vs N_Orders / TP%", color="white", fontsize=10)
        ax.set_xlabel("TP %", color="white")
        ax.set_ylabel("N Orders", color="white")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    print(f"  Heatmap сохранён: {save_path}")
    plt.close()