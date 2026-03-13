"""
grid_optimizer.py
=================
Оптимизатор параметров лонговой сетки ордеров.

Алгоритм: Random Search с адаптивным сужением диапазонов (Elite Guided Search).

Параметры оптимизации:
  - n_orders:  количество ордеров (1..40)
  - steps[i]:  индивидуальный шаг i-го ордера в % (0.1%..5%)
  - sizes[i]:  индивидуальный размер i-го ордера в USDT
  - tp_pct:    % тейк-профита (0.3%..10%)

Запуск:
    python grid_optimizer.py --symbol BTCUSDT --file path/to/data.csv --iters 2000
"""

from __future__ import annotations

import argparse
import random
import time
import json
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List

from grid_backtest import GridParams, BacktestResult, run_grid_backtest, compute_score


# ─────────────────────────────────────────────────────────────────────────────
# Пространство поиска
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "n_orders":  (1,   40),       # целое
    "step_min":  (0.1, 2.0),      # % минимальный шаг ордера
    "step_max":  (0.3, 8.0),      # % максимальный шаг ордера
    "size_min":  (5.0, 50.0),     # USDT минимальный размер ордера
    "size_max":  (20.0, 500.0),   # USDT максимальный размер ордера
    "tp_pct":    (0.3, 8.0),      # % тейк-профита
}

# Режимы распределения шагов ордеров
STEP_MODES  = ["linear", "geometric", "random", "front_heavy", "back_heavy"]

# Режимы распределения размеров ордеров
SIZE_MODES  = ["flat", "linear_increase", "geometric_increase", "pyramid", "random"]


# ─────────────────────────────────────────────────────────────────────────────
# Генерация параметров
# ─────────────────────────────────────────────────────────────────────────────

def _generate_steps(n: int, step_min: float, step_max: float, mode: str) -> List[float]:
    """Генерирует список индивидуальных шагов для N ордеров."""
    if n == 1:
        return [round(random.uniform(step_min, step_max), 3)]

    if mode == "linear":
        # Равномерно распределённые шаги
        return [round(step_min + (step_max - step_min) * i / (n - 1), 3) for i in range(n)]

    elif mode == "geometric":
        # Геометрически растущие шаги
        ratio = (step_max / max(step_min, 0.01)) ** (1.0 / (n - 1))
        return [round(step_min * (ratio ** i), 3) for i in range(n)]

    elif mode == "front_heavy":
        # Большие шаги в начале (первые ордера ближе)
        steps = [round(step_max - (step_max - step_min) * i / (n - 1), 3) for i in range(n)]
        return steps

    elif mode == "back_heavy":
        # Маленькие шаги в начале, большие в конце
        steps = [round(step_min + (step_max - step_min) * (i / (n - 1)) ** 2, 3) for i in range(n)]
        return steps

    else:  # random
        return [round(random.uniform(step_min, step_max), 3) for _ in range(n)]


def _generate_sizes(n: int, size_min: float, size_max: float, mode: str) -> List[float]:
    """Генерирует список индивидуальных размеров для N ордеров."""
    if n == 1:
        return [round(random.uniform(size_min, size_max), 2)]

    if mode == "flat":
        base = round(random.uniform(size_min, size_max), 2)
        return [base] * n

    elif mode == "linear_increase":
        # Нижние ордера (дальше от цены) — крупнее (мартингейл-лайт)
        return [round(size_min + (size_max - size_min) * i / (n - 1), 2) for i in range(n)]

    elif mode == "geometric_increase":
        ratio = (size_max / max(size_min, 1.0)) ** (1.0 / (n - 1))
        return [round(size_min * (ratio ** i), 2) for i in range(n)]

    elif mode == "pyramid":
        # Середина сетки — самые крупные ордера
        half = n // 2
        sizes = []
        for i in range(n):
            dist = abs(i - half) / max(half, 1)
            s = size_max - (size_max - size_min) * dist
            sizes.append(round(s, 2))
        return sizes

    else:  # random
        return [round(random.uniform(size_min, size_max), 2) for _ in range(n)]


def sample_params(space: dict = None) -> dict:
    """Генерирует случайный набор параметров."""
    if space is None:
        space = SEARCH_SPACE

    n_orders  = random.randint(int(space["n_orders"][0]), int(space["n_orders"][1]))
    step_min  = round(random.uniform(*space["step_min"]), 3)
    step_max  = round(random.uniform(*space["step_max"]), 3)
    if step_max < step_min:
        step_min, step_max = step_max, step_min
    step_max = max(step_max, step_min + 0.05)

    size_min  = round(random.uniform(*space["size_min"]), 2)
    size_max  = round(random.uniform(*space["size_max"]), 2)
    if size_max < size_min:
        size_min, size_max = size_max, size_min
    size_max = max(size_max, size_min + 1.0)

    tp_pct    = round(random.uniform(*space["tp_pct"]), 3)
    step_mode = random.choice(STEP_MODES)
    size_mode = random.choice(SIZE_MODES)

    steps = _generate_steps(n_orders, step_min, step_max, step_mode)
    sizes = _generate_sizes(n_orders, size_min, size_max, size_mode)

    return {
        "n_orders":  n_orders,
        "steps":     steps,
        "sizes":     sizes,
        "tp_pct":    tp_pct,
        "step_mode": step_mode,
        "size_mode": size_mode,
        # сохраняем диапазоны для логирования
        "step_min":  step_min,
        "step_max":  step_max,
        "size_min":  size_min,
        "size_max":  size_max,
    }


def params_to_grid(p: dict) -> GridParams:
    return GridParams(
        n_orders = p["n_orders"],
        steps    = p["steps"],
        sizes    = p["sizes"],
        tp_pct   = p["tp_pct"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Адаптивное сужение (Elite Guided Search)
# ─────────────────────────────────────────────────────────────────────────────

def narrow_space(elite_params: list, base_space: dict, shrink: float = 0.5) -> dict:
    """
    Сужает пространство поиска на основе топ-параметров.
    shrink: 0..1 — насколько сильно сужать (0.5 = на 50%)
    """
    if not elite_params:
        return base_space

    new_space = deepcopy(base_space)

    for key in ["tp_pct", "step_min", "step_max", "size_min", "size_max"]:
        vals = [p[key] for p in elite_params if key in p]
        if not vals:
            continue
        lo, hi = base_space[key]
        elite_lo = min(vals)
        elite_hi = max(vals)
        center   = (elite_lo + elite_hi) / 2
        half_range = max((hi - lo) * (1 - shrink) / 2, (elite_hi - elite_lo) * 0.6)
        new_lo = max(lo, center - half_range)
        new_hi = min(hi, center + half_range)
        if new_hi - new_lo < 0.01:
            new_hi = new_lo + 0.01
        new_space[key] = (round(new_lo, 4), round(new_hi, 4))

    # n_orders — сужаем около медианы
    n_vals = [p["n_orders"] for p in elite_params]
    med_n  = int(np.median(n_vals))
    spread = max(3, int((base_space["n_orders"][1] - base_space["n_orders"][0]) * (1 - shrink) / 2))
    new_space["n_orders"] = (
        max(1, med_n - spread),
        min(40, med_n + spread)
    )

    return new_space


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция оптимизации
# ─────────────────────────────────────────────────────────────────────────────

def optimize(
    df:           pd.DataFrame,
    n_iter:       int   = 2000,
    commission:   float = 0.0006,
    initial_capital: float = 1000.0,
    save_to:      str   = "grid_results.csv",
    top_n:        int   = 20,
    print_every:  int   = 100,
    seed:         int   = None,
    min_orders:   int   = 1,      # минимальное кол-во ордеров в сетке
    elite_frac:   float = 0.05,   # доля лучших для сужения пространства
    narrow_every: int   = 300,    # каждые N итераций сужаем пространство
    narrow_shrink: float = 0.3,   # насколько сильно сужать
) -> pd.DataFrame:
    """
    Random Search + Elite Guided Narrowing.

    Возвращает DataFrame со всеми результатами (топ по score).
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    space   = deepcopy(SEARCH_SPACE)
    space["n_orders"] = (max(min_orders, int(SEARCH_SPACE["n_orders"][0])),
                          int(SEARCH_SPACE["n_orders"][1]))
    results = []
    best    = None
    best_score = -999.0
    t0 = time.time()

    print(f"\n{'═'*60}")
    print(f"  Grid Long Strategy Optimizer")
    print(f"  Итераций: {n_iter}  |  Данных: {len(df)} баров")
    print(f"  Период: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"{'═'*60}\n")

    for idx in range(1, n_iter + 1):

        # ── Адаптивное сужение пространства ──────────────────────────────────
        if idx % narrow_every == 0 and results:
            valid   = [r for r in results if r["score"] > -999]
            if len(valid) >= 10:
                k       = max(1, int(len(valid) * elite_frac))
                top_res = sorted(valid, key=lambda x: x["score"], reverse=True)[:k]
                elite   = [r["params"] for r in top_res]
                space   = narrow_space(elite, SEARCH_SPACE, shrink=narrow_shrink)
                print(f"  [iter {idx}] Пространство сужено. n_orders: {space['n_orders']}, "
                      f"tp_pct: {space['tp_pct']}, step: {space['step_min']}..{space['step_max']}")

        # ── Генерация и бэктест ───────────────────────────────────────────────
        p = sample_params(space)
        gp = params_to_grid(p)

        try:
            r = run_grid_backtest(df, gp, commission=commission, initial_capital=initial_capital)
            s = r.score
        except Exception as e:
            s = -999.0
            r = BacktestResult()

        row = {
            "score":             s,
            "total_pnl":         round(r.total_pnl, 4),
            "max_drawdown":      round(r.max_drawdown, 4),
            "sharpe":            round(r.sharpe, 4),
            "win_rate":          round(r.win_rate, 2),
            "profit_factor":     round(r.profit_factor, 4),
            "trade_count":       r.trade_count,
            "avg_trade_minutes": round(r.avg_trade_minutes, 1),
            "max_trade_minutes": round(r.max_trade_minutes, 1),
            "n_orders":          p["n_orders"],
            "tp_pct":            p["tp_pct"],
            "step_min":          p["step_min"],
            "step_max":          p["step_max"],
            "step_mode":         p["step_mode"],
            "size_min":          p["size_min"],
            "size_max":          p["size_max"],
            "size_mode":         p["size_mode"],
            "params":            p,       # полные параметры для воспроизведения
        }
        results.append(row)

        if s > best_score and s > -999:
            # Если score лучше — проверяем что прибыль не упала
            if best is None or r.total_pnl >= best["total_pnl"]:
                best = row
                best_score = s
                _print_new_best(idx, row, r)

        if idx % print_every == 0 or idx == n_iter:
            _print_progress(idx, n_iter, time.time() - t0, results)

    # ── Финал ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    valid   = [r for r in results if r["score"] > -999]
    df_res  = pd.DataFrame(valid).sort_values("score", ascending=False)

    # Убираем колонку params из CSV (она содержит вложенные списки)
    save_cols = [c for c in df_res.columns if c != "params"]
    if save_to:
        df_res[save_cols].to_csv(save_to, index=False)
        print(f"\n  Результаты сохранены: {save_to}")

    _print_summary(df_res, top_n, elapsed)

    # Сохраняем полные параметры лучших в JSON
    if best:
        best_json = save_to.replace(".csv", "_best_params.json") if save_to else "best_params.json"
        top_params = [r["params"] for r in valid[:top_n] if "params" in r]
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(top_params, f, indent=2, ensure_ascii=False)
        print(f"  Лучшие параметры (JSON): {best_json}")

    return df_res, best


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные принтеры
# ─────────────────────────────────────────────────────────────────────────────

def _is_new_best(r: BacktestResult, best: dict) -> bool:
    """
    Новая стратегия попадает в топ ТОЛЬКО если:
      - PnL строго больше текущего лучшего
    Среднее время учитывается через score (штраф), но не как фильтр.
    """
    if best is None:
        return True
    return r.total_pnl > best["total_pnl"]


def _print_new_best(idx: int, row: dict, r: BacktestResult):
    print(f"  ★ [#{idx:>5}]  Score={row['score']:>10.4f}  "
          f"PnL={r.total_pnl:>8.2f}$  MaxDD={r.max_drawdown:>7.2f}$  "
          f"Trades={r.trade_count:>4}  AvgMin={r.avg_trade_minutes:>6.0f}  "
          f"MaxMin={r.max_trade_minutes:>7.0f}  "
          f"N={row['n_orders']:>2}  TP={row['tp_pct']:.2f}%  "
          f"steps={row['step_min']:.2f}..{row['step_max']:.2f}%  "
          f"sizes={row['size_min']:.0f}..{row['size_max']:.0f}$")
    # ── Последние 365 дней ────────────────────────────────────────────────
    if r.last_365_stats and r.last_365_stats.get('trades', 0) > 0:
        s = r.last_365_stats
        print(f"             └─ последние 365д ({s['period']}): "
              f"PnL={s['pnl']:>+8.2f}$  "
              f"Trades={s['trades']:>4}  "
              f"AvgMin={s['avg_minutes']:>6.0f}  "
              f"MaxMin={s.get('max_minutes', 0):>7.0f}")


def _print_progress(idx: int, total: int, elapsed: float, results: list):
    valid = [r for r in results if r["score"] > -999]
    pct   = idx / total * 100
    eta   = (elapsed / idx) * (total - idx) if idx > 0 else 0
    best  = max((r["score"] for r in valid), default=-999.0)
    print(f"  [{pct:>5.1f}%]  {idx}/{total}  "
          f"Валидных: {len(valid)}/{idx}  "
          f"Best score: {best:.4f}  "
          f"ETA: {eta/60:.1f}мин")


def _print_summary(df_res: pd.DataFrame, top_n: int, elapsed: float):
    print(f"\n{'═'*60}")
    print(f"  ИТОГ  |  Время: {elapsed/60:.1f} мин  |  Строк: {len(df_res)}")
    print(f"{'═'*60}")
    if df_res.empty:
        print("  Нет валидных результатов.")
        return

    display_cols = [
        "score", "total_pnl", "max_drawdown", "sharpe",
        "win_rate", "profit_factor", "trade_count", "avg_trade_minutes",
        "n_orders", "tp_pct", "step_min", "step_max", "step_mode",
        "size_min", "size_max", "size_mode",
    ]
    display_cols = [c for c in display_cols if c in df_res.columns]
    print(df_res[display_cols].head(top_n).to_string(index=False))
    print()