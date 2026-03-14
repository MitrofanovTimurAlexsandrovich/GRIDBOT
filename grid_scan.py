"""
grid_scan.py
============
Monte Carlo перебор пространства сеток.

Для каждой комбинации (N, grid_pct, tp_pct) генерируем случайные
распределения шагов и размеров из всего допустимого пространства:
  - шаги: любые значения >= MIN_STEP_PCT, сумма = grid_pct
  - размеры: монотонно нарастающие (каждый >= предыдущего), сумма = CAPITAL

Подходы генерации шагов (все равновероятны):
  uniform     — все равные
  arithmetic  — арифметическая прогрессия (случайный шаг d)
  geometric   — геометрическая прогрессия (случайный множитель r)
  random      — случайный разброс через Dirichlet

Подходы генерации размеров (все равновероятны):
  linear      — линейный рост
  power       — степенной рост (случайная степень 1..4)
  geometric   — геометрический рост (случайный r)
  random_mono — случайный монотонный (отсортированные случайные веса)
"""
from __future__ import annotations
import os, csv, time, random
import numpy as np
import pandas as pd
from grid_backtest import GridParams, run_grid_backtest

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
FILE       = "/BYBIT_DOGEUSDT_LINEAR_2021_2026.csv"# "C:/Users/Madness/PycharmProjects/Crupto_Data_Joiner/Raw_Data/BYBIT_DOGEUSDT_LINEAR_2021_2026.csv"
CAPITAL    = 1000.0
COMMISSION = 0.0018
OUT_DIR    = "results"

MIN_N = 5;  MAX_N = 30
TP_MIN = 0.5;  TP_MAX = 10.0;  TP_STEP = 0.1
GRID_MIN = 10;  GRID_MAX = 90;  GRID_STEP = 1
MIN_STEP_PCT = 0.1   # минимальный шаг между ордерами

# Сколько случайных наборов шагов+размеров пробовать
# на каждую комбинацию (N, grid_pct, tp_pct)
SAMPLES_PER_CELL = 10

SEED = None  # None = разные каждый раз, число = воспроизводимо
PRINT_EVERY = 2000
# ══════════════════════════════════════════════════════


# ── Генераторы шагов ──────────────────────────────────────────────────────────

def gen_steps_uniform(n: int, grid_pct: float) -> list[float] | None:
    step = grid_pct / n
    if step < MIN_STEP_PCT:
        return None
    return [round(step, 4)] * n


def gen_steps_arithmetic(n: int, grid_pct: float) -> list[float] | None:
    """a, a+d, a+2d, ..., a+(n-1)d  сумма=grid_pct  a>=MIN_STEP_PCT"""
    if n == 1:
        return gen_steps_uniform(1, grid_pct)
    # a = grid_pct/n - d*(n-1)/2
    # d случайный в диапазоне чтобы a >= MIN_STEP_PCT
    avg = grid_pct / n
    # max d при a = MIN_STEP_PCT: d = 2*(avg - MIN_STEP_PCT)/(n-1)
    max_d = 2 * (avg - MIN_STEP_PCT) / (n - 1)
    if max_d <= 0:
        return gen_steps_uniform(n, grid_pct)
    d = random.uniform(0, max_d)
    a = avg - d * (n - 1) / 2
    # случайно выбираем направление: asc или desc
    if random.random() < 0.5:
        steps = [a + d * i for i in range(n)]
    else:
        steps = [a + d * i for i in range(n - 1, -1, -1)]
    if any(s < MIN_STEP_PCT for s in steps):
        return None
    return [round(s, 4) for s in steps]


def gen_steps_geometric(n: int, grid_pct: float) -> list[float] | None:
    """геометрическая прогрессия с случайным коэффициентом r"""
    if n == 1:
        return gen_steps_uniform(1, grid_pct)
    # r от 0.5 до 3.0
    r = random.uniform(0.5, 3.0)
    if abs(r - 1.0) < 0.01:
        return gen_steps_uniform(n, grid_pct)
    geo_sum = sum(r ** i for i in range(n))
    a = grid_pct / geo_sum
    if random.random() < 0.5:
        steps = [a * (r ** i) for i in range(n)]
    else:
        steps = [a * (r ** i) for i in range(n - 1, -1, -1)]
    if any(s < MIN_STEP_PCT for s in steps):
        return None
    return [round(s, 4) for s in steps]


def gen_steps_random(n: int, grid_pct: float) -> list[float] | None:
    """случайный разброс: Dirichlet даёт равномерное покрытие симплекса"""
    alpha = [1.0] * n
    raw   = np.random.dirichlet(alpha)
    steps = raw * grid_pct
    # масштабируем чтобы минимум >= MIN_STEP_PCT
    min_s = steps.min()
    if min_s < MIN_STEP_PCT:
        # сдвигаем: вычитаем дефицит равномерно
        deficit = (MIN_STEP_PCT - min_s) * n
        if deficit >= grid_pct:
            return None
        steps = steps + (MIN_STEP_PCT - min_s)
        steps = steps / steps.sum() * grid_pct
    if any(s < MIN_STEP_PCT for s in steps):
        return None
    return [round(float(s), 4) for s in steps]


STEP_GENERATORS = [
    gen_steps_uniform,
    gen_steps_arithmetic,
    gen_steps_geometric,
    gen_steps_random,
]


# ── Генераторы размеров ───────────────────────────────────────────────────────

def gen_sizes_linear(n: int, capital: float) -> list[float]:
    weights = [float(i + 1) for i in range(n)]
    total   = sum(weights)
    return [round(w / total * capital, 4) for w in weights]


def gen_sizes_power(n: int, capital: float) -> list[float]:
    power   = random.uniform(1.0, 4.0)
    weights = [(i + 1) ** power for i in range(n)]
    total   = sum(weights)
    return [round(w / total * capital, 4) for w in weights]


def gen_sizes_geometric(n: int, capital: float) -> list[float]:
    r       = random.uniform(1.05, 3.0)
    weights = [r ** i for i in range(n)]
    total   = sum(weights)
    return [round(w / total * capital, 4) for w in weights]


def gen_sizes_random_mono(n: int, capital: float) -> list[float]:
    """случайные монотонно нарастающие веса"""
    raw     = sorted(random.uniform(0.5, 1.0) for _ in range(n))
    total   = sum(raw)
    return [round(w / total * capital, 4) for w in raw]


SIZE_GENERATORS = [
    gen_sizes_linear,
    gen_sizes_power,
    gen_sizes_geometric,
    gen_sizes_random_mono,
]


# ── Вспомогательные ──────────────────────────────────────────────────────────

def step_desc(steps: list[float]) -> str:
    if len(set(steps)) == 1:
        return "uniform"
    if all(steps[i] <= steps[i+1] for i in range(len(steps)-1)):
        return "asc"
    if all(steps[i] >= steps[i+1] for i in range(len(steps)-1)):
        return "desc"
    return "mixed"


def load_data(path: str) -> pd.DataFrame:
    print(f"  Загрузка: {path}")
    df = pd.read_csv(path).iloc[:-1]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Строк: {len(df)}  |  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df


def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    symbol   = os.path.basename(FILE).replace(".csv", "")
    out_path = os.path.join(OUT_DIR, f"grid_scan_{symbol}.csv")
    df       = load_data(FILE)

    tp_values   = [round(TP_MIN + i * TP_STEP, 2)
                   for i in range(round((TP_MAX - TP_MIN) / TP_STEP) + 1)]
    grid_values = list(range(GRID_MIN, GRID_MAX + GRID_STEP, GRID_STEP))
    n_values    = list(range(MIN_N, MAX_N + 1))

    # Оценка общего кол-ва итераций
    valid_cells = sum(
        1 for n in n_values for g in grid_values
        if g / n >= MIN_STEP_PCT
    )
    total = valid_cells * len(tp_values) * SAMPLES_PER_CELL
    print(f"\n  Ячеек (N×grid): {valid_cells:,}  × TP:{len(tp_values)} × samples:{SAMPLES_PER_CELL}")
    print(f"  Итого итераций: {total:,}  (~{total/600/60:.0f} ч при 600 iter/sec)")

    fieldnames = [
        "n_orders", "grid_pct", "tp_pct",
        "step_type", "size_type",
        "step_min", "step_max", "step_pattern",
        "size_min", "size_max",
        "score", "total_pnl", "max_drawdown",
        "trade_count", "avg_trade_minutes", "max_trade_minutes",
        "win_rate", "profit_factor", "sharpe",
        "last365_pnl", "last365_trades", "last365_avg_min", "last365_max_min",
    ]

    file_exists = os.path.exists(out_path)
    csv_file    = open(out_path, "a", newline="", encoding="utf-8")
    writer      = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    done = valid = 0
    best_score = -999.0
    t0 = time.time()

    print(f"\n{'═'*70}")
    print(f"  Grid Scanner (Monte Carlo)  |  Результаты: {out_path}")
    print(f"{'═'*70}\n")

    for n in n_values:
        for grid_pct in grid_values:
            if grid_pct / n < MIN_STEP_PCT:
                continue

            for tp in tp_values:
                for _ in range(SAMPLES_PER_CELL):
                    done += 1

                    # Выбираем случайный генератор шагов и размеров
                    step_gen = random.choice(STEP_GENERATORS)
                    size_gen = random.choice(SIZE_GENERATORS)

                    steps = step_gen(n, grid_pct)
                    if steps is None:
                        # fallback на uniform
                        steps = gen_steps_uniform(n, grid_pct)
                        if steps is None:
                            continue
                        step_gen = gen_steps_uniform

                    sizes = size_gen(n, CAPITAL)

                    gp = GridParams(n_orders=n, steps=steps,
                                    sizes=sizes, tp_pct=tp)
                    r  = run_grid_backtest(df, gp,
                                           commission=COMMISSION,
                                           initial_capital=CAPITAL)
                    s365 = r.last_365_stats

                    writer.writerow({
                        "n_orders":          n,
                        "grid_pct":          grid_pct,
                        "tp_pct":            tp,
                        "step_type":         step_gen.__name__.replace("gen_steps_", ""),
                        "size_type":         size_gen.__name__.replace("gen_sizes_", ""),
                        "step_min":          round(min(steps), 4),
                        "step_max":          round(max(steps), 4),
                        "step_pattern":      step_desc(steps),
                        "size_min":          round(min(sizes), 2),
                        "size_max":          round(max(sizes), 2),
                        "score":             round(r.score, 6),
                        "total_pnl":         round(r.total_pnl, 4),
                        "max_drawdown":      round(r.max_drawdown, 4),
                        "trade_count":       r.trade_count,
                        "avg_trade_minutes": round(r.avg_trade_minutes, 1),
                        "max_trade_minutes": round(r.max_trade_minutes, 1),
                        "win_rate":          round(r.win_rate, 2),
                        "profit_factor":     round(r.profit_factor, 4),
                        "sharpe":            round(r.sharpe, 4),
                        "last365_pnl":       round(s365.get("pnl", 0), 4),
                        "last365_trades":    s365.get("trades", 0),
                        "last365_avg_min":   round(s365.get("avg_minutes", 0), 1),
                        "last365_max_min":   round(s365.get("max_minutes", 0), 1),
                    })
                    csv_file.flush()

                    if r.score > -999:
                        valid += 1
                        if r.score > best_score:
                            best_score = r.score
                            print(
                                f"  ★ [{done:>9,}]  Score={r.score:>10.4f}  "
                                f"PnL={r.total_pnl:>8.2f}$  MaxDD={r.max_drawdown:>7.2f}$  "
                                f"Trades={r.trade_count:>5}  MaxMin={r.max_trade_minutes:>7.0f}  "
                                f"N={n:>2}  TP={tp:.1f}%  Grid={grid_pct}%  "
                                f"steps={step_gen.__name__.replace('gen_steps_','')}/"
                                f"{step_desc(steps)}  "
                                f"sizes={size_gen.__name__.replace('gen_sizes_','')}"
                            )

                    if done % PRINT_EVERY == 0:
                        elapsed = time.time() - t0
                        eta = (elapsed / done) * (total - done)
                        print(
                            f"  [{done/total*100:>5.1f}%] {done:>9,}/{total:,}  "
                            f"Валидных:{valid:,}  Best:{best_score:.4f}  "
                            f"ETA:{eta/3600:.1f}ч"
                        )

    csv_file.close()
    elapsed = time.time() - t0
    print(f"\n{'═'*70}")
    print(f"  Готово за {elapsed/3600:.1f} ч  |  Всего:{done:,}  Валидных:{valid:,}")
    print(f"  Лучший score: {best_score:.4f}  |  Результаты: {out_path}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()