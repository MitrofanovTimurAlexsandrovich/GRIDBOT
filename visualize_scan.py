"""
visualize_scan.py
=================
Визуализирует набор параметров из grid_scan CSV по score.
Все настройки — в блоке CONFIG.
"""
from __future__ import annotations
import os, json
import pandas as pd
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

# Путь к CSV рыночных данных
FILE = "C:/Users/Madness/PycharmProjects/Crupto_Data_Joiner/Raw_Data/BYBIT_DOGEUSDT_LINEAR_2021_2026.csv"

# Путь к CSV результатов grid_scan
SCAN_CSV = ""   # пусто = авто из FILE: results/grid_scan_<SYMBOL>.csv

# Какой score визуализировать (ищет строку с ближайшим значением)
# None = показать таблицу топ-N и спросить
FIND_SCORE = None

# Показать только таблицу топ-N без бэктеста
LIST_ONLY  = False
LIST_TOP_N = 30

# Капитал и комиссия
CAPITAL    = 1000.0
COMMISSION = 0.0018
REINVEST   = False      # True = реинвестировать прибыль в каждую новую сетку
OUT_DIR    = "results"

# ═════════════════════════════════════════════════════════════════════════════


def load_data(path: str) -> pd.DataFrame:
    print(f"  Загрузка данных: {path}")
    df = pd.read_csv(path).iloc[:-1]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Строк: {len(df)}  |  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df


def list_csv(csv_path: str, top_n: int):
    df = pd.read_csv(csv_path)
    df_valid = df[df["score"] > -999].sort_values("score", ascending=False).reset_index()
    cols = ["index", "score", "total_pnl", "max_drawdown", "trade_count",
            "max_trade_minutes", "n_orders", "grid_pct", "tp_pct",
            "step_type", "step_pattern", "size_type"]
    cols = [c for c in cols if c in df_valid.columns]
    print(f"\n  Топ-{top_n} из {csv_path}  (валидных: {len(df_valid):,})\n")
    header = "  " + "  ".join(f"{c:>16}" for c in cols)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, row in df_valid.head(top_n).iterrows():
        vals = "  ".join(
            f"{row[c]:>16.4f}" if isinstance(row[c], float)
            else f"{row[c]:>16}"
            for c in cols
        )
        print(f"  {vals}")
    print()


def reconstruct_steps(row: pd.Series, n: int, grid_pct: float) -> list[float]:
    """
    Восстанавливает шаги из CSV строки.
    В grid_scan шаги генерируются случайно — точно восстановить нельзя,
    поэтому восстанавливаем приближённо по step_min, step_max, step_pattern.
    """
    step_min = float(row["step_min"])
    step_max = float(row["step_max"])
    pattern  = str(row.get("step_pattern", "uniform"))

    if pattern == "uniform" or step_min == step_max:
        return [round(grid_pct / n, 4)] * n

    if pattern == "asc":
        # линейно от step_min до step_max
        steps = np.linspace(step_min, step_max, n)
    elif pattern == "desc":
        steps = np.linspace(step_max, step_min, n)
    else:
        # mixed — равномерно, сохраняем хотя бы сумму
        steps = np.linspace(step_min, step_max, n)

    # нормируем чтобы сумма = grid_pct
    steps = steps / steps.sum() * grid_pct
    return [round(float(s), 4) for s in steps]


def reconstruct_sizes(row: pd.Series, n: int, capital: float) -> list[float]:
    """Восстанавливает размеры по size_type и size_min/size_max."""
    size_type = str(row.get("size_type", "linear"))
    if size_type == "linear":
        weights = [float(i + 1) for i in range(n)]
    elif size_type == "power":
        # восстанавливаем степень из min/max
        size_min = float(row["size_min"])
        size_max = float(row["size_max"])
        if size_min > 0 and size_max > size_min and n > 1:
            power = np.log(size_max / size_min) / np.log(n)
            power = max(1.0, min(4.0, power))
        else:
            power = 2.0
        weights = [(i + 1) ** power for i in range(n)]
    elif size_type == "geometric":
        size_min = float(row["size_min"])
        size_max = float(row["size_max"])
        if size_min > 0 and size_max > size_min and n > 1:
            r = (size_max / size_min) ** (1.0 / (n - 1))
            r = max(1.01, min(5.0, r))
        else:
            r = 1.5
        weights = [r ** i for i in range(n)]
    else:  # random_mono или неизвестный
        weights = [float(i + 1) for i in range(n)]

    total = sum(weights)
    return [round(w / total * capital, 4) for w in weights]


def run_and_plot(row: pd.Series, df_market: pd.DataFrame, symbol: str, score_tag: str):
    from grid_backtest import GridParams, run_grid_backtest
    from grid_visualizer import plot_best_result

    n        = int(row["n_orders"])
    grid_pct = float(row["grid_pct"])
    tp_pct   = float(row["tp_pct"])

    steps = reconstruct_steps(row, n, grid_pct)
    sizes = reconstruct_sizes(row, n, CAPITAL)

    params = {
        "n_orders":   n,
        "steps":      steps,
        "sizes":      sizes,
        "tp_pct":     tp_pct,
        "step_min":   float(row["step_min"]),
        "step_max":   float(row["step_max"]),
        "size_min":   float(row["size_min"]),
        "size_max":   float(row["size_max"]),
        "step_mode":  str(row.get("step_type", "?")),
        "size_mode":  str(row.get("size_type", "?")),
    }

    print(f"\n  Параметры:")
    print(f"  N={n}  TP={tp_pct:.2f}%  Grid={grid_pct}%")
    print(f"  step_type={params['step_mode']}  pattern={row.get('step_pattern','?')}  "
          f"steps={params['step_min']:.2f}..{params['step_max']:.2f}%")
    print(f"  size_type={params['size_mode']}  "
          f"sizes={params['size_min']:.0f}..{params['size_max']:.0f}$")
    print(f"  (steps/sizes восстановлены приближённо из CSV метаданных)")

    gp = GridParams(n_orders=n, steps=steps, sizes=sizes, tp_pct=tp_pct)
    print("\n  Запуск бэктеста...")
    r = run_grid_backtest(df_market, gp, commission=COMMISSION, initial_capital=CAPITAL, reinvest=REINVEST)

    print(f"\n  {'═'*52}")
    print(f"  PnL:           {r.total_pnl:+.2f} $")
    print(f"  Max Drawdown:  {r.max_drawdown:.2f} $")
    print(f"  Сделок:        {r.trade_count}")
    print(f"  Ср. время:     {r.avg_trade_minutes:.0f} мин  ({r.avg_trade_minutes/60:.1f} ч)")
    print(f"  Макс. время:   {r.max_trade_minutes:.0f} мин  ({r.max_trade_minutes/60:.1f} ч)")
    print(f"  Win Rate:      {r.win_rate:.1f}%")
    print(f"  Profit Factor: {r.profit_factor:.2f}")
    print(f"  Sharpe:        {r.sharpe:.3f}")
    print(f"  Score:         {r.score:.4f}  (в CSV: {float(row['score']):.4f})")
    print(f"  Реинвест:      {'ВКЛ ♻' if REINVEST else 'ВЫКЛ'}")

    if r.yearly_stats:
        print(f"\n  {'Год':>6}  {'PnL ($)':>10}  {'Сделок':>8}  {'Ср.вр':>8}  {'Макс.вр':>10}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
        for yr in sorted(r.yearly_stats):
            s = r.yearly_stats[yr]
            print(f"  {yr:>6}  {s['pnl']:>+10.2f}  {s['trades']:>8}  "
                  f"{s['avg_minutes']/60:>6.1f}ч  {s.get('max_minutes',0)/60:>8.1f}ч")
    print(f"  {'═'*52}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_p = os.path.join(OUT_DIR, f"grid_scan_viz_{symbol}_{score_tag}.png")
    plot_best_result(r, params, df_market, symbol=symbol, save_path=save_p)
    print(f"  График: {save_p}")


def main():
    symbol   = os.path.basename(FILE).replace(".csv", "")
    scan_csv = SCAN_CSV if SCAN_CSV else os.path.join(OUT_DIR, f"grid_scan_{symbol}.csv")

    if not os.path.exists(scan_csv):
        print(f"  CSV не найден: {scan_csv}")
        return

    if LIST_ONLY:
        list_csv(scan_csv, LIST_TOP_N)
        return

    df_scan = pd.read_csv(scan_csv)
    df_valid = df_scan[df_scan["score"] > -999].sort_values("score", ascending=False).reset_index(drop=True)
    if df_valid.empty:
        print("  Нет валидных строк в CSV")
        return

    # Определяем score для поиска
    find_score = FIND_SCORE
    if find_score is None:
        list_csv(scan_csv, LIST_TOP_N)
        try:
            val = input("  Введи score (или Enter для лучшего): ").strip()
            find_score = float(val) if val else df_valid["score"].iloc[0]
        except (ValueError, KeyboardInterrupt):
            print("  Отмена.")
            return

    # Находим строку с ближайшим score
    idx  = (df_valid["score"] - find_score).abs().idxmin()
    row  = df_valid.iloc[idx]
    diff = abs(row["score"] - find_score)
    print(f"\n  Найдена строка #{idx}  score={row['score']:.4f}"
          + (f"  (искали {find_score:.4f}, отклонение {diff:.4f})" if diff > 0.0001 else ""))

    score_tag = f"{row['score']:.4f}".replace(".", "_")
    df_market = load_data(FILE)
    run_and_plot(row, df_market, symbol, score_tag)


if __name__ == "__main__":
    main()