"""
visualize_row.py
================
Визуализирует конкретный набор параметров из CSV результатов или JSON.
Все настройки задаются в блоке CONFIG ниже.
"""

from __future__ import annotations
import argparse, json, os
import pandas as pd
import numpy as np
import ast

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — всё настраивается здесь
# ═════════════════════════════════════════════════════════════════════════════

# Путь к CSV с рыночными данными (тот же файл что использовался при оптимизации)
FILE = "C:/Users/Madness/PycharmProjects/Crupto_Data_Joiner/Raw_Data/BYBIT_BTCUSDT_LINEAR_2020_2026.csv"

# Источник параметров — выбери один из двух вариантов:
# CSV: результаты оптимизации (шаги/размеры для random-режимов могут не совпасть)
# JSON: точные параметры (рекомендуется)
# Оставь пустым "" — тогда путь построится автоматически из имени FILE
CSV_PATH  = "C:/Users/Madness/PycharmProjects/GRIDBOT/results/grid_opt_BYBIT_DOGEUSDT_LINEAR_2021_2026.csv"
JSON_PATH = ""

# Какой источник использовать: "csv" или "json"
SOURCE = "csv"

# Номер строки/записи — None = скрипт спросит при запуске
ROW = None

# Поиск по score — если задан, игнорирует ROW и ищет строку с ближайшим score
# Пример: FIND_SCORE = 1679.6830  |  None = не используется
FIND_SCORE = 5.496077

# Показать таблицу топ-N строк CSV и выйти (True = только просмотр, без графика)
LIST_ONLY = False
LIST_TOP_N = 20

# Капитал и комиссия
CAPITAL    = 1000.0
COMMISSION = 0.0018
REINVEST   = True      # True = реинвестировать прибыль в каждую новую сетку

# Папка для сохранения графика
OUT_DIR = "results"

# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    print(f"  Загрузка данных: {file_path}")
    df = pd.read_csv(file_path).iloc[:-1]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Строк: {len(df)}  |  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df


def params_from_csv_row(csv_path: str, row_idx: int) -> dict:
    """
    Восстанавливает параметры из строки CSV.
    Автоматически определяет формат: grid_opt или grid_scan.
    """
    df   = pd.read_csv(csv_path)
    if row_idx >= len(df):
        raise ValueError(f"Строка {row_idx} не существует, в файле {len(df)} строк")
    row  = df.iloc[row_idx]
    cols = set(df.columns)

    n      = int(row["n_orders"])
    tp_pct = float(row["tp_pct"])

    # grid_scan CSV — перенаправляем пользователя
    if "grid_pct" in cols or ("step_type" in cols and "step_mode" not in cols):
        raise ValueError(
            "grid_scan CSV detected (has grid_pct / step_type columns).\n"
            "  Use visualize_scan.py for grid_scan results.\n"
            "  visualize_row.py works with grid_opt CSV or JSON."
        )

    # Неизвестный формат
    if "step_mode" not in cols or "size_mode" not in cols:
        raise ValueError(
            f"Unknown CSV format. Columns: {sorted(cols)}\n"
            "  Expected grid_opt CSV (step_mode, size_mode) or JSON (SOURCE='json')."
        )

    # grid_opt CSV
    step_min  = float(row["step_min"])
    step_max  = float(row["step_max"])
    step_mode = str(row["step_mode"])
    size_min  = float(row["size_min"])
    size_max  = float(row["size_max"])
    size_mode = str(row["size_mode"])

    from grid_optimizer import _generate_steps, _generate_sizes
    import random
    random.seed(42)
    steps = _generate_steps(n, step_min, step_max, step_mode)
    sizes = _generate_sizes(n, size_min, size_max, size_mode)

    if "random" in (step_mode, size_mode):
        print("  Режим random — шаги/размеры могут отличаться от оригинала.")
        print("     Используй JSON для точного воспроизведения.")

    return {
        "n_orders":  n,
        "steps":     steps,
        "sizes":     sizes,
        "tp_pct":    tp_pct,
        "step_mode": step_mode,
        "size_mode": size_mode,
        "step_min":  step_min,
        "step_max":  step_max,
        "size_min":  size_min,
        "size_max":  size_max,
    }


def params_from_json(json_path: str, row_idx: int) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if row_idx >= len(data):
        raise ValueError(f"Индекс {row_idx} не существует, в JSON {len(data)} записей")
    return data[row_idx]


def list_csv(csv_path: str, top_n: int = 20):
    df = pd.read_csv(csv_path)
    cols = ["score", "total_pnl", "max_drawdown", "trade_count",
            "avg_trade_minutes", "n_orders", "tp_pct",
            "step_min", "step_max", "step_mode", "size_mode"]
    cols = [c for c in cols if c in df.columns]
    print(f"\n  Топ-{top_n} строк из {csv_path}:\n")
    print(f"  {'#':>4}  " + "  ".join(f"{c:>14}" for c in cols))
    print(f"  {'─'*4}  " + "  ".join("─"*14 for _ in cols))
    for i, row in df.head(top_n).iterrows():
        vals = "  ".join(f"{row[c]:>14.4f}" if isinstance(row[c], float)
                         else f"{row[c]:>14}" for c in cols)
        print(f"  {i:>4}  {vals}")
    print()


def main():
    # Режим: просто показать список CSV
    if LIST_ONLY:
        if not CSV_PATH:
            print("  LIST_ONLY=True но CSV_PATH не задан")
            return
        list_csv(CSV_PATH, LIST_TOP_N)
        return

    # Определяем символ из имени файла данных
    symbol = os.path.basename(FILE).replace(".csv", "")

    # Авто-пути если не заданы явно
    symbol = os.path.basename(FILE).replace(".csv", "")
    csv_path  = CSV_PATH  if CSV_PATH  else os.path.join(OUT_DIR, f"grid_opt_{symbol}.csv")
    json_path = JSON_PATH if JSON_PATH else os.path.join(OUT_DIR, f"grid_opt_{symbol}_best_params.json")

    # Поиск по score — всегда ищет по CSV независимо от SOURCE
    row = ROW
    if FIND_SCORE is not None:
        if not csv_path or not os.path.exists(csv_path):
            print(f"  FIND_SCORE задан но CSV не найден: {csv_path}")
            return
        df_csv = pd.read_csv(csv_path)
        if "score" not in df_csv.columns:
            print("  Колонка score не найдена в CSV")
            return
        idx = (df_csv["score"] - FIND_SCORE).abs().idxmin()
        row = int(idx)
        print(f"  Найдена строка #{row} с score={df_csv.loc[idx, 'score']:.4f} "
              f"(искали {FIND_SCORE})")

    # Интерактивный ввод номера строки
    if row is None:
        if SOURCE == "csv" and csv_path:
            list_csv(csv_path, LIST_TOP_N)
        try:
            row = int(input("  Введи номер строки/записи: "))
        except (ValueError, KeyboardInterrupt):
            print("  Отмена.")
            return

    # Загружаем параметры
    if SOURCE == "json":
        params = params_from_json(json_path, row)
        source = f"{json_path} [#{row}]"
    else:
        params = params_from_csv_row(csv_path, row)
        source = f"{csv_path} [строка #{row}]"

    print(f"\n  Параметры из: {source}")
    print(f"  N={params['n_orders']}  TP={params['tp_pct']:.3f}%  "
          f"steps={params['step_min']:.2f}..{params['step_max']:.2f}%  "
          f"sizes={params['size_min']:.0f}..{params['size_max']:.0f}$  "
          f"step_mode={params['step_mode']}  size_mode={params['size_mode']}")

    # Загружаем данные и запускаем бэктест
    df = load_data(FILE)

    from grid_backtest import GridParams, run_grid_backtest
    from grid_visualizer import plot_best_result

    gp = GridParams(
        n_orders = params["n_orders"],
        steps    = params["steps"],
        sizes    = params["sizes"],
        tp_pct   = params["tp_pct"],
    )
    print("  Запуск бэктеста...")
    r = run_grid_backtest(df, gp, commission=COMMISSION, initial_capital=CAPITAL, reinvest=REINVEST)

    print(f"\n  {'═'*50}")
    print(f"  PnL:           {r.total_pnl:+.2f} $")
    print(f"  Max Drawdown:  {r.max_drawdown:.2f} $")
    print(f"  Сделок:        {r.trade_count}")
    print(f"  Ср. время:     {r.avg_trade_minutes:.0f} мин  ({r.avg_trade_minutes/60:.1f} ч)")
    print(f"  Макс. время:   {r.max_trade_minutes:.0f} мин  ({r.max_trade_minutes/60:.1f} ч)")
    print(f"  Win Rate:      {r.win_rate:.1f}%")
    print(f"  Profit Factor: {r.profit_factor:.2f}")
    print(f"  Sharpe:        {r.sharpe:.3f}")
    print(f"  Score:         {r.score:.4f}")
    print(f"  Реинвест:      {'ВКЛ ♻' if REINVEST else 'ВЫКЛ'}")

    if r.yearly_stats:
        print()
        print(f"  {'Год':>6}  {'PnL':>10}  {'Сделок':>8}  {'Ср.время':>10}  {'Макс.время':>12}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*12}")
        for yr in sorted(r.yearly_stats):
            s = r.yearly_stats[yr]
            print(f"  {yr:>6}  {s['pnl']:>+10.2f}  {s['trades']:>8}  "
                  f"{s['avg_minutes']/60:>8.1f}ч  {s.get('max_minutes',0)/60:>10.1f}ч")
    print(f"  {'═'*50}\n")

    # Сохраняем график
    os.makedirs(OUT_DIR, exist_ok=True)
    tag    = f"row{row}"
    save_p = os.path.join(OUT_DIR, f"grid_viz_{symbol}_{tag}.png")
    plot_best_result(r, params, df, symbol=symbol, save_path=save_p)
    print(f"  График: {save_p}")


if __name__ == "__main__":
    main()