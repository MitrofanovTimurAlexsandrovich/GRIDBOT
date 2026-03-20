"""
main_grid.py
============
Точка входа для оптимизации лонговой сетки.
Все настройки задаются в блоке CONFIG ниже — просто отредактируйте и запустите:

    python main_grid.py
"""

from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np

from grid_backtest import GridParams, run_grid_backtest
from grid_optimizer import optimize
from grid_visualizer import plot_best_result, plot_heatmap


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — всё настраивается здесь
# ═════════════════════════════════════════════════════════════════════════════

# Путь к CSV с 1-минутными данными (timestamp, open, high, low, close, volume)
FILE = "/BYBIT_DOGEUSDT_LINEAR_2021_2026.csv"

# Название монеты (используется в именах файлов и заголовках графиков)
# Оставь пустым "" — тогда возьмётся автоматически из имени файла
SYMBOL = ""

# Количество итераций оптимизации (больше = точнее, но дольше)
# ~500 итераций ≈ 1-2 мин | ~2000 ≈ 5-10 мин | ~5000 ≈ 20-30 мин
ITERS = 3000

# Начальный капитал в USDT
CAPITAL = 1000.0

# Комиссия Bybit (taker): 0.0006 = 0.06%
COMMISSION   = 0.0018   # 0.18% — taker fee Bybit
REINVEST     = False    # True = реинвестировать прибыль в каждую новую сетку

# Минимальный лот в монетах (0 = без ограничений)
# Пример: DOGE=1.0, BTC=0.001, ETH=0.01
MIN_CONTRACT = 1

# Список допустимых тейк-профитов (%).
# Только эти значения будут проверяться оптимизатором.
# Пример: TP_LIST = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
TP_LIST = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # 0.5..10.0 шаг 0.5

# Пространство поиска параметров сетки
SEARCH_SPACE = {
    "n_orders":  (5,   10),      # диапазон количества ордеров (целое)
    "step_min":  (0.1, 10),     # % минимальный шаг ордера
    "step_max":  (0.3, 10),     # % максимальный шаг ордера
    "size_min":  (5.0, 100),    # USDT минимальный размер ордера
    "size_max":  (20.0, 500.0),  # USDT максимальный размер ордера
    # tp_pct не нужен — используется TP_LIST выше
}

# Таймфрейм данных в минутах (1 = не ресемплировать, рекомендуется)
TF_MINUTES = 1

# Сколько лучших результатов выводить в таблице
TOP_N = 20

# Папка для сохранения результатов (CSV, JSON, PNG)
OUT_DIR = "results"

# Random seed для воспроизводимости (None = каждый раз новый прогон)
SEED = None

# Режим: "" = оптимизация | путь к JSON = запустить бэктест по готовым параметрам
# Пример: BACKTEST_JSON = "results/grid_opt_BTCUSDT_best_params.json"
BACKTEST_JSON = ""

# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка и подготовка данных
# ─────────────────────────────────────────────────────────────────────────────

def load_data(file_path: str, tf_minutes: int = 1) -> pd.DataFrame:
    """
    Загружает CSV, при необходимости ресемплирует.
    Ожидает колонки: timestamp, open, high, low, close, volume
    """
    print(f"  Загрузка данных: {file_path}")
    df = pd.read_csv(file_path).iloc[:-1]   # убираем последнюю незакрытую свечу

    # Нормализуем timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = {"open", "high", "low", "close"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")

    # Ресемплинг если нужен
    if tf_minutes > 1:
        print(f"  Ресемплинг на {tf_minutes}м...")
        df = _resample(df, tf_minutes)

    print(f"  Строк: {len(df)}  |  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}min"
    df = df.set_index("timestamp")
    agg = df.resample(rule, label="left", closed="left").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    })
    if "volume" in df.columns:
        agg["volume"] = df["volume"].resample(rule, label="left", closed="left").sum()
    return agg.dropna().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Одиночный бэктест по JSON параметрам
# ─────────────────────────────────────────────────────────────────────────────

def run_from_json(df: pd.DataFrame, json_path: str, symbol: str, out_dir: str,
                  capital: float, commission: float):
    """Запускает бэктест и визуализацию для параметров из JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        params_list = json.load(f)

    if isinstance(params_list, list):
        params = params_list[0]   # берём лучший
    else:
        params = params_list

    print(f"\n  Запуск бэктеста по параметрам из {json_path}...")
    gp = GridParams(
        n_orders = params["n_orders"],
        steps    = params["steps"],
        sizes    = params["sizes"],
        tp_pct   = params["tp_pct"],
    )
    result = run_grid_backtest(df, gp, commission=commission, initial_capital=capital)

    print(f"\n  {'─'*50}")
    print(f"  Результаты бэктеста:")
    print(f"  PnL:           {result.total_pnl:+.2f} $")
    print(f"  Max Drawdown:  {result.max_drawdown:.2f} $")
    print(f"  Сделок:        {result.trade_count}")
    print(f"  Ср. время:     {result.avg_trade_minutes:.0f} мин")
    print(f"  Win Rate:      {result.win_rate:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Sharpe:        {result.sharpe:.3f}")
    print(f"  Score:         {result.score:.4f}")
    print(f"  {'─'*50}\n")

    plot_best_result(
        result, params, df,
        symbol    = symbol,
        save_path = os.path.join(out_dir, f"grid_backtest_{symbol}.png"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # настройки читаются из блока CONFIG в начале файла

    # Символ из имени файла если не задан явно
    symbol = SYMBOL if SYMBOL else os.path.basename(FILE).replace(".csv", "")

    # Папка результатов
    os.makedirs(OUT_DIR, exist_ok=True)

    # Загрузка данных
    df = load_data(FILE, tf_minutes=TF_MINUTES)

    # ── Режим: одиночный бэктест по JSON ─────────────────────────────────────
    if BACKTEST_JSON:
        run_from_json(df, BACKTEST_JSON, symbol, OUT_DIR, CAPITAL, COMMISSION)
        return

    # ── Режим: оптимизация ────────────────────────────────────────────────────
    save_csv  = os.path.join(OUT_DIR, f"grid_opt_{symbol}.csv")
    save_json = os.path.join(OUT_DIR, f"grid_opt_{symbol}_best_params.json")

    df_results, best_row = optimize(
        df              = df,
        n_iter          = ITERS,
        commission      = COMMISSION,
        initial_capital = CAPITAL,
        save_to         = save_csv,
        top_n           = TOP_N,
        print_every     = 100,
        seed            = SEED,
        elite_frac      = 0.05,
        narrow_every    = 400,
        narrow_shrink   = 0.35,
        min_contract    = MIN_CONTRACT,
        tp_list         = TP_LIST,
        search_space    = SEARCH_SPACE,
    )

    if df_results.empty:
        print("  Нет валидных результатов.")
        return

    # ── Визуализация победителя (score↑ + pnl↑) ─────────────────────────────
    # best_row — тот кто победил по правилу отбора в оптимизаторе
    # если оптимизатор ничего не нашёл — берём топ по score из таблицы
    if best_row is None and not df_results.empty:
        best_row = df_results.iloc[0]
    best_params = best_row.get("params") if best_row is not None else None

    # Восстанавливаем params из JSON если нужно
    if best_params is None and os.path.exists(save_json):
        with open(save_json, "r") as f:
            all_params = json.load(f)
        best_params = all_params[0] if all_params else None

    if best_params:
        gp = GridParams(
            n_orders     = best_params["n_orders"],
            steps        = best_params["steps"],
            sizes        = best_params["sizes"],
            tp_pct       = best_params["tp_pct"],
            min_contract = MIN_CONTRACT,
        )
        print("\n  Строим финальный бэктест лучшей конфигурации...")
        best_result = run_grid_backtest(df, gp, commission=COMMISSION,
                                        initial_capital=CAPITAL, reinvest=REINVEST,
                                        min_contract=MIN_CONTRACT)

        plot_best_result(
            best_result, best_params, df,
            symbol    = symbol,
            save_path = os.path.join(OUT_DIR, f"grid_best_{symbol}.png"),
        )

        plot_heatmap(
            df_results,
            save_path = os.path.join(OUT_DIR, f"grid_heatmap_{symbol}.png"),
        )

        print(f"\n  {'═'*50}")
        print(f"  ЛУЧШАЯ КОНФИГУРАЦИЯ — {symbol}")
        print(f"  {'═'*50}")
        print(f"  N ордеров:     {best_params['n_orders']}")
        print(f"  TP:            {best_params['tp_pct']:.3f}%")
        print(f"  Шаги (мод.):   {best_params['step_mode']}  {best_params['step_min']:.3f}..{best_params['step_max']:.3f}%")
        print(f"  Размеры (мод.):{best_params['size_mode']}  {best_params['size_min']:.1f}..{best_params['size_max']:.1f}$")
        print()
        print(f"  PnL:           {best_result.total_pnl:+.2f} $")
        print(f"  Max Drawdown:  {best_result.max_drawdown:.2f} $")
        print(f"  Сделок:        {best_result.trade_count}")
        print(f"  Ср. время:     {best_result.avg_trade_minutes:.0f} мин")
        print(f"  Win Rate:      {best_result.win_rate:.1f}%")
        print(f"  Profit Factor: {best_result.profit_factor:.2f}")
        print(f"  Sharpe:        {best_result.sharpe:.3f}")
        print(f"  Score:         {best_result.score:.4f}")
        print(f"  {'═'*50}\n")

        # ── Статистика по годам ───────────────────────────────────────────
        if best_result.yearly_stats:
            print(f"  {'─'*50}")
            print(f"  {'Год':>6}  {'PnL ($)':>10}  {'Сделок':>8}  {'Ср. время':>12}")
            print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*12}")
            for yr in sorted(best_result.yearly_stats):
                s = best_result.yearly_stats[yr]
                avg_h = s['avg_minutes'] / 60
                print(f"  {yr:>6}  {s['pnl']:>+10.2f}  {s['trades']:>8}  {avg_h:>9.1f} ч")
            print(f"  {'─'*50}\n")

        # Печатаем полные шаги и размеры
        print(f"  Детальная сетка ({best_params['n_orders']} ордеров):")
        print(f"  {'Ордер':>6}  {'Шаг%':>8}  {'Кумул.%':>10}  {'Размер$':>10}  {'Цена (от 100$)':>16}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*16}")
        cum = 0.0
        for i, (s, sz) in enumerate(zip(best_params["steps"], best_params["sizes"]), 1):
            cum += s
            lvl = 100.0 * (1 - cum / 100)
            print(f"  {i:>6}  {s:>8.3f}  {cum:>10.3f}  {sz:>10.2f}  {lvl:>16.4f}")
        print()


if __name__ == "__main__":
    main()