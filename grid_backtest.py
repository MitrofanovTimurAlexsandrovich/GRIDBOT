"""
grid_backtest.py
================
Движок бэктеста для лонговой сетки ордеров.

Логика:
  1. Открывается сетка из N лимитных лонг-ордеров ниже текущей цены.
  2. Каждый ордер имеет индивидуальный шаг (% от цены входа) и размер (USDT).
  3. Когда цена касается уровня ордера — он исполняется (добавляется к позиции).
  4. TP рассчитывается от средневзвешенной цены входа всей позиции.
  5. При достижении TP — позиция закрывается, сетка перезапускается.
  6. Торгуем на 1-минутных свечах для максимальной точности.

Метрики оценки:
  - Максимальный PnL
  - Минимальная MaxDrawdown
  - Минимальное среднее время в сделке (в минутах)

Score = PnL/|MaxDD| * (1 / avg_trade_minutes)  — комбинированная метрика
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Параметры сетки
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GridParams:
    """Полное описание одной конфигурации сетки."""
    n_orders:    int               # кол-во ордеров (1..40)
    steps:       List[float]       # шаги каждого ордера в % (длина = n_orders)
    sizes:       List[float]       # размер каждого ордера в USDT (длина = n_orders)
    tp_pct:      float             # % тейк-профита от avg_entry
    min_contract: float = 0.0     # минимальный лот в монетах (0 = без ограничений)

    def validate(self) -> bool:
        if self.n_orders < 1 or self.n_orders > 40:
            return False
        if len(self.steps) != self.n_orders:
            return False
        if len(self.sizes) != self.n_orders:
            return False
        if any(s <= 0 for s in self.steps):
            return False
        if any(s <= 0 for s in self.sizes):
            return False
        if self.tp_pct <= 0:
            return False
        return True

    def min_viable_price(self) -> float:
        """
        Минимальная цена актива при которой хотя бы первый ордер
        исполнится (qty >= min_contract).
        Используется для ранней отбраковки конфигураций.
        """
        if self.min_contract <= 0:
            return 0.0
        # Первый ордер (рыночный) — самый маленький
        return self.sizes[0] / self.min_contract

    def count_viable_orders(self, price: float) -> int:
        """
        Считает сколько ордеров дадут qty >= min_contract при данной цене.
        Используется для отбраковки: если мало ордеров — конфиг бесполезен.
        """
        if self.min_contract <= 0 or price <= 0:
            return self.n_orders
        import math
        count = 0
        for size in self.sizes:
            qty = math.floor(size / price / self.min_contract) * self.min_contract
            if qty >= self.min_contract:
                count += 1
        return count


# ─────────────────────────────────────────────────────────────────────────────
# Результаты бэктеста
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    total_pnl:         float = 0.0
    max_drawdown:      float = 0.0
    max_dd_bar:        int   = 0     # индекс бара с максимальной просадкой
    max_orders_filled: int   = 0     # макс. кол-во ордеров сетки задействованных одновременно
    trade_count:       int   = 0
    avg_trade_minutes: float = 0.0
    max_trade_minutes: float = 0.0
    win_rate:          float = 0.0
    profit_factor:     float = 0.0
    sharpe:            float = 0.0
    score:             float = -999.0
    initial_capital:   float = 1000.0
    reinvest:          bool  = False
    equity_curve:      list  = field(default_factory=list)   # (timestamp, equity)
    trade_log:         list  = field(default_factory=list)   # список dict по каждой сделке
    yearly_stats:      dict  = field(default_factory=dict)   # статистика по годам
    last_year_stats:   dict  = field(default_factory=dict)   # последние 365 дней
    last_365_stats:    dict  = field(default_factory=dict)   # последние 365 дней


# ─────────────────────────────────────────────────────────────────────────────
# Основной бэктест
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_backtest(
    df:         pd.DataFrame,
    params:     GridParams,
    commission: float = 0.0018,   # 0.18% — taker fee Bybit
    initial_capital: float = 1000.0,
    reinvest: bool = False,        # реинвестировать прибыль в каждую новую сетку
    min_contract: float = 0.0,    # минимальный лот в монетах (0 = без ограничений)
) -> BacktestResult:
    """
    Запускает бэктест лонговой сетки на 1-минутных данных.

    df должен содержать колонки: timestamp, open, high, low, close
    """

    if not params.validate():
        return BacktestResult()

    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    n_bars  = len(df)

    # ── Накопленные метрики ───────────────────────────────────────────────────
    equity          = initial_capital
    peak_equity     = initial_capital
    max_dd          = 0.0
    max_dd_bar      = 0     # бар с максимальной просадкой
    max_orders_hit  = 0     # максимально задействованных ордеров одновременно
    closed_pnl      = 0.0
    trade_log       = []
    equity_curve    = []

    # ── Состояние сетки ───────────────────────────────────────────────────────
    in_position     = False
    grid_levels     = []    # абсолютные цены уровней
    order_filled    = []    # bool: исполнен ли ордер i
    position_qty    = 0.0   # суммарный объём в монетах
    position_cost   = 0.0   # суммарная стоимость входов (для avg_entry)
    tp_price        = 0.0
    first_fill_bar  = -1    # бар когда ПЕРВЫЙ ордер реально исполнился

    def _setup_grid(entry_price: float) -> Tuple[list, list, float]:
        """Рассчитывает уровни сетки и TP от текущей цены."""
        levels = []
        cumulative_step = 0.0
        for i in range(params.n_orders):
            cumulative_step += params.steps[i]
            level = entry_price * (1.0 - cumulative_step / 100.0)
            levels.append(level)
        return levels

    def _calc_tp(avg_entry: float) -> float:
        return avg_entry * (1.0 + params.tp_pct / 100.0)

    # Хелпер: округляет qty ВНИЗ до кратного min_contract
    # floor — чтобы не тратить больше денег чем запланировано
    # Если qty < min_contract — возвращает 0 (ордер пропускается)
    _min_lot = min_contract if min_contract > 0 else 0.0
    def _round_qty(qty: float) -> float:
        if _min_lot <= 0:
            return qty
        import math
        return math.floor(qty / _min_lot) * _min_lot

    # ── Главный цикл ──────────────────────────────────────────────────────────
    i = 0
    while i < n_bars:
        bar_low   = lows[i]
        bar_high  = highs[i]
        bar_close = closes[i]

        # ── Нет позиции: выставляем сетку от текущей цены ───────────────────
        if not in_position:
            ref_price    = bar_close
            grid_levels  = _setup_grid(ref_price)
            order_filled = [False] * params.n_orders
            position_qty  = 0.0
            position_cost = 0.0
            tp_price      = 0.0
            first_fill_bar = i    # ордер 0 — рыночный, сразу на этом баре
            in_position   = True

            # Нормируем sizes: при reinvest=True используем текущий equity
            total_sizes    = sum(params.sizes)
            grid_capital   = equity if reinvest else initial_capital
            scaled_sizes   = [s / total_sizes * grid_capital for s in params.sizes]

            # ── Ордер 0: рыночное исполнение по текущей цене ─────────────────
            market_qty     = _round_qty(scaled_sizes[0] / bar_close)
            if market_qty > 0:
                market_cost    = market_qty * bar_close
                market_fee     = market_cost * commission
                position_qty  += market_qty
                position_cost += market_cost + market_fee
                equity        -= (market_cost - scaled_sizes[0])  # компенсируем разницу от округления
                equity        -= market_fee
            order_filled[0] = True  # помечаем исполненным даже если qty=0 (слишком мал)

        # ── Проверяем исполнение лимитных ордеров 1..N (цена падает к уровню) ─
        if in_position:
            for j in range(1, params.n_orders):  # j=0 уже исполнен рыночно
                if not order_filled[j] and bar_low <= grid_levels[j]:
                    fill_price = grid_levels[j]
                    qty        = _round_qty(scaled_sizes[j] / fill_price)
                    order_filled[j] = True
                    if qty <= 0:
                        continue
                    real_cost      = qty * fill_price
                    fee            = real_cost * commission
                    position_qty  += qty
                    position_cost += real_cost + fee
                    equity        -= fee

            # Пересчитываем TP и проверяем — возможно несколько раз на одной свече
            while in_position and position_qty > 0:
                avg_entry = position_cost / position_qty
                tp_price  = _calc_tp(avg_entry)

                if bar_high < tp_price:
                    break  # TP не достигнут — выходим из while

                # ── Закрываем позицию по TP ───────────────────────────────────
                close_value = position_qty * tp_price
                fee_close   = close_value * commission
                trade_pnl   = close_value - position_cost - fee_close
                closed_pnl += trade_pnl
                equity     += trade_pnl

                duration_min = i - first_fill_bar

                orders_now = sum(order_filled)
                if orders_now > max_orders_hit:
                    max_orders_hit = orders_now
                trade_log.append({
                    "open_bar":    first_fill_bar,
                    "close_bar":   i,
                    "duration":    duration_min,
                    "orders_hit":  orders_now,
                    "avg_entry":   position_cost / position_qty,
                    "tp_price":    tp_price,
                    "pnl":         trade_pnl,
                    "timestamp":   df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
                    "open_ts":     df["timestamp"].iloc[first_fill_bar] if "timestamp" in df.columns else first_fill_bar,
                })

                if equity > peak_equity:
                    peak_equity = equity
                dd = peak_equity - equity
                if dd > max_dd:
                    max_dd = dd

                # ── Немедленно переоткрываем сетку по цене закрытия (tp_price) ─
                # на той же свече — без перехода к следующему бару
                reopen_price  = tp_price
                grid_levels   = _setup_grid(reopen_price)
                order_filled  = [False] * params.n_orders
                position_qty  = 0.0
                position_cost = 0.0
                first_fill_bar = i

                total_sizes  = sum(params.sizes)
                grid_capital = equity if reinvest else initial_capital
                scaled_sizes = [s / total_sizes * grid_capital for s in params.sizes]

                # Ордер 0 — рыночный по цене переоткрытия (= tp_price)
                market_qty = _round_qty(scaled_sizes[0] / reopen_price)
                if market_qty > 0:
                    market_cost    = market_qty * reopen_price
                    market_fee     = market_cost * commission
                    position_qty  += market_qty
                    position_cost += market_cost + market_fee
                    equity        -= (market_cost - scaled_sizes[0])
                    equity        -= market_fee
                order_filled[0] = True

                # Остальные лимитные ордера на этой же свече уже не могут
                # исполниться — цена переоткрытия выше, bar_low уже известен.
                # Проверяем только те уровни которые ниже bar_low (не коснулись)
                for j in range(1, params.n_orders):
                    if bar_low <= grid_levels[j]:
                        fill_price = grid_levels[j]
                        qty        = _round_qty(scaled_sizes[j] / fill_price)
                        order_filled[j] = True
                        if qty <= 0:
                            continue
                        real_cost      = qty * fill_price
                        fee            = real_cost * commission
                        position_qty  += qty
                        position_cost += real_cost + fee
                        equity        -= fee

                # Цикл while продолжается — проверяем TP снова на том же bar_high

        # Equity curve — каждый бар по bar_low (честная просадка)
        mtm_low = equity
        if in_position and position_qty > 0:
            mtm_low += position_qty * bar_low - position_cost
        equity_curve.append((
            df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
            mtm_low
        ))

        # Глобальная просадка — по bar_low (реальный худший момент свечи)
        mtm_equity = equity
        if in_position and position_qty > 0:
            mtm_equity += position_qty * bar_low - position_cost

        if mtm_equity > peak_equity:
            peak_equity = mtm_equity
        dd = peak_equity - mtm_equity
        if dd > max_dd:
            max_dd     = dd
            max_dd_bar = i

        i += 1

    # ── Принудительное закрытие незакрытой позиции на последней свече ─────────
    # Только для статистики времени — НЕ влияет на PnL и Score
    if in_position and position_qty > 0:
        last_bar   = len(df) - 1
        last_price = float(df["close"].iloc[last_bar])
        last_ts    = df["timestamp"].iloc[last_bar] if "timestamp" in df.columns else last_bar

        # Считаем нереализованный PnL только для отображения — не прибавляем к closed_pnl
        close_value     = position_qty * last_price
        fee_close       = close_value * commission
        unrealized_pnl  = close_value - position_cost - fee_close
        duration_min    = last_bar - first_fill_bar

        trade_log.append({
            "open_bar":       first_fill_bar,
            "close_bar":      last_bar,
            "duration":       duration_min,
            "orders_hit":     sum(order_filled),
            "avg_entry":      position_cost / position_qty,
            "tp_price":       None,
            "pnl":            unrealized_pnl,  # нереализованный — не в closed_pnl
            "timestamp":      last_ts,
            "open_ts":        df["timestamp"].iloc[first_fill_bar] if "timestamp" in df.columns else first_fill_bar,
            "forced":         True,
            "unrealized":     True,            # маркер: не входит в total_pnl
        })
        # closed_pnl и equity НЕ обновляются — форсированная сделка не в PnL
        orders_now = sum(order_filled)
        if orders_now > max_orders_hit:
            max_orders_hit = orders_now

    # ── Вычисляем итоговые метрики ────────────────────────────────────────────
    result = BacktestResult()
    result.initial_capital = initial_capital
    result.reinvest        = reinvest
    result.equity_curve = equity_curve
    result.trade_log    = trade_log
    result.total_pnl    = closed_pnl
    result.max_drawdown = max_dd
    result.max_dd_bar       = max_dd_bar
    result.max_orders_filled = max_orders_hit
    result.trade_count  = len(trade_log)

    # Для статистики используем только реальные (закрытые по TP) сделки
    closed_trades = [t for t in trade_log if not t.get("unrealized", False)]
    result.trade_count = len(closed_trades)  # переопределяем: только TP-сделки

    # Все сделки включая форсированную — для замера времени
    all_trades_for_time = trade_log  # включает unrealized для честного MaxMin

    if closed_trades:
        pnls      = [t["pnl"] for t in closed_trades]
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p <= 0]
        durations_closed = [t["duration"] for t in closed_trades]

        result.win_rate      = len(wins) / len(pnls) * 100
        result.avg_trade_minutes = float(np.mean(durations_closed))

        # max_trade_minutes — по ВСЕМ сделкам включая незакрытую форсированную
        durations_all = [t["duration"] for t in all_trades_for_time]
        result.max_trade_minutes = float(np.max(durations_all))

        gross_profit = sum(wins)   if wins   else 0.0
        gross_loss   = abs(sum(losses)) if losses else 1e-9
        result.profit_factor = gross_profit / gross_loss

        if len(pnls) > 1:
            pnl_arr = np.array(pnls)
            result.sharpe = (pnl_arr.mean() / (pnl_arr.std() + 1e-9)) * np.sqrt(len(pnls))

    # ── Статистика по годам + последние 365 дней ────────────────────────
    if trade_log and 'timestamp' in df.columns:
        last_ts    = pd.to_datetime(df['timestamp'].iloc[-1])
        cutoff_365 = last_ts - pd.Timedelta(days=365)
        yearly = {}
        last_365 = {'pnl': 0.0, 'trades': 0, 'duration_sum': 0, 'max_minutes': 0}

        # PnL/trades/avg — только закрытые по TP
        # max_minutes — все включая форсированную (честное время зависания)
        closed_only = [t for t in trade_log if not t.get('unrealized', False)]
        forced_list = [t for t in trade_log if t.get('unrealized', False)]

        for t in closed_only:
            open_ts  = pd.to_datetime(t.get('open_ts', t['timestamp']))
            close_ts = pd.to_datetime(t['timestamp'])

            # Годовая статистика — по году ЗАКРЫТИЯ
            yr = close_ts.year
            if yr not in yearly:
                yearly[yr] = {'pnl': 0.0, 'trades': 0, 'duration_sum': 0, 'max_minutes': 0}
            yearly[yr]['pnl']          += t['pnl']
            yearly[yr]['trades']       += 1
            yearly[yr]['duration_sum'] += t['duration']
            yearly[yr]['max_minutes']   = max(yearly[yr]['max_minutes'], t['duration'])

            # last_365: закрылась внутри последних 365 дней
            if close_ts >= cutoff_365:
                last_365['pnl']          += t['pnl']
                last_365['trades']       += 1
                last_365['duration_sum'] += t['duration']
                last_365['max_minutes']   = max(last_365['max_minutes'], t['duration'])

        # Форсированная сделка — учитываем её duration в max_minutes
        # для yearly (год открытия) и last_365 (если активна в периоде)
        for t in forced_list:
            open_ts  = pd.to_datetime(t.get('open_ts', t['timestamp']))
            close_ts = pd.to_datetime(t['timestamp'])
            yr = open_ts.year
            if yr not in yearly:
                yearly[yr] = {'pnl': 0.0, 'trades': 0, 'duration_sum': 0, 'max_minutes': 0}
            yearly[yr]['max_minutes'] = max(yearly[yr]['max_minutes'], t['duration'])
            # В last_365 по max_minutes если сделка активна в периоде
            if open_ts <= last_ts and close_ts >= cutoff_365:
                last_365['max_minutes'] = max(last_365['max_minutes'], t['duration'])

        for yr in yearly:
            tr = yearly[yr]['trades']
            yearly[yr]['avg_minutes'] = yearly[yr]['duration_sum'] / tr if tr else 0
        tr365 = last_365['trades']
        last_365['avg_minutes'] = last_365['duration_sum'] / tr365 if tr365 else 0
        last_365['period'] = f"{cutoff_365.strftime('%Y-%m-%d')} — {last_ts.strftime('%Y-%m-%d')}"
        result.yearly_stats   = yearly
        result.last_365_stats = last_365
        result.last_year_stats = last_365  # алиас для совместимости

    result.score = compute_score(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Функция оценки (Score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(r: BacktestResult, min_trades: int = 5) -> float:
    """
    Score = survival * pnl * speed_bonus

    survival     = (initial_capital - MaxDD) / initial_capital
                   сколько капитала остаётся в худший момент (0..1)
                   чем ближе к 1 — тем лучше, при MaxDD >= capital = 0

    penalty_max = 1 / (1 + (max/avg - 1))^2 — прогрессирующий штраф за выброс
                  max==avg -> 0 штрафа | max=5×avg -> 0.04 | max=10×avg -> 0.01

    Score = PnL × penalty_max

    Возвращает -999 если сделок мало, PnL <= 0 или капитал уничтожен.
    """
    if r.trade_count < min_trades:
        return -999.0
    if r.total_pnl <= 0:
        return -999.0
    if r.last_365_stats.get("trades", 0) < 12:
        return -999.0

    # прогрессирующий штраф за выброс max vs avg
    ratio       = r.max_trade_minutes / max(r.avg_trade_minutes, 1.0)
    excess      = max(ratio - 1.0, 0.0)
    penalty_max = 1.0 / (1.0 + excess) ** 2

    score = r.total_pnl * penalty_max
    return round(score, 6)