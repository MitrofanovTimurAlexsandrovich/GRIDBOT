# Grid Long Strategy Optimizer

Оптимизатор лонговой сетки ордеров для крипто-торговли.

## Структура проекта

```
grid_trading/
├── grid_backtest.py      # Движок бэктеста
├── grid_optimizer.py     # Оптимизатор (Random Search + Elite Guided)
├── grid_visualizer.py    # Графики и отчёты
├── main_grid.py          # Точка входа (CLI)
└── requirements.txt
```

## Логика стратегии

1. Сетка из **N лимитных лонг-ордеров** выставляется ниже текущей цены
2. Каждый ордер имеет **индивидуальный шаг** (% от точки входа) и **индивидуальный размер** (USDT)
3. При касании уровня — ордер исполняется, добавляется к позиции
4. **TP** рассчитывается от средневзвешенной цены входа всей позиции
5. При достижении TP — позиция закрывается, сетка **перезапускается** от новой цены
6. Данные: **1-минутный таймфрейм** (максимальная точность)

## Параметры оптимизации

| Параметр | Диапазон | Описание |
|---|---|---|
| `n_orders` | 1..40 | Кол-во ордеров в сетке |
| `steps[i]` | 0.1%..8% каждый | Индивидуальный шаг i-го ордера |
| `sizes[i]` | 5$..500$ каждый | Индивидуальный размер i-го ордера |
| `tp_pct` | 0.3%..8% | % тейк-профита от avg_entry |

**Режимы шагов:** linear, geometric, random, front_heavy, back_heavy  
**Режимы размеров:** flat, linear_increase, geometric_increase, pyramid, random

## Критерий оценки (Score)

```
Score = (PnL / MaxDD) * speed_bonus * profit_factor_bonus
speed_bonus = 1 / log(avg_minutes + 2)
```

Максимизирует PnL, минимизирует MaxDD и среднее время в сделке.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск оптимизации

```bash
# Базовый запуск
python main_grid.py --file /path/to/BYBIT_BTCUSDT_LINEAR_2020_2026.csv

# Полный пример
python main_grid.py \
    --file /path/to/BYBIT_BTCUSDT_LINEAR_2020_2026.csv \
    --symbol BTCUSDT \
    --iters 3000 \
    --capital 1000 \
    --commission 0.0006 \
    --out results/BTCUSDT \
    --seed 42

# Для другой монеты
python main_grid.py \
    --file /path/to/BYBIT_ETHUSDT_LINEAR_2021_2026.csv \
    --symbol ETHUSDT \
    --iters 2000 \
    --capital 1000

# Ресемплинг (если хотите 5м вместо 1м для скорости)
python main_grid.py --file data.csv --tf 5

# Запустить бэктест по найденным параметрам
python main_grid.py \
    --file data.csv \
    --backtest results/BTCUSDT/grid_opt_BTCUSDT_best_params.json
```

## Ожидаемый вывод

```
  ★ [#  47]  Score=    3.2451  PnL=  128.34$  MaxDD=  12.45$  Trades=  87  AvgMin=  180  N=12  TP=1.20%  steps=0.30..1.80%  sizes=15..85$
  ★ [# 312]  Score=    4.1230  PnL=  247.81$  MaxDD=  18.23$  Trades= 143  AvgMin=  145  N= 8  TP=0.95%  steps=0.20..2.10%  sizes=20..120$
```

## Результаты

После завершения в папке `--out` будет:
- `grid_opt_{SYMBOL}.csv` — все результаты оптимизации
- `grid_opt_{SYMBOL}_best_params.json` — топ-20 конфигураций (полные параметры)
- `grid_best_{SYMBOL}.png` — детальный отчёт лучшей конфигурации
- `grid_heatmap_{SYMBOL}.png` — heatmap n_orders vs tp_pct

## Советы по настройке

- **Быстрый тест**: `--iters 500` (≈1 мин)
- **Нормальный прогон**: `--iters 2000` (≈5-10 мин)
- **Глубокий поиск**: `--iters 5000+`
- Каждые 400 итераций пространство поиска **автоматически сужается** вокруг лучших найденных параметров

## Формат входных данных

CSV с колонками: `timestamp, open, high, low, close, volume`

```
timestamp,open,high,low,close,volume
2020-01-01 00:00:00,7195.5,7196.0,7194.5,7195.0,1234.5
...
```
