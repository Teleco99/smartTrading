# AI Coding Agent Instructions for DWX Trading System

## Project Overview
This is a **Python-based automated trading system** that simulates and executes trading strategies in real-time via MetaTrader 4/5. The system combines technical indicators (RSI, MACD), machine learning models (Neural Networks, Random Forest, Linear Regression), and live market connectivity through socket-based communication with MetaTrader Expert Advisors.

**Python 3.11 required** - incompatible with other versions.

## Architecture & Data Flow

### Core Layers
- **views/** - Streamlit UI with 5 pages: home, data download, optimization, simulation, MetaTrader connector
- **simulation/** - Backtesting engine: DataLoader → SimulationController → Strategy signal generation
- **analytics/** - Strategy logic and ML models: Indicators, Strategy, MultiOutputRegression, NeuralNetwork, RandomForest
- **darwinex/** - Live MetaTrader integration: DwxController manages socket communication via dwx_client
- **metrics/** - Performance calculation: ErrorMetrics (RMSE), Metrics (profitability, drawdown)

### Data Pipeline
1. **DataLoader** loads CSV (tab-delimited, format: `<DATE> <TIME> <CLOSE>`) → reindexes to uniform 5min intervals
2. **SimulationController** splits into train/test, applies indicators or trains ML models
3. **Strategy** generates buy/sell signals based on configured indicators or predictions
4. **Visualizer** plots results; **Metrics** calculates P&L
5. **DwxController** (live mode) receives tick/bar data from MetaTrader, applies same strategy in real-time

### Key Constraint: Time Filtering
Many components filter data to `horario_permitido` (trading hours like `('08:00', '16:30')`) using `.between_time()`.

## Critical Patterns & Conventions

### Signal Generation (Strategy.py)
All `generate_*_signals()` methods:
- Return DataFrame with 'Signal' column containing [1, -1, 0] (buy, sell, no-action)
- Use `.shift(1)` for lookback comparisons
- Handle NaN edge cases at index [0]

### Indicator Optimization
Each indicator has an `optimize_*()` static method that:
- Accepts `progress_callback(completed, status_msg)` for UI updates
- Uses `itertools.product()` to grid-search parameters
- Returns list of dicts ranked by Sharpe ratio or profitability
- Modifies data in-place (adds MACD, RSI, Prediction columns)

### ML Model Interface (MultiOutputRegression, NeuralNetwork, RandomForest)
- `optimize()` - grid search, returns best params
- `train(X, y)` - fits model with historical data
- `predict(X)` - returns ndarray of predictions
- Sampling rate parameter controls train window size: `actual_window = n_lookback * sampling_rate`

### Real-Time Data Management (DwxController)
- `_buffer` - circular deque (maxlen) of OHLC candles from MetaTrader
- `on_bar_data()` callback populates buffer and triggers strategy re-evaluation
- Maintains `operaciones[]` log of executed trades
- Symbol/timeframe filtering in callbacks prevents cross-symbol interference

## Common Workflows

### Running Simulation
```bash
# Terminal: set PYTHONPATH=. && streamlit run app.py
# UI: Load CSV → choose strategy (RSI/MACD/Regression/Neural) → set parameters
# SimulationController optimizes params on training set, applies to test set
```

### Adding New Strategy Type
1. Add method to `Strategy.generate_<name>_signals()` returning Signal column
2. Update `OptimizerController` with `optimize_<name>()` delegating to analytics model
3. Add UI selector in relevant view file

### Live Trading Workflow
1. Ensure MetaTrader EA (DWX_Server_MT5.mq5) compiled and running
2. Configure `mt5_path` in DwxController.__init__() to your MQL5/Files directory
3. Views/darwinexView.py instantiates DwxController, which auto-subscribes to bars
4. Incoming bars trigger `on_bar_data()` → signal generation → order placement via dwx_client

## File Conventions

### Data Format
- CSV input: `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL>` (tab-delimited)
- Index: DatetimeIndex after loading; all ops use index for time filtering
- Columns: Always include `<CLOSE>`, indicators add MACD, RSI, Prediction dynamically

### Optimization Output
- JSON in `optimizaciones/` named: `{symbol}_{timeframe}_{strategy}_params.json`
- Contains winning parameter sets from grid search

### Progress Callbacks
UI views pass `progress_callback(step, message)` to heavy computations for real-time feedback.

## Dependencies & Integration Points
- **streamlit** - UI framework (multi-page app)
- **pandas/numpy** - data manipulation and arrays
- **scikit-learn** - Random Forest, data splitting
- **tensorflow-cpu** - Neural Network training
- **yfinance** - historical data download
- **dwx_client** - socket communication to MetaTrader (vendored in darwinex/)

## Testing & Validation
- Unit tests in `darwinex/tests/` include symbol list, performance, multi-pair tests
- No explicit test runner configured; run individual test files via `python path/to/test.py`
- Verify MetaTrader connectivity by checking for new trades in `operaciones[]` log

## Common Pitfalls
1. **Missing MetaTrader EA** - DwxController will fail silently if EA not compiled/running
2. **Wrong Python version** - Only 3.11 works reliably with TensorFlow-cpu==2.12.0
3. **Time zone mismatch** - MetaTrader sends UTC times; filter logic expects local trading hours
4. **NaN propagation** - Always drop NaN after indicator calculation or add dropna() check
5. **Circular buffer overflow** - DwxController._buffer has fixed maxlen; oldest bars discarded automatically
