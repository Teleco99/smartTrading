import numpy as np
import pandas as pd
from itertools import product

class Indicators:
    RISK_PCT = 0.005      # 0.5% de riesgo 
    RR = 2.0             # 2:1
    MAX_BARS = 5         # máximo 5 velas abiertas

    @staticmethod
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        # Recorta las filas sobre las que se va a calcular
        valid_data = data[data['<CLOSE>'].notna()]

        # Calcula columnas intermedias sobre los datos truncados
        ema_short = valid_data['<CLOSE>'].ewm(span=short_window, adjust=False).mean()
        ema_long = valid_data['<CLOSE>'].ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        macd_signal = macd.ewm(span=signal_window, adjust=False).mean()

        # Asigna al DataFrame original, solo en las filas afectadas
        data.loc[macd.index, 'MACD'] = macd
        data.loc[macd_signal.index, 'MACD_Signal'] = macd_signal

    @staticmethod
    def optimize_macd(data, progress_callback, short_range=[11, 13, 15], long_range=[25, 27, 29], signal_range=[7, 9, 11], verbose=False):
        max_score = float('-inf')
        total = sum(1 for _ in product(short_range, long_range, signal_range))
        completed = 0

        for short, long, signal in product(short_range, long_range, signal_range):
            temp = data.copy()
            Indicators.calculate_macd(temp, short_window=short, long_window=long, signal_window=signal)
            temp.dropna(inplace=True)

            if 'Interpolado' not in data.columns:
                temp['Interpolado'] = False

            temp['MACD_prev'] = temp['MACD'].shift(1)
            temp['MACD_Signal_prev'] = temp['MACD_Signal'].shift(1)
            temp['Entrada'] = 0

            # Detectar posibles entradas
            for i in range(len(temp) - 1):
                macd_prev = temp['MACD_prev'].iloc[i]
                sig_prev = temp['MACD_Signal_prev'].iloc[i]
                macd_now = temp['MACD'].iloc[i]
                sig_now = temp['MACD_Signal'].iloc[i]

                if macd_prev < sig_prev and macd_now > sig_now:
                    temp.at[temp.index[i], 'Entrada'] = 1
                else:
                    temp.at[temp.index[i], 'Entrada'] = 0

            # Simulación de trading real
            temp['TradeReturn'] = 0.0
            in_trade = False
            entry_price = 0
            stop_loss = 0
            take_profit = 0

            for i in range(len(temp) - 1):

                # Entrada real
                if temp['Entrada'].iloc[i] == 1 and not in_trade:
                    in_trade = True
                    entry_price = temp['<CLOSE>'].iloc[i]
                    stop_loss = entry_price * (1 - Indicators.RISK_PCT)
                    take_profit = entry_price * (1 + Indicators.RISK_PCT * Indicators.RR)
                    continue

                if not in_trade:
                    continue

                close = temp['<CLOSE>'].iloc[i]
                macd_now = temp['MACD'].iloc[i]
                sig_now = temp['MACD_Signal'].iloc[i]

                # Vender en Take Profit
                if close >= take_profit:
                    temp.at[temp.index[i], 'TradeReturn'] = (take_profit / entry_price) - 1
                    in_trade = False
                    continue

                # Vender en Stop Loss
                if close <= stop_loss:
                    temp.at[temp.index[i], 'TradeReturn'] = (stop_loss / entry_price) - 1
                    in_trade = False
                    continue

                # Vender en MACD sobrecomprado
                if macd_now < sig_now:
                    temp.at[temp.index[i], 'TradeReturn'] = (close / entry_price) - 1
                    in_trade = False
                    continue

                temp.at[temp.index[i], 'TradeReturn'] = 0.0

            valid_rows = temp['Interpolado'] != True
            score = (1 + temp.loc[valid_rows, 'TradeReturn']).prod() - 1

            if verbose: 
                print(f"MACD probado: short={short}, long={long}, signal={signal}, score={score:.2%}")

            if score >= max_score:
                best_params = {
                    'short': short,
                    'long': long,
                    'signal': signal,
                    'max_score': score * 100,   # almacenar como porcentaje
                }
                max_score = score

            completed += 1
            progress_callback(completed / total, "Optimizando MACD")

        print(f"🟢 Mejor MACD: Params={best_params}, Score total={max_score:.2%}")

        return best_params

    @staticmethod
    def calculate_rsi(data, window=14):
        valid_data = data['<CLOSE>'].dropna()

        # Calcular diferencia de precios
        delta = valid_data.diff()

        # Separar ganancias y pérdidas
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Suavizado exponencial
        avg_gain = gain.ewm(span=window, min_periods=window).mean()
        avg_loss = loss.ewm(span=window, min_periods=window).mean()

        # Calcular RS y RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Reindexar RSI al índice original de data, rellenando con NaN donde falte
        rsi_full = rsi.reindex(data.index)

        # Asignar a la columna RSI
        data['RSI'] = rsi_full

    @staticmethod
    def optimize_rsi(data, progress_callback, window_range=range(9, 19), verbose=False):
        max_score = float('-inf')
        total = len(window_range)
        completed = 0

        for window in window_range:
            temp = data.copy()
            Indicators.calculate_rsi(temp, window=window)
            temp.dropna(inplace=True)

            if 'Interpolado' not in data.columns:
                temp['Interpolado'] = False

            temp['RSI_prev'] = temp['RSI'].shift(1)
            temp['Entrada'] = 0

            for i in range(len(temp) - 1):  # -1 porque miramos i+1
                rsi_prev = temp['RSI_prev'].iloc[i]
                rsi_now = temp['RSI'].iloc[i]

                # Entrada: RSI_PREW < 30 and RSI_NOW >= 30
                if rsi_prev < 30 and rsi_now >= 30:
                    temp.at[temp.index[i], 'Entrada'] = 1
                else:
                    temp.at[temp.index[i], 'Entrada'] = 0

            temp['TradeReturn'] = 0.0
            in_trade = False
            entry_price = 0
            stop_loss = 0
            take_profit = 0

            for i in range(len(temp) - 1):
                # Detectar entrada
                if temp['Entrada'].iloc[i] == 1 and not in_trade:
                    in_trade = True
                    entry_price = temp['<CLOSE>'].iloc[i]
                    stop_loss = entry_price * (1 - Indicators.RISK_PCT)
                    take_profit = entry_price * (1 + Indicators.RISK_PCT * Indicators.RR)
                    continue

                if not in_trade:
                    continue

                # Datos de la siguiente vela
                close = temp['<CLOSE>'].iloc[i]
                rsi_now = temp['RSI'].iloc[i]

                # Vender en Take Profit
                if close >= take_profit:
                    exit_price = temp['<CLOSE>'].iloc[i]
                    temp.at[temp.index[i], 'TradeReturn'] = (take_profit / entry_price) - 1
                    in_trade = False
                    continue

                # Vender en Stop Loss
                if close <= stop_loss:
                    exit_price = temp['<CLOSE>'].iloc[i]
                    temp.at[temp.index[i], 'TradeReturn'] = (stop_loss / entry_price) - 1
                    in_trade = False
                    continue

                # Vender en RSI sobrecomprado
                if rsi_now > 70:
                    exit_price = temp['<CLOSE>'].iloc[i]
                    temp.at[temp.index[i], 'TradeReturn'] = (exit_price / entry_price) - 1
                    in_trade = False
                    continue

                temp.at[temp.index[i], 'TradeReturn'] = 0.0

            valid_rows = temp['Interpolado'] != True
            score = (1 + temp.loc[valid_rows, 'TradeReturn']).prod() - 1

            if verbose:
                print(f"RSI probado: window={window}, score={score:.2%}")

            if score >= max_score:
                best_params = {
                    'window': window,
                    'max_score': score * 100,   # almacenar como porcentaje
                }
                max_score = score

            completed += 1
            progress_callback(completed / total, "Optimizando RSI")

        print(f"🟢 Mejor RSI: {best_params}")

        return best_params
