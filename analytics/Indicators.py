import numpy as np
import pandas as pd
from itertools import product

class Indicators:

    @staticmethod
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        # Cálculo de la media exponencial rápida
        data['EMA_short'] = data['<CLOSE>'].ewm(span=short_window, adjust=False).mean()
        
        # Cálculo de la media exponencial lenta
        data['EMA_long'] = data['<CLOSE>'].ewm(span=long_window, adjust=False).mean()
        
        # Línea MACD: diferencia entre la EMA corta y la larga
        data['MACD'] = data['EMA_short'] - data['EMA_long']
        
        # Línea de señal: media exponencial de la línea MACD
        data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

    @staticmethod
    def optimize_macd(data, progress_callback, short_range=range(10, 16), long_range=range(24, 31), signal_range=range(7, 11), horizon=3):
        best_params = None
        max_score = float('-inf')
        total = sum(1 for _ in product(short_range, long_range, signal_range))
        completed = 0

        for short, long, signal in product(short_range, long_range, signal_range):
            if short >= long:
                continue

            temp = data.copy()
            Indicators.calculate_macd(temp, short_window=short, long_window=long, signal_window=signal)
            temp.dropna(inplace=True)

            if 'Interpolado' not in temp.columns:
                raise ValueError("Falta la columna 'Interpolado' en los datos.")

            temp['MACD_prev'] = temp['MACD'].shift(1)
            temp['MACD_Signal_prev'] = temp['MACD_Signal'].shift(1)
            temp['Position'] = 0
            position_counter = 0

            for i in range(len(temp) - horizon - 1):
                macd_prev = temp['MACD_prev'].iloc[i]
                macd_sig_prev = temp['MACD_Signal_prev'].iloc[i]
                macd_now = temp['MACD'].iloc[i]
                macd_sig_now = temp['MACD_Signal'].iloc[i]

                # Entrada: cruce MACD > Señal
                if macd_prev < macd_sig_prev and macd_now > macd_sig_now:
                    position_counter += 1  # abrir nueva posición
                    temp.at[temp.index[i], 'Position'] = position_counter

                # Salida: cruce MACD < Señal
                elif macd_prev > macd_sig_prev and macd_now < macd_sig_now and position_counter > 0:
                    position_counter -= 1  # cerrar una posición
                    temp.at[temp.index[i], 'Position'] = position_counter

                else:
                    temp.at[temp.index[i], 'Position'] = position_counter

            temp['ExitPrice'] = temp['<CLOSE>'].shift(-horizon)
            temp['FutureReturn'] = temp['ExitPrice'] / temp['<CLOSE>'] - 1
            temp['Strategy'] = (temp['Position'].shift(1) - temp['Position']) * temp['FutureReturn']

            valid_rows = temp['Interpolado'] != True
            score = (1 + temp.loc[valid_rows, 'Strategy']).prod() - 1

            print(f"MACD probado: short={short}, long={long}, signal={signal}, score={score:.4%}")

            if score > max_score:
                best_params = (short, long, signal)
                max_score = score

            completed += 1
            progress_callback(completed / total)

        print(f"🟢 Mejor MACD: Params={best_params}, Score total={max_score:.4%}")
        print(temp[(temp["Position"].diff() != 0) & (temp['Interpolado'] != True)][['<CLOSE>', 'ExitPrice', 'Position', 'FutureReturn', 'Strategy']])

        return best_params

    @staticmethod
    def calculate_rsi(data, window=14):
        # Diferencia entre precios consecutivos
        delta = data['<CLOSE>'].diff(1)
        
        # Separar ganancias y pérdidas
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Suavizado exponencial
        avg_gain = pd.Series(gain).ewm(span=window, min_periods=window).mean()
        avg_loss = pd.Series(loss).ewm(span=window, min_periods=window).mean()

        # Inicializar RSI
        data['RSI'] = np.nan
        for i in range(1, len(data)):
            avg_gain_i = avg_gain.iloc[i]
            avg_loss_i = avg_loss.iloc[i]
            rs_i = np.inf if avg_loss_i == 0 else avg_gain_i / avg_loss_i
            data.loc[data.index[i], 'RSI'] = 100 - (100 / (1 + rs_i))

    @staticmethod
    def optimize_rsi(data, progress_callback, window_range=range(9, 15), horizon=3):
        best_window = None
        max_score = float('-inf')
        total = len(window_range)
        completed = 0

        for window in window_range:
            temp = data.copy()
            Indicators.calculate_rsi(temp, window=window)
            temp.dropna(inplace=True)

            if 'Interpolado' not in temp.columns:
                raise ValueError("Falta la columna 'Interpolado' en los datos.")

            temp['RSI_prev'] = temp['RSI'].shift(1)
            temp['Position'] = 0
            position_counter = 0

            for i in range(1, len(temp) - 1):  # -1 porque miramos i+1
                rsi_prev = temp['RSI_prev'].iloc[i]
                rsi_now = temp['RSI'].iloc[i]

                # Entrada: cruce confirmado de sobreventa (30)
                if rsi_prev < 30 and rsi_now >= 30:
                    position_counter += 1  # abrir nueva operación
                    temp.at[temp.index[i], 'Position'] = position_counter

                # Salida: RSI > 70
                elif rsi_now > 70 and position_counter > 0:
                    position_counter -= 1  # cerrar una operación
                    temp.at[temp.index[i], 'Position'] = position_counter

                else:
                    temp.at[temp.index[i], 'Position'] = position_counter

            # Estrategia sobre posiciones
            temp['ExitPrice'] = temp['<CLOSE>'].shift(-horizon)
            temp['FutureReturn'] = temp['ExitPrice'] / temp['<CLOSE>'] - 1
            temp['Strategy'] = (temp['Position'].shift(1) - temp['Position']) * temp['FutureReturn']

            valid_rows = temp['Interpolado'] != True
            score = (1 + temp.loc[valid_rows, 'Strategy']).prod() - 1

            print(f"RSI probado: window={window}, score={score:.4%}")

            if score > max_score:
                best_window = window
                max_score = score

            completed += 1
            progress_callback(completed / total)

        print(f"🟢 Mejor RSI: window={best_window}, Score={max_score:.4%}")
        print(temp[(temp["Position"].diff() != 0) & valid_rows][['<CLOSE>', 'ExitPrice', 'Position', 'FutureReturn', 'Strategy']])

        return best_window
