from metrics.Metrics import Metrics
from itertools import product

import pandas as pd

class Indicators:
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
    def optimize_macd(data, progress_callback, short_range=range(5, 25), long_range=range(15, 45), signal_range=range(5, 25), verbose=False):
        MAX_ABIERTAS = 1
        
        resultados = []

        total = sum(1 for _ in product(short_range, long_range, signal_range))
        completed = 0

        for short, long, signal in product(short_range, long_range, signal_range):
            temp = data.copy()
            Indicators.calculate_macd(temp, short_window=short, long_window=long, signal_window=signal)
            temp.dropna(inplace=True)

            temp['MACD_prev'] = temp['MACD'].shift(1)
            temp['MACD_Signal_prev'] = temp['MACD_Signal'].shift(1)

            capital_por_operacion = 100
            abiertas = []
            operaciones = []

            # Detectar posibles entradas
            for i in range(len(temp) - 1):
                macd_prev = temp['MACD_prev'].iloc[i]
                sig_prev = temp['MACD_Signal_prev'].iloc[i]
                macd_now = temp['MACD'].iloc[i]
                sig_now = temp['MACD_Signal'].iloc[i]

                # Entrada en sobreventa 
                if macd_prev < sig_prev and macd_now >= sig_now and len(abiertas) < MAX_ABIERTAS:                    
                    precio_actual = temp['<CLOSE>'].iloc[i]
                    fecha_actual = temp.index[i]
                    cantidad_activos = capital_por_operacion / precio_actual

                    abiertas.append({
                        'compra_fecha': fecha_actual,
                        'compra_precio': precio_actual,
                        'cantidad_activos': cantidad_activos
                    })

                    if verbose:
                        print(f"  ENTRADA  -> {fecha_actual} precio_actual={precio_actual:.5f} len(abiertas)={len(abiertas)}")

                # Salida en sobrecompra
                if macd_prev > sig_prev and macd_now <= sig_now and len(abiertas) > 0:
                    operacion = abiertas.pop()
                    precio_actual = temp['<CLOSE>'].iloc[i]   
                    fecha_actual = temp.index[i] 
                    
                    compra_precio = operacion['compra_precio']
                    cantidad = operacion['cantidad_activos']

                    ganancia = (precio_actual - compra_precio) * cantidad

                    operaciones.append({
                        'compra_fecha': operacion['compra_fecha'],
                        'compra_precio': compra_precio,
                        'venta_fecha': fecha_actual,
                        'venta_precio': precio_actual,
                        'cantidad_activos': cantidad,
                        'ganancia': ganancia
                    })

                    if verbose:
                        print(f"  SALIDA   -> {fecha_actual} precio_actual={precio_actual:.5f} ganancia={ganancia:.4f} len(operaciones)={len(operaciones)}")

            metricas = Metrics(operaciones)
            profit = metricas.profit_factor()
            numero_operaciones = metricas.numero_de_operaciones()
            return_factor = metricas.return_factor()
            ganancia_total = metricas.beneficio_neto()

            if verbose: 
                print(f"MACD probado: short={short}, long={long}, signal={signal}, profit_total={profit:.2}")

            resultados.append({
                'long': long,
                'short': short,
                'signal': signal,
                'profit': profit,
                'numero_operaciones': numero_operaciones,
                'return_factor': return_factor,
                'ganancia_total': ganancia_total
            })

            completed += 1
            progress_callback(completed / total, "Optimizando MACD")

        df = pd.DataFrame(resultados).sort_values('profit', ascending=False).reset_index(drop=True)

        return df

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
        MAX_ABIERTAS = 1

        resultados = []

        total = len(window_range)
        completed = 0

        for window in window_range:
            temp = data.copy()
            Indicators.calculate_rsi(temp, window=window)
            temp.dropna(inplace=True)

            temp['RSI_prev'] = temp['RSI'].shift(1)

            capital_por_operacion = 100
            abiertas = []
            operaciones = []

            for i in range(len(temp) - 1):  # -1 porque miramos i+1
                rsi_now = temp['RSI'].iloc[i]

                # Entrada en sobreventa (cuando salga de sobreventa)
                if rsi_now >= 30 and len(abiertas) < MAX_ABIERTAS:
                    precio_actual = temp['<CLOSE>'].iloc[i]
                    fecha_actual = temp.index[i]
                    cantidad_activos = capital_por_operacion / precio_actual

                    abiertas.append({
                        'compra_fecha': fecha_actual,
                        'compra_precio': precio_actual,
                        'cantidad_activos': cantidad_activos
                    })

                    if verbose:
                        print(f"  ENTRADA  -> {fecha_actual} precio_actual={precio_actual:.5f} len(abiertas)={len(abiertas)}")

                # Salida en sobrecompra (cuando salga de sobrecompra) si tenemos operación abierta
                if rsi_now <= 70 and len(abiertas) > 0:
                    operacion = abiertas.pop()
                    precio_actual = temp['<CLOSE>'].iloc[i]   
                    fecha_actual = temp.index[i] 
                    
                    compra_precio = operacion['compra_precio']
                    cantidad = operacion['cantidad_activos']

                    ganancia = (precio_actual - compra_precio) * cantidad

                    operaciones.append({
                        'compra_fecha': operacion['compra_fecha'],
                        'compra_precio': compra_precio,
                        'venta_fecha': fecha_actual,
                        'venta_precio': precio_actual,
                        'cantidad_activos': cantidad,
                        'ganancia': ganancia
                    })

                    if verbose:
                        print(f"  SALIDA   -> {fecha_actual} precio_actual={precio_actual:.5f} ganancia={ganancia:.4f} len(operaciones)={len(operaciones)}")

            metricas = Metrics(operaciones)
            profit = metricas.profit_factor()
            numero_operaciones = metricas.numero_de_operaciones()
            return_factor = metricas.return_factor()
            ganancia_total = metricas.beneficio_neto()

            resultados.append({
                'window': window,
                'profit': profit,
                'numero_operaciones': numero_operaciones,
                'return_factor': return_factor,
                'ganancia_total': ganancia_total
            })

            if verbose:
                print(f"RSI probado: window={window}, profit_total={profit:.2}")

            completed += 1
            progress_callback(completed / total, "Optimizando RSI")

        df = pd.DataFrame(resultados).sort_values('profit', ascending=False).reset_index(drop=True)

        return df
