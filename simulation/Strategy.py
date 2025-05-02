from datetime import datetime

import pandas as pd

class Strategy:
    def __init__(self, capital_por_operacion=100, horario_permitido=('08:00', '17:00')):
        # Capital fijo invertido por operación
        self.capital_por_operacion = capital_por_operacion
        self.horario_permitido = horario_permitido
        self.largo_abierta = False  # Solo operaciones largas

    def generate_rsi_signals(self, data):
        '''Genera señales con RSI:
        Entrada si RSI cruza 30 hacia arriba (confirmado), salida si RSI > 70.
        '''

        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        for i in range(1, len(data) - 1):
            rsi_prev = data['RSI'].iloc[i - 1]
            rsi_now = data['RSI'].iloc[i]

            if rsi_prev < 30 and rsi_now >= 30:
                signals.iloc[i] = 1
            elif rsi_now > 70:
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals

    def generate_macd_signals(self, data):
        '''Genera señales con MACD:
        Entrada si MACD > Señal, salida si MACD < Señal.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        for i in range(1, len(data) - 1): 
            macd_prev = data['MACD'].iloc[i - 1]
            sig_prev = data['MACD_Signal'].iloc[i - 1]
            macd_now = data['MACD'].iloc[i]
            sig_now = data['MACD_Signal'].iloc[i]

            if macd_prev < sig_prev and macd_now > sig_now:
                signals.iloc[i] = 1
            elif macd_now < sig_now:
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals

    def generate_rsi_macd_signals(self, data):
        '''RSI + MACD:
        Entrada si RSI cruza 30 hacia arriba (confirmado) y MACD > señal.
        Salida si RSI > 70 o MACD < señal.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        for i in range(1, len(data) - 1):
            rsi_prev = data['RSI'].iloc[i - 1]
            rsi_now = data['RSI'].iloc[i]
            macd_now = data['MACD'].iloc[i]
            sig_now = data['MACD_Signal'].iloc[i]

            if rsi_prev < 30 and rsi_now >= 30 and macd_now > sig_now:
                signals.iloc[i] = 1
            elif rsi_now > 70 or macd_now < sig_now:
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals

    def generate_rsi_prediction_signals(self, data, horizon=3, spread=0.0005):
        '''RSI + Predicción:
        Entrada si RSI cruza 30 hacia arriba y predicción es alcista.
        Salida si RSI > 70.
        '''

        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        if 'Prediction' not in data.columns:
            raise ValueError("Falta la columna 'Prediction'.")

        for i in range(1, len(data) - horizon - 1):
            slope = data['Prediction'].iloc[i + horizon] - data['Prediction'].iloc[i]
            min_slope = data['<CLOSE>'].iloc[i] * spread

            rsi_prev = data['RSI'].iloc[i - 1]
            rsi_now = data['RSI'].iloc[i]

            if (slope > min_slope and rsi_prev < 30 and rsi_now >= 30):
                signals.iloc[i] = 1
            elif rsi_now > 70 and slope < -min_slope:
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals

    def generate_macd_prediction_signals(self, data, horizon=3, spread=0.0005):
        '''MACD + Predicción:
        Entrada si MACD > Señal y predicción es alcista.
        Salida si MACD < Señal.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        if 'Prediction' not in data.columns:
            raise ValueError("Falta la columna 'Prediction'.")

        for i in range(1, len(data) - horizon - 1):
            slope = data['Prediction'].iloc[i + horizon] - data['Prediction'].iloc[i]
            min_slope = data['<CLOSE>'].iloc[i] * spread

            macd_now = data['MACD'].iloc[i]
            macd_signal_now = data['MACD_Signal'].iloc[i]

            if slope > min_slope and macd_now > macd_signal_now:
                signals.iloc[i] = 1
            elif macd_now < macd_signal_now or slope < -min_slope:
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals

    def generate_rsi_macd_prediction_signals(self, data, horizon=3, spread=0.0005):
        '''RSI + MACD + Predicción:
        Entrada si:
            - RSI cruza 30 hacia arriba (confirmado)
            - MACD > señal
            - Predicción alcista
        Salida si RSI > 70 y MACD < señal.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        if 'Prediction' not in data.columns:
            raise ValueError("Falta la columna 'Prediction'.")

        for i in range(1, len(data) - horizon - 1):
            slope = data['Prediction'].iloc[i + horizon] - data['Prediction'].iloc[i]
            min_slope = data['<CLOSE>'].iloc[i] * spread

            rsi_prev = data['RSI'].iloc[i - 1]
            rsi_now = data['RSI'].iloc[i]

            macd = data['MACD'].iloc[i]
            macd_signal = data['MACD_Signal'].iloc[i]

            if (
                slope > min_slope and
                rsi_prev < 30 and rsi_now >= 30 and
                macd > macd_signal
            ):
                signals.iloc[i] = 1
            elif (rsi_now > 70 or 
                  macd < macd_signal or 
                  slope < -min_slope
            ):
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals

    def apply_strategy(self, data, signals, max_operaciones_abiertas=5, take_profit=0.02, stop_loss=0.01):
        '''Simula operaciones largas múltiples según señales, ejecutando en el siguiente precio disponible.'''

        operaciones = []
        abiertas = []

        signals['Operacion'] = 0

        condicion_abierta = False

        horario_permitido = self.horario_permitido

        for i in range(1, len(data) - 1):  # evitamos el último índice para poder usar i+1
            señal = signals['Signal'].iloc[i]

            precio_actual = data['<CLOSE>'].iloc[i]
            fecha_actual = data.index[i]

            # Verificar horario de la señal
            hora_actual = fecha_actual.time()
            hora_inicio, hora_fin = horario_permitido
            hora_inicio = datetime.strptime(hora_inicio, '%H:%M').time()
            hora_fin = datetime.strptime(hora_fin, '%H:%M').time()

            if not (hora_inicio <= hora_actual <= hora_fin):
                continue  # no operamos fuera de horario

            if señal == 1 and len(abiertas) < max_operaciones_abiertas and not condicion_abierta:
                cantidad_activos = self.capital_por_operacion / precio_actual

                abiertas.append({
                    'compra_fecha': fecha_actual,
                    'compra_precio': precio_actual,
                    'cantidad_activos': cantidad_activos,
                    'take_profit_price': precio_actual * (1 + take_profit),
                    'stop_loss_price': precio_actual * (1 - stop_loss)
                })

                signals.at[fecha_actual, 'Operacion'] = 1  

                condicion_abierta = True

            for operacion in abiertas[:]:
                stop_loss = operacion['stop_loss_price']
                take_profit = operacion['take_profit_price']
                compra_precio = operacion['compra_precio']
                cantidad = operacion['cantidad_activos']

                cerrar_por_sl_tp = precio_actual <= stop_loss or precio_actual >= take_profit
                cerrar_por_senal = señal == -1

                if cerrar_por_sl_tp or cerrar_por_senal:
                    ganancia = (precio_actual - compra_precio) * cantidad

                    operaciones.append({
                        'compra_fecha': operacion['compra_fecha'],
                        'compra_precio': compra_precio,
                        'venta_fecha': fecha_actual,
                        'venta_precio': precio_actual,
                        'cantidad_activos': cantidad,
                        'ganancia': ganancia
                    })

                    abiertas.remove(operacion)

                    signals.at[fecha_actual, 'Operacion'] = -1
                
                    condicion_abierta = False

            else:
                condicion_abierta = False

        return operaciones

