from datetime import datetime

import pandas as pd

class Strategy:
    
    def __init__(self, capital_operacion=1000):
        # Capital fijo invertido por operación
        self.capital_operacion = capital_operacion
        self.condicion_abierta = False

    def generate_rsi_signals(self, data):
        '''Genera señales con RSI:
        Entrada si RSI cruza 30, salida si RSI cruza 70.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        rsi_prev = data['RSI'].shift(1)
        rsi_now  = data['RSI']

        for i in range(len(data)):
            if i == 0:
                continue

            # Entrada en sobrecompra
            if rsi_prev.iloc[i] < 30 and rsi_now.iloc[i] > 30:
                signals.iloc[i] = 1

            # Salida en sobreventa
            elif rsi_prev.iloc[i] > 70 and rsi_now.iloc[i] < 70:
                signals.iloc[i] = -1

        return signals

    def generate_macd_signals(self, data):
        '''Genera señales con MACD:
        Entrada si cruza MACD > Señal, salida si cruza MACD < Señal.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        macd_prev = data['MACD'].shift(1)
        sig_prev  = data['MACD_Signal'].shift(1)

        for i in range(len(data)): 
            if i == 0:
                continue

            macd_now = data['MACD'].iloc[i]
            sig_now  = data['MACD_Signal'].iloc[i]

            # Entrada en sobrecompra
            if macd_prev.iloc[i] < sig_prev.iloc[i] and macd_now > sig_now:
                signals.iloc[i] = 1

            # Salida en sobreventa
            elif macd_prev.iloc[i] > sig_prev.iloc[i] and macd_now < sig_now:
                signals.iloc[i] = -1

        return signals

    def generate_rsi_prediction_signals(self, data, sampling_rate=3, horizon=1):
        '''RSI + Predicción:
        Entrada si RSI cruza 30 hacia arriba y predicción es alcista.
        Salida si RSI > 70 o predicción es bajista.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        if 'Prediction' not in data.columns:
            raise ValueError("Falta la columna 'Prediction'.")
        
        total_offset = horizon * sampling_rate

        rsi_prev = data['RSI'].shift(1)
        rsi_now  = data['RSI']

        for i in range(len(data) - total_offset):
            future_idx = i + total_offset
            
            if pd.isna(data['Prediction'].iloc[i]) or pd.isna(data['Prediction'].iloc[future_idx]):
                continue  # Saltar si falta alguna predicción

            prediction = data['Prediction'].iloc[future_idx]
            current_price = data['<CLOSE>'].iloc[i]

            if (rsi_prev.iloc[i] < 30 and rsi_now.iloc[i] > 30) and prediction > current_price:
                signals.iloc[i] = 1
            elif (rsi_prev.iloc[i] > 70 and rsi_now.iloc[i] < 70) or prediction < current_price:
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals

    def generate_macd_prediction_signals(self, data, sampling_rate=3, horizon=1):
        '''MACD + Predicción:
        Entrada si MACD > Señal y predicción es alcista.
        Salida si MACD < Señal.
        '''
        signals = pd.DataFrame(index=data.index, columns=['Signal'])
        signals['Signal'] = 0

        if 'Prediction' not in data.columns:
            raise ValueError("Falta la columna 'Prediction'.")

        total_offset = horizon * sampling_rate

        macd_prev = data['MACD'].shift(1)
        sig_prev  = data['MACD_Signal'].shift(1)
        macd_now = data['MACD']
        sig_now  = data['MACD_Signal']
        
        for i in range(len(data) - total_offset):
            future_idx = i + total_offset

            if pd.isna(data['Prediction'].iloc[i]) or pd.isna(data['Prediction'].iloc[future_idx]):
                continue  # Saltar si falta alguna predicción

            prediction = data['Prediction'].iloc[future_idx]
            current_price = data['<CLOSE>'].iloc[i]

            if (macd_prev.iloc[i] < sig_prev.iloc[i] and macd_now.iloc[i] > sig_now.iloc[i]) and prediction > current_price:
                signals.iloc[i] = 1
            elif (macd_prev.iloc[i] > sig_prev.iloc[i] and macd_now.iloc[i] < sig_now.iloc[i]) or prediction < current_price:
                signals.iloc[i] = -1

        # Anular señales en días interpolados
        if 'Interpolado' in data.columns:
            signals.loc[data['Interpolado'] == True, 'Signal'] = 0

        return signals
    
    def apply_strategy(self, data, signals, num_operaciones_abiertas, max_operaciones_abiertas=1):
        """
        Evalúa la señal más reciente y decide la operación a realizar ("buy", "sell", None),
        anotándola en signals['Operacion'] y respetando el horario y el límite de operaciones abiertas.
        """

        operacion = None  # Valor por defecto

        if data.empty or signals.empty:
            return operacion

        # Último dato disponible
        ultima_fecha = signals.index[-1]
        señal = signals['Signal'].iloc[-1]
        hora_actual = ultima_fecha.time()

        if señal == 1 and num_operaciones_abiertas < max_operaciones_abiertas and not self.condicion_abierta:
            operacion = "buy"
            signals.at[ultima_fecha, 'Operacion'] = 1

            self.condicion_abierta = True
        elif señal == -1 and num_operaciones_abiertas > 0:
            operacion = "sell"
            signals.at[ultima_fecha, 'Operacion'] = -1

            self.condicion_abierta = False
        else:
            signals.at[ultima_fecha, 'Operacion'] = 0

        return operacion

    def simulate_strategy(self, data, signals, max_operaciones_abiertas=1):
        '''Simula operaciones largas múltiples según señales, ejecutando en el siguiente precio disponible.'''

        operaciones = []
        abiertas = []

        signals['Operacion'] = 0

        horizon = 1
        for i in range(len(data) - horizon):  
            señal = signals['Signal'].iloc[i]

            precio_actual = data['<CLOSE>'].iloc[i]
            fecha_actual = data.index[i]
          
            if señal == 1 and len(abiertas) < max_operaciones_abiertas:
                cantidad_activos = self.capital_operacion / precio_actual
                
                abiertas.append({
                    'compra_fecha': fecha_actual,
                    'compra_precio': precio_actual,
                    'cantidad_activos': cantidad_activos
                })

                signals.at[fecha_actual, 'Operacion'] = 1  

                self.condicion_abierta = True

            if (señal == -1 and len(abiertas) > 0) or (i == len(data) - horizon - 1):
                # cerrar todas las abiertas al precio_actual
                while abiertas:
                    operacion = abiertas.pop()
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

                signals.at[fecha_actual, 'Operacion'] = -1
                self.condicion_abierta = False

        return operaciones

