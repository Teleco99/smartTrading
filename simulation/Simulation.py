from analytics.Indicators import Indicators 
from analytics.MultiOutputRegression import MultiOutputRegression
from analytics.NeuralNetwork import NeuralNetwork
from simulation.Strategy import Strategy 
from metrics.ErrorMetrics import ErrorMetrics
from sklearn.preprocessing import MinMaxScaler

class Simulation:
    def __init__(self, training_data, test_data, capital_por_operacion=1000):
        self.capital_por_operacion = capital_por_operacion
        self.training_data = training_data
        self.test_data = test_data
        self.signals = None

    def calculate_regresion(self, progress_callback):
        '''Calcula regresión lineal y guarda predicciones en dataframe'''

        X = []

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()
        training_data_daily = training_data_daily.between_time('08:00', '16:30')
        test_data_daily = test_data_daily.between_time('08:00', '16:30')

        window_optimo = MultiOutputRegression.optimize_window(training_data_daily, progress_callback)
        model = MultiOutputRegression(window=window_optimo)
        model.train(test_data_daily['<CLOSE>'])

        rango = len(test_data_daily) - model.window - model.horizon

        if rango <= 0:
            print("❌ No hay suficientes datos para backtesting.")
            return

        for i in range(rango):
            X.append(test_data_daily[i:i + model.window]['<CLOSE>'].values)  

        # Si no hay suficientes datos
        if len(X) == 0:
            return []

        # Calcular predicciones y guardar la de horizon
        y_pred = model.predict(X)
        y_pred_final = y_pred[:, -1]

        valid_indices = test_data_daily.index[model.window + model.horizon - 1:]

        # Ajustar si no coinciden exactamente 
        if len(valid_indices) > len(y_pred_final):
            valid_indices = valid_indices[:len(y_pred_final)]
        elif len(valid_indices) < len(y_pred_final):
            y_pred_final = y_pred_final[:len(valid_indices)]

        # Asignar las predicciones
        self.test_data.loc[valid_indices, 'Prediction'] = y_pred_final

    def calculate_pred_neuralNetwork(self, neuralNetwork_cb):

        horizon = 3

        # Crear modelo
        nn_model = NeuralNetwork()

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()
        training_data_daily = training_data_daily.between_time('08:00', '16:30')
        test_data_daily = test_data_daily.between_time('08:00', '16:30')

        # Escalar los datos
        scaler = MinMaxScaler()
        training_data_scaled = scaler.fit_transform(training_data_daily['<CLOSE>'].values.reshape(-1, 1)).flatten()
        test_data_scaled = scaler.transform(test_data_daily['<CLOSE>'].values.reshape(-1, 1)).flatten()

        # Optimizar hiperparámetros
        best_params = nn_model.optimize(data=training_data_daily, progress_callback=neuralNetwork_cb, scaler=scaler)

        if not best_params:
            neuralNetwork_cb(1, "❌ No hay suficientes datos para optimizar la red")

        # Preparar entradas y salidas sobre datos escalados
        X_train, y_train = nn_model.prepare_data(training_data_scaled)
        X_test, y_test = nn_model.prepare_data(test_data_scaled)

        # Entrenar la red
        nn_model.train(X_train, y_train)

        # Guardar predicciones reales
        y_pred_real_list = []
        y_test_real_list = []

        neuralNetwork_cb(1, "⚙️ Entrenando red neuronal")

        # Predecir y reentrenar con cada nuevo precio
        for i in range(len(X_test)):

            # 1. Predecir la variación de precio para esta muestra
            y_pred_change = nn_model.predict(X_test[i:i+1])

            # 2. Reconstruir el precio escalado
            y_pred_scaled = y_pred_change[0] + X_test[i, -1]
            y_test_scaled = y_test[i] + X_test[i, -1]

            # 3. Invertir escala
            y_pred_real = scaler.inverse_transform([[y_pred_scaled]])[0, 0]
            y_test_real = scaler.inverse_transform([[y_test_scaled]])[0, 0]

            y_pred_real_list.append(y_pred_real)
            y_test_real_list.append(y_test_real)

            # 4. ⚡ Reentrenar con el dato real
            X_new = X_test[i:i+1]  # Input que usaste
            y_new = y_test[i:i+1]  # Salida real

            nn_model.train(X_new, y_new, epochs=1, batch_size=1)

        # Calcular error
        error_metrics = ErrorMetrics(y_test_real_list, y_pred_real_list)
        avg_rmse = error_metrics.rmse()

        print(f"Error final de red neuronal (avg_rmse): {avg_rmse}")

        # Crear la lista de índices donde quieres poner las predicciones
        valid_indices = test_data_daily.index[nn_model.input_shape + horizon - 1:]

        # Ajustar si no coinciden exactamente 
        if len(valid_indices) > len(y_pred_real_list):
            valid_indices = valid_indices[:len(y_pred_real_list)]
        elif len(valid_indices) < len(y_pred_real_list):
            y_pred_real_list = y_pred_real_list[:len(valid_indices)]

        # Asignar las predicciones en la columna 'Prediction' de self.test_data
        self.test_data.loc[valid_indices, 'Prediction'] = y_pred_real_list

    def run_rsi_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI'''

        operaciones = []

        # Calcula RSI
        rsi_window = Indicators.optimize_rsi(self.training_data, progress_callback=progress_callback)
        Indicators.calculate_rsi(self.test_data, window=rsi_window)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar señales RSI
        self.signals = strategy.generate_rsi_signals(self.test_data)

        # Aplicar la estrategia basada en las señales
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones

    def run_macd_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de MACD'''

        operaciones = []

        # Calcula MACD
        short, long, signal = Indicators.optimize_macd(self.training_data, progress_callback=progress_callback)
        Indicators.calculate_macd(self.test_data, short_window=short, long_window=long, signal_window=signal)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar señales MACD
        self.signals = strategy.generate_macd_signals(self.test_data)

        # Aplicar la estrategia basada en las señales
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones

    def run_rsi_macd_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y MACD'''
        rsi_cb = lambda f: progress_callback(f)
        macd_cb = lambda f: progress_callback(1 + f)

        operaciones = []

        # Calcular los indicadores
        rsi_window = Indicators.optimize_rsi(self.training_data, progress_callback=rsi_cb)
        short, long, signal = Indicators.optimize_macd(self.training_data, progress_callback=macd_cb)
        Indicators.calculate_rsi(self.test_data, window=rsi_window)
        Indicators.calculate_macd(self.test_data, short_window=short, long_window=long, signal_window=signal)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar las señales (RSI + MACD)
        self.signals = strategy.generate_rsi_macd_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones

    def run_rsi_regresion_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''

        rsi_cb = lambda f: progress_callback(f)
        regresion_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Calcular RSI
        rsi_window = Indicators.optimize_rsi(self.training_data, progress_callback=rsi_cb)
        Indicators.calculate_rsi(self.test_data, window=rsi_window)

        # Calcular regresión y guardar predicciones en test_data
        self.calculate_regresion(regresion_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones

    def run_macd_regresion_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        macd_cb = lambda f: progress_callback(f)
        regresion_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Calcular MACD
        short, long, signal = Indicators.optimize_macd(self.training_data, progress_callback=macd_cb)
        Indicators.calculate_macd(self.test_data, short_window=short, long_window=long, signal_window=signal)

        # Calcular regresión
        self.calculate_regresion(regresion_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_rsi_macd_regresion_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        rsi_cb = lambda f: progress_callback(f)
        macd_cb = lambda f: progress_callback(1 + f)
        regresion_cb = lambda f, message="": progress_callback(2 + f, message)

        operaciones = []

        # Calcular RSI y MACD
        rsi_window = Indicators.optimize_rsi(self.training_data, progress_callback=rsi_cb)
        short, long, signal = Indicators.optimize_macd(self.training_data, progress_callback=macd_cb)
        Indicators.calculate_rsi(self.test_data, window=rsi_window)
        Indicators.calculate_macd(self.test_data, short_window=short, long_window=long, signal_window=signal)

        # Calcular regresión
        self.calculate_regresion(regresion_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_rsi_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_rsi_neuralNetwork_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        rsi_cb = lambda f: progress_callback(f)
        neuralNetwork_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Calcular RSI
        rsi_window = Indicators.optimize_rsi(self.training_data, progress_callback=rsi_cb)
        Indicators.calculate_rsi(self.test_data, window=rsi_window)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_macd_neuralNetwork_strategy(self, progress_callback):
        macd_cb = lambda f: progress_callback(f)
        neuralNetwork_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Calcular MACD
        short, long, signal = Indicators.optimize_macd(self.training_data, progress_callback=macd_cb)
        Indicators.calculate_macd(self.test_data, short_window=short, long_window=long, signal_window=signal)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones
    def run_rsi_macd_neuralNetwork_strategy(self, progress_callback):
        rsi_cb = lambda f: progress_callback(f)
        macd_cb = lambda f: progress_callback(1 + f)
        neuralNetwork_cb = lambda f, message="": progress_callback(2 + f, message)

        operaciones = []

        # Calcular RSI y MACD
        rsi_window = Indicators.optimize_rsi(self.training_data, progress_callback=rsi_cb)
        short, long, signal = Indicators.optimize_macd(self.training_data, progress_callback=macd_cb)
        Indicators.calculate_rsi(self.test_data, window=rsi_window)
        Indicators.calculate_macd(self.test_data, short_window=short, long_window=long, signal_window=signal)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_rsi_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones