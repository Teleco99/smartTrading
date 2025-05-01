from analytics.Indicators import Indicators 
from analytics.MultiOutputRegression import MultiOutputRegression
from analytics.NeuralNetwork import NeuralNetwork
from simulation.Strategy import Strategy 
from metrics.ErrorMetrics import ErrorMetrics

import numpy as np

class Simulation:
    def __init__(self, training_data, test_data, capital_por_operacion=1000, horario_permitido=('08:00', '16:30')):
        self.capital_por_operacion = capital_por_operacion
        self.training_data = training_data
        self.test_data = test_data
        self.horario_permitido = horario_permitido
        self.signals = None

    def calculate_regresion(self, progress_callback):
        '''Calcula regresión lineal y guarda predicciones en dataframe'''

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()
        training_data_daily = training_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])
        test_data_daily = test_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])

        # Encontramos parámetros óptimos
        best_params = MultiOutputRegression.optimize(data=training_data_daily, progress_callback=progress_callback)
        window_optimo = best_params['window']
        buffer_size = best_params['buffer']

        model = MultiOutputRegression(window=window_optimo)
        
        # Puntos de entrenamiento iniciales
        X_test, y_test = MultiOutputRegression.prepare_data(test_data_daily['<CLOSE>'], window=window_optimo)

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        # Guardar predicciones 
        y_pred_list = []
        y_test_list = []

        progress_callback(1, "⚙️ Calculando predicciones")

        horizon = 3

        # Total de pasos en la simulación en tiempo real
        steps = len(test_close)

        for i in range(steps):
            # Número real de puntos a coger
            train_size = buffer_size + window_optimo

            # Coger últimos puntos de training + inicio de test
            train_window = train_close[-train_size:]
            test_window = test_close[:i + 1]

            # Datos disponibles hasta el punto actual
            current_data = np.concatenate([train_window, test_window])
            
            # Entrenamiento solo con datos anteriores al objetivo   
            train_data = current_data[:-(horizon)]
                
            X_train, y_train = MultiOutputRegression.prepare_data(train_data, window=window_optimo)

            # Entrenar el modelo con datos actuales
            model.train(X_train, y_train)

            # Preparar ventana para predicción del siguiente valor
            last_input = current_data[-(window_optimo + horizon) : -horizon]  # Última ventana de entrada
            y_true = current_data[-1]
            
            y_pred = model.predict(last_input)
            y_pred_real = y_pred[0, -1]

            # Guardar resultados
            y_pred_list.append(y_pred_real)
            y_test_list.append(y_true)
            
            if i == 0:
                print(f"  Reentrenando modelo...")
                print(f"  Reentrenado con muestra {i}")
                print(f"  📈 last_input: {last_input}")
                print(f"  🎯 y_true (valor real actual): {y_true}")
                print(f"  🤖 y_pred completo: {y_pred}")
                print(f"  🏁 y_pred_real (último valor predicho): {y_pred_real}")
                print("-" * 40)

        # Calcular error
        error_metrics = ErrorMetrics(y_test_list, y_pred_list)
        avg_rmse = error_metrics.rmse()

        print(f"Error final de regresión (avg_rmse): {avg_rmse}")

        valid_indices = test_data_daily.index[:len(y_pred_list)]

        # Asignar las predicciones
        self.test_data.loc[valid_indices, 'Prediction'] = y_pred_list

    def calculate_pred_neuralNetwork(self, progress_callback):

        horizon = 3

        # Crear modelo
        nn_model = NeuralNetwork()

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()
        training_data_daily = training_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])
        test_data_daily = test_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])

        # Optimizar hiperparámetros
        best_params = nn_model.optimize(data=training_data_daily, progress_callback=progress_callback)
        window = best_params['window']
        buffer = best_params['buffer']

        if best_params:
            nn_model = NeuralNetwork(buffer_size=buffer, input_shape=window, learning_rate=best_params['learning_rate'],
                                     hidden_neurons=best_params['neurons'], batch_size=best_params['batch_size'])
        else:
            progress_callback(1, "❌ No hay suficientes datos para optimizar la red")

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        print(train_close)

        y_pred_list = []
        y_test_list = []

        # Total de pasos en la simulación en tiempo real
        steps = len(test_close)

        for i in range(steps):
            progress_callback(i / steps, f"⚙️ Calculando predicción {i} de {steps}")

            # Número real de puntos a coger
            train_size = buffer + window

            # Coger últimos puntos de training + inicio de test
            train_window = train_close[-train_size:]
            test_window = test_close[:i + 1]

            # Datos disponibles hasta el punto actual
            current_data = np.concatenate([train_window, test_window])
            
            # Entrenamiento solo con datos anteriores al objetivo   
            train_data = current_data[:-(horizon)]

            X_train, y_train = nn_model.prepare_data(train_data)

            # Entrenar el modelo con datos actuales
            nn_model.train(X_train, y_train)

            # Preparar ventana para predicción del siguiente valor
            last_input = current_data[-(window + horizon) : -horizon].reshape(1, -1) 
            y_true = current_data[-1]
            
            y_pred = nn_model.predict(last_input)
            y_pred_real = y_pred[0, -1]
            
            if i < 5:
                print(f"  🏋️ last_input: {last_input}")
                print(f"  🔍 y_pred: {y_pred_real}")
                print(f"  y_true: {y_true}")
            
            y_pred_list.append(y_pred_real)
            y_test_list.append(y_true)

        if len(y_pred_list) > 0 and len(y_test_list) > 0:
            # Calcular error
            error_metrics = ErrorMetrics(y_test_list, y_pred_list)
            avg_rmse = error_metrics.rmse()

            print(f"Error final de red neuronal (avg_rmse): {avg_rmse}")

            valid_indices = test_data_daily.index[:len(y_pred_list)]

            # Asignar las predicciones en la columna 'Prediction' de self.test_data
            self.test_data.loc[valid_indices, 'Prediction'] = y_pred_list

            self.test_data['Prediction'] = self.test_data['Prediction'].rolling(window=3, min_periods=1).mean()

    def run_rsi_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI'''

        operaciones = []

        # Calcula RSI
        rsi_window = Indicators.optimize_rsi(self.training_data, progress_callback=progress_callback)
        Indicators.calculate_rsi(self.test_data, window=rsi_window)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

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

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_rsi_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.apply_strategy(self.test_data, self.signals)

        return operaciones
    





    