from analytics.Indicators import Indicators 
from analytics.MultiOutputRegression import MultiOutputRegression
from analytics.NeuralNetwork import NeuralNetwork
from analytics.RandomForest import RandomForest
from analytics.Strategy import Strategy 
from metrics.ErrorMetrics import ErrorMetrics

import numpy as np

class SimulationController:
    def __init__(self, training_data, test_data, capital_por_operacion=1000, horario_permitido=('08:00', '16:30')):
        self.capital_por_operacion = capital_por_operacion
        self.training_data = training_data
        self.test_data = test_data
        self.horario_permitido = horario_permitido
        self.signals = None

    def calculate_rsi(self, progress_callback): 
        best_params = Indicators.optimize_rsi(self.training_data, progress_callback=progress_callback, verbose=True)
        Indicators.calculate_rsi(self.test_data, window=best_params['window'])

    def calculate_macd(self, progress_callback): 
        best_params = Indicators.optimize_macd(self.training_data, progress_callback=progress_callback, verbose=True)
        Indicators.calculate_macd(self.test_data, short_window=best_params['short'], long_window=best_params['long'], signal_window=best_params['signal'])
    
    def calculate_regresion(self, progress_callback, sampling_rate=3, horizon=1):
        '''Calcula regresión lineal y guarda predicciones en dataframe'''

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()
        training_data_daily = training_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])
        test_data_daily = test_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])

        # Encontramos parámetros óptimos
        best_params = MultiOutputRegression.optimize(data=training_data_daily, progress_callback=progress_callback, verbose=True)
        window_optimo = best_params['window']
        buffer_size = best_params['buffer']

        model = MultiOutputRegression(window=window_optimo, buffer_size=buffer_size, sampling_rate=sampling_rate)

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        # Guardar predicciones 
        y_pred_list = []
        y_test_list = []

        progress_callback(1, "⚙️ Calculando predicciones")

        horizon = 1

        # Total de pasos en la simulación en tiempo real
        steps = len(test_close)

        for i in range(steps):

            # Número real de puntos a coger
            train_size = window_optimo * sampling_rate + model.buffer_size

            # Coger últimos puntos de training + inicio de test
            train_window = train_close[-train_size + i:]
            test_window = test_close[:i + horizon]

            # Datos disponibles hasta el punto actual
            current_data = np.concatenate([train_window, test_window])
            
            # Entrenamiento solo con datos anteriores al objetivo   
            train_data = current_data[:-horizon]
            X_train, y_train = MultiOutputRegression.prepare_data(train_data, window=window_optimo)
            
            # Entrenar el modelo con datos actuales
            model.train(X_train, y_train)

            # Preparar ventana para predicción del siguiente valor
            last_input = current_data[-(window_optimo * sampling_rate + horizon) : -horizon]  
            y_true = current_data[-1]
            
            y_pred = model.predict(last_input)
            y_pred_real = y_pred[0, -1]
            
            # Guardar resultados
            y_pred_list.append(y_pred_real)
            y_test_list.append(y_true)

        # Calcular error
        error_metrics = ErrorMetrics(y_test_list, y_pred_list)
        avg_rmse = error_metrics.rmse()

        print(f"Error final de regresión (avg_rmse): {avg_rmse}")

        valid_indices = test_data_daily.index[:len(y_pred_list)]

        # Asignar las predicciones
        self.test_data.loc[valid_indices, 'Prediction'] = y_pred_list

    def calculate_pred_neuralNetwork(self, progress_callback):
        # Crear modelo
        nn_model = NeuralNetwork()

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()
        training_data_daily = training_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])
        test_data_daily = test_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])

        # Optimizar hiperparámetros
        best_params = nn_model.optimize(data=training_data_daily, progress_callback=progress_callback, verbose=True)
        window = best_params['window']
        buffer = best_params['buffer']

        if best_params:
            nn_model = NeuralNetwork(buffer_size=buffer, input_shape=window, learning_rate=best_params['learning_rate'],
                                     hidden_neurons=best_params['neurons'], batch_size=best_params['batch_size'])
        else:
            progress_callback(1, "❌ No hay suficientes datos para optimizar la red")

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        y_pred_list = []
        y_test_list = []

        # Total de pasos en la simulación en tiempo real
        steps = len(test_close)

        horizon = 1

        for i in range(steps):
            progress_callback(i / steps, f"⚙️ Calculando predicción {i} de {steps}")

            # Número real de puntos a coger
            train_size = window * nn_model.sampling_rate + nn_model.buffer_size 

            # Coger últimos puntos de training + inicio de test
            train_window = train_close[-train_size + i:]
            test_window = test_close[:i + horizon]

            # Datos disponibles hasta el punto actual
            current_data = np.concatenate([train_window, test_window])
            
            # Entrenamiento solo con datos anteriores al objetivo   
            train_data = current_data[:-horizon]

            X_train, y_train = nn_model.prepare_data(train_data)

            # Entrenar el modelo con datos actuales
            nn_model.train(X_train, y_train)

            # Preparar ventana para predicción del siguiente valor
            last_input = current_data[-(window * nn_model.sampling_rate + horizon)  : -horizon].reshape(1, -1) 
            y_true = current_data[-1]

            y_pred = nn_model.predict(last_input)
            y_pred_real = y_pred[0, -1]

            # Guardar resultados
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

    def calculate_pred_randomForest(self, progress_callback, data_callback, debug=True):
        # Crear modelo
        model = RandomForest()

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()
        training_data_daily = training_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])
        test_data_daily = test_data_daily.between_time(self.horario_permitido[0], self.horario_permitido[1])

        # Optimizar hiperparámetros
        best_params = model.optimize(data=training_data_daily, progress_callback=progress_callback, verbose=False)
        window = best_params['window']
        buffer = best_params['buffer']

        if best_params:
            model = RandomForest(buffer_size=buffer, window=window, n_estimators=best_params['n_estimators'], 
                                    max_depth=best_params['max_depth'], min_samples_split=best_params['min_sample_split'])
        else:
            progress_callback(1, "❌ No hay suficientes datos para optimizar la red")

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        y_pred_list = []
        y_test_list = []

        # Total de pasos en la simulación en tiempo real
        steps = len(test_close)

        horizon = 1

        for i in range(steps):
            progress_callback(i / steps, f"⚙️ Calculando predicción {i} de {steps}")

            # Número real de puntos a coger
            train_size = window * model.sampling_rate + model.buffer_size 

            # Coger últimos puntos de training + inicio de test
            train_window = train_close[-train_size + i:]
            test_window = test_close[:i + horizon]

            # Datos disponibles hasta el punto actual
            current_data = np.concatenate([train_window, test_window])
            
            # Entrenamiento solo con datos anteriores al objetivo   
            current_train_data = current_data[:-horizon]

            X_train, y_train = model.prepare_data(current_train_data, window)

            # Entrenar el modelo con datos actuales
            model.train(X_train, y_train)

            # Preparar ventana para predicción del siguiente valor
            last_input = current_data[-(window * model.sampling_rate + horizon)  : -horizon] 
            y_true = current_data[-1]

            y_pred = model.predict(last_input)

            # Guardar resultados
            y_pred_list.append(y_pred)
            y_test_list.append(y_true)

            # Asegurar que la ventana esté en el formato correcto para imprimir
            sampled_window = last_input[::model.sampling_rate][-model.window:]
            sampled_window = np.array(sampled_window).reshape(1, -1)  # Convertir la ventana en array 2D

            # Mostrar datos de entrenamiento por pantalla
            if debug and i==5 and data_callback is not None:
                print(f" Calculando predicción con len(window)={window}, len(last_input)={len(last_input)}, len(current_train_data)={len(current_train_data)}")
                data_callback("entrenamiento", i, X_train, y_train)
                data_callback("validacion", i, sampled_window, y_pred)
                data_callback("errores (entrada=predicción, salida=reales)", i, y_pred_list, y_test_list)

        if len(y_pred_list) > 0 and len(y_test_list) > 0:
            # Calcular error
            error_metrics = ErrorMetrics(y_test_list, y_pred_list)
            avg_rmse = error_metrics.rmse()

            print(f"Error final de red neuronal (avg_rmse): {avg_rmse}")

            valid_indices = test_data_daily.index[:len(y_pred_list)]

            # Asignar las predicciones en la columna 'Prediction' de self.test_data
            self.test_data.loc[valid_indices, 'Prediction'] = y_pred_list

    def run_rsi_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI'''

        operaciones = []

        # Optimiza y calcula RSI
        self.calculate_rsi(progress_callback)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar señales RSI
        self.signals = strategy.generate_rsi_signals(self.test_data)

        # Aplicar la estrategia basada en las señales
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones

    def run_macd_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de MACD'''

        operaciones = []

        # Optimiza y calcula MACD
        self.calculate_macd(progress_callback)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar señales MACD
        self.signals = strategy.generate_macd_signals(self.test_data)

        # Aplicar la estrategia basada en las señales
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones

    def run_rsi_macd_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y MACD'''
        rsi_cb = lambda f, message="": progress_callback(f, message)
        macd_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular los indicadores
        self.calculate_rsi(rsi_cb)
        self.calculate_macd(macd_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + MACD)
        self.signals = strategy.generate_rsi_macd_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones

    def run_rsi_regresion_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''

        rsi_cb = lambda f, message="": progress_callback(f, message=message)
        regresion_cb = lambda f, message="": progress_callback(1 + f, message=message)

        operaciones = []

        # Optimizar y calcular RSI
        self.calculate_rsi(rsi_cb)

        # Calcular regresión y guardar predicciones en test_data
        self.calculate_regresion(regresion_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones

    def run_macd_regresion_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        macd_cb = lambda f, message="": progress_callback(f, message)
        regresion_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular MACD
        self.calculate_macd(macd_cb)

        # Calcular regresión
        self.calculate_regresion(regresion_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_rsi_macd_regresion_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        rsi_cb = lambda f, message="": progress_callback(f, message)
        macd_cb = lambda f, message="": progress_callback(1 + f, message)
        regresion_cb = lambda f, message="": progress_callback(2 + f, message)

        operaciones = []

        # Optimizar y calcular los indicadores
        self.calculate_rsi(rsi_cb)
        self.calculate_macd(macd_cb)

        # Calcular regresión
        self.calculate_regresion(regresion_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_rsi_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_rsi_neuralNetwork_strategy(self, progress_callback):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        rsi_cb = lambda f, message="": progress_callback(f, message)
        neuralNetwork_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular RSI
        self.calculate_rsi(rsi_cb)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_macd_neuralNetwork_strategy(self, progress_callback):
        macd_cb = lambda f, message="": progress_callback(f, message)
        neuralNetwork_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular MACD
        self.calculate_macd(macd_cb)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    def run_rsi_macd_neuralNetwork_strategy(self, progress_callback):
        rsi_cb = lambda f, message="": progress_callback(f, message)
        macd_cb = lambda f, message="": progress_callback(1 + f, message)
        neuralNetwork_cb = lambda f, message="": progress_callback(2 + f, message)

        operaciones = []

        # Optimizar y calcular los indicadores
        self.calculate_rsi(rsi_cb)
        self.calculate_macd(macd_cb)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_rsi_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_rsi_randomForest_strategy(self, progress_callback, data_callback=None):
        rsi_cb = lambda f, message="": progress_callback(f, message)
        randomForest_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular los indicadores
        self.calculate_rsi(rsi_cb)

        # Calcular predicciones de red neuronal
        self.calculate_pred_randomForest(randomForest_cb, data_callback)

        strategy = Strategy(capital_por_operacion=self.capital_por_operacion, horario_permitido=self.horario_permitido)

        # Generar las señales (RSI + Random Forest)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    





    