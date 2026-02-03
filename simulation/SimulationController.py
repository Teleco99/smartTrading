from analytics.Indicators import Indicators 
from analytics.MultiOutputRegression import MultiOutputRegression
from analytics.NeuralNetwork import NeuralNetwork
from analytics.RandomForest import RandomForest
from analytics.Strategy import Strategy 
from metrics.ErrorMetrics import ErrorMetrics

import pandas as pd
import numpy as np

class SimulationController:
    def __init__(self, training_data, test_data, capital_operacion=1000, 
                 frecuencia_reentrenamiento=12, best_ind_params=None, best_mod_params=None):
        self.capital_operacion = capital_operacion
        self.training_data = training_data
        self.test_data = test_data
        self.frecuencia_reentrenamiento = frecuencia_reentrenamiento
        self.signals = None
        self.best_ind_params = best_ind_params
        self.best_mod_params = best_mod_params

    def calculate_rsi(self, progress_callback, data_callback): 
        # Encontramos parámetros óptimos
        if self.best_ind_params is None:
            optimize_result = Indicators.optimize_rsi(self.training_data, progress_callback=progress_callback, verbose=True)
            best_params = optimize_result.iloc[0].to_dict()
        else:
            best_params = self.best_ind_params

        #data_callback("optimización", X=pd.DataFrame([best_params]))

        # Calcular RSI con los mejores parámetros
        Indicators.calculate_rsi(self.test_data, window=int(best_params['window']))

    def calculate_macd(self, progress_callback, data_callback): 
        # Encontramos parámetros óptimos
        if self.best_ind_params is None:
            optimize_result = Indicators.optimize_macd(self.training_data, progress_callback=progress_callback, verbose=True)
            best_params = optimize_result.iloc[0].to_dict()
        else:
            best_params = self.best_ind_params

        #data_callback("optimización", X=pd.DataFrame([best_params]))

        # Calcular MACD con los mejores parámetros
        Indicators.calculate_macd(self.test_data, short_window=int(best_params['short']), 
                                  long_window=int(best_params['long']), signal_window=int(best_params['signal']))
    
    def calculate_regresion(self, progress_callback, data_callback, frecuencia_reentrenamiento=24):
        '''Calcula regresión lineal y guarda predicciones en dataframe'''

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()

        # Encontramos parámetros óptimos
        if self.best_mod_params is None:
            optimize_result = MultiOutputRegression.optimize(data=training_data_daily, progress_callback=progress_callback, verbose=True)
            best_params = optimize_result.iloc[0].to_dict()
        else:
            best_params = self.best_mod_params

        #data_callback("optimización", X=pd.DataFrame([best_params]))
        
        window_optimo = int(best_params['window'])
        buffer_size = int(best_params['buffer'])

        model = MultiOutputRegression(window=window_optimo, buffer_size=buffer_size)

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        # Guardar predicciones 
        y_pred_list = []
        y_test_list = []

        progress_callback(1, "⚙️ Calculando predicciones")

        horizon = 1

        full_window = window_optimo * model.sampling_rate
        train_size = full_window + buffer_size
        current_train_data = train_close[-train_size:]

        # Preparar entradas y salidas de datos de entrenamiento
        X_train, y_train = MultiOutputRegression.prepare_data(data=current_train_data, window=window_optimo, sampling_rate=model.sampling_rate)
        
        # Entrenar inicial
        model.train(X_train, y_train)

        # Validar con un único bucle, reentreno cada frecuencia_reentrenamiento
        end_limit = len(test_close) - horizon + 1
        for i in range(full_window, end_limit):
            # Reentreno cada frecuencia_reentrenamiento
            if frecuencia_reentrenamiento != 0 and i % frecuencia_reentrenamiento == 0:
                # Datos disponibles hasta ahora
                available = np.concatenate([train_close, test_close[:i]])
                current_train_data = available[-train_size:]

                X_train, y_train = MultiOutputRegression.prepare_data(data=current_train_data, window=window_optimo, sampling_rate=model.sampling_rate)
                model.train(X_train, y_train)

            last_input = test_close[i - full_window : i]
            y_true = test_close[i + horizon - 1]

            y_pred = model.predict(last_input)
            y_pred_real = y_pred[0, -1]
            
            # Guardar resultados
            y_pred_list.append(y_pred_real)
            y_test_list.append(y_true)

        # Calcular error
        error_metrics = ErrorMetrics(y_test_list, y_pred_list)
        avg_rmse = error_metrics.rmse()

        print(f"Error final de regresión (avg_rmse): {avg_rmse}")

        start = full_window + (horizon - 1)
        valid_indices = test_data_daily.index[start : start + len(y_pred_list)]

        # Asignar las predicciones
        self.test_data.loc[valid_indices, 'Prediction'] = y_pred_list

    def calculate_pred_neuralNetwork(self, progress_callback, data_callback, frecuencia_reentrenamiento=24):
        # Crear modelo
        model = NeuralNetwork()

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()

        # Optimizar hiperparámetros
        if self.best_mod_params is None:
             optimize_result = model.optimize(data=training_data_daily, progress_callback=progress_callback, verbose=True)
             best_params = optimize_result.iloc[0].to_dict()
        else:
            best_params = self.best_mod_params

        window = 1

        model = NeuralNetwork(input_shape=window, learning_rate=best_params['learning_rate'],
                                hidden_neurons=int(best_params['neurons']), batch_size=int(best_params['batch_size']))

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        y_pred_list = []
        y_test_list = []

        horizon = 1
        full_window = window * model.sampling_rate

        # Preparar datos iniciales de entrenamiento
        X_train, y_train = model.prepare_data(data=train_close, window=window)
        data_callback("training", X=X_train, y=y_train)
        # Entrenar inicial
        model.train(X_train, y_train)

        # Validar en bloques, reentreno cada frecuencia_reentrenamiento
        end_limit = len(test_close) - horizon + 1
        
        i = full_window
        while i < end_limit:
            # Predicciones del tramo actual antes de reentreno
            next_retrain = i + frecuencia_reentrenamiento if frecuencia_reentrenamiento != 0 else end_limit
            j_end = min(next_retrain, end_limit)

            X_val = []
            y_val = []
            for j in range(i, j_end):
                X_val.append(test_close[j - full_window : j])
                y_val.append(test_close[j + horizon - 1])
            data_callback("predict", X=X_val, y=y_val)
            # Predicciones en batch para todo el tramo
            y_pred = model.predict(X_val)
            y_pred_list.extend([pred[-1] for pred in y_pred])
            y_test_list.extend(y_val)

            progress_callback(i / end_limit, f"⚙️ Calculando predicción {i} de {end_limit}")

            # Reentreno (warm start)
            if frecuencia_reentrenamiento != 0 and j_end < end_limit:
                # Datos disponibles hasta ahora
                available = np.concatenate([train_close, test_close[:i]])
                X_train, y_train = model.prepare_data(data=available, window=window)
                data_callback("retraining", X=X_train, y=y_train)
                model.train(X_train, y_train)

            i = j_end

        if len(y_pred_list) > 0 and len(y_test_list) > 0:
            error_metrics = ErrorMetrics(y_test_list, y_pred_list)
            rmse = error_metrics.rmse()
            print(f"Error final de red neuronal (rmse): {rmse}")

            start = full_window + (horizon - 1)
            valid_indices = test_data_daily.index[start : start + len(y_pred_list)]

            # Asignar las predicciones en la columna 'Prediction' de self.test_data
            self.test_data.loc[valid_indices, 'Prediction'] = y_pred_list

    def calculate_pred_randomForest(self, progress_callback, data_callback, debug=True, frecuencia_reentrenamiento=24):
        # Crear modelo
        model = RandomForest()

        # Filtrar horario operativo
        training_data_daily = self.training_data.copy()
        test_data_daily = self.test_data.copy()

        # Optimizar hiperparámetros
        if self.best_mod_params is None:
            optimize_result = model.optimize(data=training_data_daily, progress_callback=progress_callback, verbose=False)
            best_params = optimize_result.iloc[0].to_dict()
        else:
            best_params = self.best_mod_params

        #data_callback("optimización", X=pd.DataFrame([best_params]))

        window = int(best_params['window'])
        buffer = int(best_params['buffer'])

        model = RandomForest(buffer_size=buffer, window=window, n_estimators=int(best_params['n_estimators']),
                                max_depth=int(best_params['max_depth']), min_samples_split=int(best_params['min_samples_split']))
        model._build_model(warm_start=True)

        train_close = training_data_daily['<CLOSE>'].values
        test_close = test_data_daily['<CLOSE>'].values

        y_pred_list = []
        y_test_list = []

        horizon = 1
        full_window = window * model.sampling_rate
        train_size = full_window + buffer

        # Preparar datos iniciales de entrenamiento
        current_train_data = train_close[-train_size:]
        X_train, y_train = model.prepare_data(data=current_train_data, window=window)
        
        # Entrenar inicial
        model.train(X_train, y_train)

        # Validar con un único bucle, reentreno cada frecuencia_reentrenamiento
        end_limit = len(test_close) - horizon + 1
        for i in range(full_window, end_limit):
            progress_callback(i / end_limit, f"⚙️ Calculando predicción {i} de {end_limit}")

            # Reentreno cada frecuencia_reentrenamiento
            if frecuencia_reentrenamiento != 0 and i % frecuencia_reentrenamiento == 0:
                # Datos disponibles hasta ahora
                available = np.concatenate([train_close, test_close[:i]])
                current_train_data = available[-train_size:]

                X_train, y_train = model.prepare_data(data=current_train_data, window=window)
                model.train(X_train, y_train)

            last_input = test_close[i - full_window : i]
            y_true = test_close[i + horizon - 1]

            y_pred = model.predict(last_input)
            
            # Guardar resultados
            y_pred_list.append(y_pred[0] if isinstance(y_pred, np.ndarray) else y_pred)
            y_test_list.append(y_true)

            # Mostrar datos de entrenamiento por pantalla
            if debug and i == 5 and data_callback is not None:
                print(f" Calculando predicción con len(window)={window}, len(last_input)={len(last_input)}, len(current_train_data)={len(current_train_data)}")
                data_callback("entrenamiento", i, X_train, y_train)
                data_callback("validacion", i, np.array([y_pred]).reshape(1, -1), y_pred)
                data_callback("errores (entrada=predicción, salida=reales)", i, y_pred_list, y_test_list)

        if len(y_pred_list) > 0 and len(y_test_list) > 0:
            # Calcular error
            error_metrics = ErrorMetrics(y_test_list, y_pred_list)
            rmse = error_metrics.rmse()

            print(f"Error final de Random Forest (rmse): {rmse}")

            start = full_window + (horizon - 1)
            valid_indices = test_data_daily.index[start : start + len(y_pred_list)]

            # Asignar las predicciones en la columna 'Prediction' de self.test_data
            self.test_data.loc[valid_indices, 'Prediction'] = y_pred_list

    def run_rsi_strategy(self, progress_callback, data_callback=None):
        '''Aplica la estrategía basada en señales de RSI'''
        operaciones = []

        # Optimiza y calcula RSI
        self.calculate_rsi(progress_callback, data_callback)

        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar señales RSI
        self.signals = strategy.generate_rsi_signals(self.test_data)

        # Aplicar la estrategia basada en las señales
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones

    def run_macd_strategy(self, progress_callback, data_callback=None):
        '''Aplica la estrategía basada en señales de MACD'''

        operaciones = []

        # Optimiza y calcula MACD
        self.calculate_macd(progress_callback, data_callback)

        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar señales MACD
        self.signals = strategy.generate_macd_signals(self.test_data)

        # Aplicar la estrategia basada en las señales
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones

    def run_rsi_regresion_strategy(self, progress_callback, data_callback=None):
        '''Aplica la estrategía basada en señales de RSI y regresión'''

        rsi_cb = lambda f, message="": progress_callback(f, message=message)
        regresion_cb = lambda f, message="": progress_callback(1 + f, message=message)

        operaciones = []

        # Optimizar y calcular RSI
        self.calculate_rsi(rsi_cb, data_callback)

        # Calcular regresión y guardar predicciones en test_data
        self.calculate_regresion(regresion_cb, data_callback, frecuencia_reentrenamiento=self.frecuencia_reentrenamiento)

        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones

    def run_macd_regresion_strategy(self, progress_callback, data_callback=None):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        macd_cb = lambda f, message="": progress_callback(f, message)
        regresion_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular MACD
        self.calculate_macd(macd_cb, data_callback)

        # Calcular regresión
        self.calculate_regresion(regresion_cb, data_callback, frecuencia_reentrenamiento=self.frecuencia_reentrenamiento)
       
        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar las señales (RSI + regresión)
        self.signals = strategy.generate_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_rsi_neuralNetwork_strategy(self, progress_callback, data_callback=None):
        '''Aplica la estrategía basada en señales de RSI y regresión'''
        rsi_cb = lambda f, message="": progress_callback(f, message)
        neuralNetwork_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular RSI
        self.calculate_rsi(rsi_cb, data_callback)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb, data_callback, frecuencia_reentrenamiento=self.frecuencia_reentrenamiento)

        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_macd_neuralNetwork_strategy(self, progress_callback, data_callback=None):
        macd_cb = lambda f, message="": progress_callback(f, message)
        neuralNetwork_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular MACD
        self.calculate_macd(macd_cb, data_callback)

        # Calcular predicciones de red neuronal
        self.calculate_pred_neuralNetwork(neuralNetwork_cb, data_callback, frecuencia_reentrenamiento=self.frecuencia_reentrenamiento)

        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar las señales (RSI + red neuronal)
        self.signals = strategy.generate_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_rsi_randomForest_strategy(self, progress_callback, data_callback=None):
        rsi_cb = lambda f, message="": progress_callback(f, message)
        randomForest_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular los indicadores
        self.calculate_rsi(rsi_cb, data_callback=None)

        # Calcular predicciones de Random Forest
        self.calculate_pred_randomForest(randomForest_cb, data_callback=None, frecuencia_reentrenamiento=self.frecuencia_reentrenamiento)

        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar las señales (RSI + Random Forest)
        self.signals = strategy.generate_rsi_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    
    def run_macd_randomForest_strategy(self, progress_callback, data_callback=None):
        macd_cb = lambda f, message="": progress_callback(f, message)
        randomForest_cb = lambda f, message="": progress_callback(1 + f, message)

        operaciones = []

        # Optimizar y calcular los indicadores
        self.calculate_macd(macd_cb, data_callback=None)

        # Calcular predicciones de Random Forest
        self.calculate_pred_randomForest(randomForest_cb, data_callback=None, frecuencia_reentrenamiento=self.frecuencia_reentrenamiento)
        
        strategy = Strategy(capital_operacion=self.capital_operacion)

        # Generar las señales (MACD + Random Forest)
        self.signals = strategy.generate_macd_prediction_signals(self.test_data) 

        # Aplicar la estrategia basada en la señales generadas
        operaciones = strategy.simulate_strategy(self.test_data, self.signals)

        return operaciones
    





    