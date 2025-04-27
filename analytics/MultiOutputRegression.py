from metrics.ErrorMetrics import ErrorMetrics
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import numpy as np
import pandas as pd

class MultiOutputRegression:
    '''Modelo de regresión multioutput con horizonte de predicción variable.'''
    
    def __init__(self, window=12, horizon=3, base_model=LinearRegression()):
        self.window = window  # Ventana de entrada (cuántos datos pasados se usan)
        self.horizon = horizon  # Horizonte de predicción (número de valores futuros)
        self.model = MultiOutputRegressor(base_model)  # Modelo multioutput

    def prepare_data(self, data):
        '''Prepara los datos para el entrenamiento y la predicción generando las ventanas.'''
        X, y = [], []

        rango = len(data) - self.window - self.horizon

        if rango <= 0:
            print("❌ No hay suficientes datos para esta configuración.")
            return

        for i in range(rango):
            X.append(data[i:i+self.window])  # Ventana de entrada
            y.append(data[i+self.window:i+self.window+self.horizon])  # Valores futuros
        
        return np.array(X), np.array(y)

    def train(self, data):
        '''Entrena el modelo con la serie temporal dada.'''

        X, y = self.prepare_data(data)

        if len(X) > 0:
            self.model.fit(X, y)
        else:
            print("❌ Datos insuficientes después de procesar las ventanas.")

    def predict(self, windows):
        '''Predice los próximos "horizon" valores usando las ventanas de entrada.'''
        predictions = []
        
        for window in windows:
            if len(window) != self.window:
                raise ValueError(f"Se esperaba una ventana de tamaño {self.window}, pero se recibió {len(window)}")
            
            # Asegurar que la ventana esté en el formato correcto para el modelo
            window = np.array(window).reshape(1, -1)  # Convertir la ventana en array 2D
            
            # Obtener las predicciones para esta ventana
            prediction = self.model.predict(window)

            # Añadir las predicciones a la lista
            predictions.append(prediction.flatten())  # Hacer que sea un vector plano de predicciones
        
        return np.array(predictions)  # Retorna un array de arrays de predicciones
    
    @staticmethod
    def optimize_window(data, progress_callback, window_range=range(16, 65, 16), horizon=3):
        '''Optimiza el parámetro window evaluando rolling forecasts sobre el training data.'''
        best_score = float('inf')
        best_window = None
        results = []

        completed = 0
        total = len(window_range)
        
        # Para no saturar al sistema con la optimización, se usan máximo los ultimos 200 datos
        data_array = data['<CLOSE>'].values[-200:]

        for window in window_range:
            try:
                errors = []

                for i in range(0, len(data_array) - window - horizon):
                    # Datos para entrenamiento: desde el inicio hasta la ventana actual  
                    if i % 10 == 0:
                        progress_callback(completed / total, f"🔎 Optimizando regresión: Probando ventana {window}: {i} de {len(data_array)} datos")

                    train_slice = data_array[i : i + window]
                    future_slice = data_array[i + window : i + window + horizon]

                    if len(future_slice) < horizon:
                        continue

                    model = MultiOutputRegression(window=window, horizon=horizon)
                    model.train(pd.Series(data_array))

                    # Predice solo el siguiente conjunto futuro
                    pred = model.predict([train_slice])[0]

                    em = ErrorMetrics(actual_values=[future_slice[-1]], predicted_values=[pred[-1]])
                    errors.append(em.rmse())

                if not errors:
                    continue

                avg_rmse = np.mean(errors)
                results.append((window, avg_rmse))

                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_window = window

                completed += 1
                progress_callback(completed / total)

                # Depuración: mostrar cada combinación
                print(f"Regresion probada: window={window}, avg_rmse={avg_rmse:.4}")

            except Exception as e:
                import traceback
                print(f"❌ Error con window={window}")
                traceback.print_exc()
                continue

        print(f"🟢 Mejor Regresión Lineal: window={best_window} con RMSE = {best_score:.4f}")
        return best_window