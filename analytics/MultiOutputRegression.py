from metrics.ErrorMetrics import ErrorMetrics
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import numpy as np
import pandas as pd

class MultiOutputRegression:
    '''Modelo de regresión multioutput con horizonte de predicción variable.'''
    
    def __init__(self, window=12, base_model=LinearRegression()):
        self.window = window  # Ventana de entrada (cuántos datos pasados se usan)
        self.model = MultiOutputRegressor(base_model)  # Modelo multioutput

    def train(self, X_data, y_data):
        '''Entrena el modelo con ventanas de entradas y salidas dadas.'''

        if len(X_data) > 0:
            self.model.fit(X_data, y_data)
        else:
            print("❌ Datos insuficientes después de procesar las ventanas.")

    def predict(self, window):
        '''Predice los próximos "horizon" valores usando la ventana de entrada.'''
        predictions = []
        
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
    def prepare_data(data, window, horizon=1):
        '''Prepara los datos para el entrenamiento y la predicción generando las ventanas.'''
        X, y = [], []

        for i in range(len(data) - window - horizon + 1):
            X.append(data[i : i + window])  # Ventana de entrada
            y.append(data[i + window : i + window + horizon])  # Valores futuros
        
        return np.array(X), np.array(y)

    @staticmethod
    def optimize(data, progress_callback, buffer_sizes=[12, 18], window_range=[12, 18], horizon=1):
        '''Optimiza el parámetro window evaluando rolling forecasts sobre training/validation split.'''
        best_score = float('inf')
        best_params = {}

        completed = 0
        total = len(window_range) * len(buffer_sizes)

        # Para no saturar: usar máximo 100 datos
        data_array = data['<CLOSE>'].values[-200:]

        # Split en entrenamiento y validación
        split_idx = int(len(data_array) * 0.7)
        train_data = data_array[:split_idx]
        val_data = data_array[split_idx:]

        for window in window_range:
            for buffer in buffer_sizes:
                    if len(train_data) < window + horizon:
                        continue

                    model = MultiOutputRegression(window=window)

                    y_pred_list = []
                    y_test_list = []

                    # Total de pasos en la simulación en tiempo real
                    steps = len(val_data)

                    # Validar en validation pero empezando desde los ultimos training
                    for i in range(steps):
                        if i % 10 == 0:
                            progress_callback(completed / total, f"🔎 Optimizando regresión: Window {window}: {i}/{steps}")

                        # Número real de puntos a coger
                        train_size = window + buffer

                        # Coger últimos puntos de training + inicio de test
                        train_window = train_data[-train_size:]
                        val_window = val_data[:i + 1]

                        # Concatenarlos
                        current_data = np.concatenate([train_window, val_window])

                        # Entrenamiento solo con datos anteriores al objetivo   
                        current_train_data = current_data[:-(horizon)]

                        # Preparar entradas (ventanas) sobre esos datos
                        X_train, y_train = MultiOutputRegression.prepare_data(data=current_train_data, window=window)

                        # Entrenar
                        model.train(X_train, y_train)

                        # Preparar ventana para predicción del siguiente valor
                        last_input = current_data[-(window + horizon) : -horizon]  # Última ventana de entrada
                        y_true = current_data[-1]
                                
                        y_pred = model.predict(last_input)
                        y_pred_real = y_pred[0, -1]

                        # Guardar resultados
                        y_pred_list.append(y_pred_real)
                        y_test_list.append(y_true)

                        if i == 0:
                            print(f"  Optimizando modelo...")
                            print(f"  Reentrenado con muestra {i}")
                            print(f"  📈 last_input: {last_input}")
                            print(f"  🎯 y_true (valor real actual): {y_true}")
                            print(f"  🤖 y_pred completo: {y_pred}")
                            print(f"  🏁 y_pred_real (último valor predicho): {y_pred_real}")
                            print("-" * 40)

                    if y_pred_list and y_test_list:
                        error_metrics = ErrorMetrics(y_test_list, y_pred_list)
                        avg_rmse = error_metrics.rmse()

                        if avg_rmse < best_score:
                            best_score = avg_rmse
                            best_params = {
                                'buffer': buffer,
                                'window': window,
                            }

                        print(f"✅ Regresión Lineal probada: window={window}, buffer={buffer}, avg_rmse={avg_rmse:.4f}")
                    else:
                        avg_rmse = float('inf')
                        print(f"❌ Regresión Lineal sin suficientes puntos: Window {window}, buffer={buffer}, avg_rmse={avg_rmse:.4f}")

                    completed += 1
                    progress_callback(completed / total)

        print(f'🟢 Mejores parámetros: {best_params}')

        return best_params