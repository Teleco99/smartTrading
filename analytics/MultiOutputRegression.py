from metrics.ErrorMetrics import ErrorMetrics
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np

class MultiOutputRegression:
    '''Modelo de regresión multioutput con horizonte de predicción variable.'''
    
    def __init__(self, window=12, buffer_size=24, sampling_rate=3):
        self.buffer_size = buffer_size
        self.window = window  # Ventana de entrada (cuántos datos pasados se usan)
        self.sampling_rate = sampling_rate  # Cada cuántos puntos usar (1 = todos, 3 = 1 de cada 3)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))  

    def train(self, X_data, y_data):
        '''Entrena el modelo con ventanas de entradas y salidas dadas.'''

        if len(X_data) > 0:
            self.model.fit(X_data, y_data)
        else:
            print("❌ Datos insuficientes después de procesar las ventanas.")

    def predict(self, window):
        '''Predice los próximos "horizon" valores usando la ventana de entrada.'''
        predictions = []
        
        if len(window) < self.window * self.sampling_rate:
            raise ValueError(f"Longitud de window es de {len(window)} puntos. Se necesitan al menos {self.window * self.sampling_rate} puntos")
            
        # Asegurar que la ventana esté en el formato correcto para el modelo (downsampling)
        sampled_window = window[::self.sampling_rate][-self.window:]
        sampled_window = np.array(sampled_window).reshape(1, -1)  # Convertir la ventana en array 2D

        # Obtener las predicciones para esta ventana
        prediction = self.model.predict(sampled_window)
        
        # Añadir las predicciones a la lista
        predictions.append(prediction.flatten())  # type: ignore # Hacer que sea un vector plano de predicciones
        
        return np.array(predictions)  # Retorna un array de arrays de predicciones
    
    @staticmethod
    def prepare_data(data, window, sampling_rate=3, horizon=1):
        '''Prepara los datos para el entrenamiento y la predicción generando las ventanas.'''
        X, y = [], []
        sampled_data = data[::sampling_rate]  # Downsamplea la serie

        for i in range(len(sampled_data) - window - horizon + 1):
            X.append(sampled_data[i : i + window])
            y.append(sampled_data[i + window : i + window + horizon])

        return np.array(X), np.array(y)

    @staticmethod
    def optimize(data, progress_callback, data_callback=None, 
                 window_sizes=[6, 9, 12], buffer_sizes=[12, 24, 36], 
                 sampling_rate=3, horizon=1, verbose=False):
        '''Optimiza window y buffer evaluando rolling forecasts sobre training/validation split.'''
        best_score = float('inf')
        best_params = {}

        completed = 0
        total = len(window_sizes) * len(buffer_sizes)

        # Para no saturar: usar máximo 2000 datos
        data_array = data['<CLOSE>'].values[-2000:]

        # Split en entrenamiento y validación
        split_idx = int(len(data_array) * 0.7)
        train_data = data_array[:split_idx]
        val_data = data_array[split_idx:]

        for window in window_sizes:
            for buffer in buffer_sizes:
                    if len(train_data) < window + horizon:
                        continue

                    model = MultiOutputRegression(window=window, sampling_rate=sampling_rate)

                    y_pred_list = []
                    y_test_list = []

                    # Total de pasos en la simulación en tiempo real
                    steps = len(val_data)

                    # Validar en validation pero empezando desde los ultimos training
                    for i in range(steps):
                        if i % 10 == 0:
                            progress_callback(completed / total, f"🔎 Optimizando regresión: Window {window}: {i}/{steps}")

                        # Número real de puntos a coger
                        train_size = window * sampling_rate + buffer

                        # Coger últimos puntos de training + inicio de test
                        train_window = train_data[-train_size + i:]
                        val_window = val_data[:i + horizon]

                        # Concatenarlos
                        current_data = np.concatenate([train_window, val_window])

                        # Entrenamiento solo con datos anteriores al objetivo   
                        current_train_data = current_data[:-horizon]

                        # Preparar entradas (ventanas) sobre esos datos
                        X_train, y_train = MultiOutputRegression.prepare_data(data=current_train_data, window=window, sampling_rate=sampling_rate)
                        
                        # Entrenar
                        model.train(X_train, y_train)

                        # Preparar ventana para predicción del siguiente valor
                        last_input = current_data[-(window * sampling_rate + horizon) : -horizon]  # Última ventana de entrada
                        y_true = current_data[-1]
                        
                        y_pred = model.predict(last_input)
                        y_pred_real = y_pred[0, -1]
                        
                        # Guardar resultados
                        y_pred_list.append(y_pred_real)
                        y_test_list.append(y_true)

                        if i==5 and data_callback is not None:
                            data_callback("errores (entrada=predicción, salida=reales)", i, y_pred_list, y_test_list)

                    if y_pred_list and y_test_list:
                        error_metrics = ErrorMetrics(y_test_list, y_pred_list)
                        avg_rmse = error_metrics.rmse()

                        if avg_rmse < best_score:
                            best_score = avg_rmse
                            best_params = {
                                'buffer': buffer,
                                'window': window,
                            }

                        if verbose:
                            print(f" Regresión Lineal probada: window={window}, buffer={buffer}, avg_rmse={avg_rmse:.4f}")
                    else:
                        avg_rmse = float('inf')
                        print(f"❌ Regresión Lineal sin suficientes puntos: Window {window}, buffer={buffer}, avg_rmse={avg_rmse:.4f}")

                    completed += 1
                    progress_callback(completed / total, "Optimizando...")

        print(f'🟢 Mejores parámetros: {best_params}')

        return best_params