from metrics.ErrorMetrics import ErrorMetrics
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd
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
                 window_sizes=range(1, 10), buffer_sizes=range(100, 500, 10),
                 frecuencia_reentrenamiento=24, sampling_rate=3, horizon=1, verbose=False):
        '''Optimiza window y buffer evaluando rolling forecasts sobre training/validation split.'''
        resultados = []

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
                    progress_callback(completed / total, f"🔎 Optimizando regresión: Probando combinación {completed} de {total}")

                    if len(train_data) < window * sampling_rate + buffer:
                        print(f"❌ Regresión Lineal sin suficientes puntos: train_data={len(train_data)}, buffer={buffer}")
                        continue

                    if len(val_data) < window * sampling_rate + horizon:
                        print(f"❌ Regresión Lineal sin suficientes puntos: val_data={len(val_data)} (warmup insuficiente)")
                        continue

                    model = MultiOutputRegression(window=window, sampling_rate=sampling_rate)

                    y_pred_list = []
                    y_val_list = []

                    train_size = window * sampling_rate + buffer
                    current_train_data = train_data[-train_size:]

                    # Preparar entradas y salidas de datos de entrenamiento
                    X_train, y_train = MultiOutputRegression.prepare_data(data=current_train_data, window=window, sampling_rate=sampling_rate)
                    
                    # Entrenar
                    model.train(X_train, y_train)
                    
                    # Validar
                    end_limit = len(val_data) - horizon + 1
                    for i in range(window * sampling_rate, end_limit):
                        # Reentrenamos tras cada frecuencia de reentrenamiento
                        if frecuencia_reentrenamiento != 0 and i % frecuencia_reentrenamiento == 0:
                            # Datos disponibles hasta ahora
                            available = np.concatenate([train_data, val_data[:i]])
                            current_train_data = available[-train_size:]
                            
                            X_train, y_train = MultiOutputRegression.prepare_data(data=current_train_data, window=window, sampling_rate=sampling_rate)  

                            model.train(X_train, y_train)

                        last_input = val_data[i - (window * sampling_rate): i]   
                        y_true = val_data[i + horizon - 1]

                        y_pred = model.predict(last_input)
                        y_pred_real = y_pred[0, -1]
                        
                        # Guardar resultados
                        y_pred_list.append(y_pred_real)
                        y_val_list.append(y_true)

                        if i==5 and data_callback is not None:
                            data_callback("errores (entrada=predicción, salida=reales)", i, y_pred_list, y_val_list)

                    # Calcular error final al terminar el bucle
                    if y_pred_list and y_val_list:
                        error_metrics = ErrorMetrics(y_val_list, y_pred_list)
                        rmse = error_metrics.rmse()
                        mape = error_metrics.mape()
                        directional_acc_thr = error_metrics.directional_accuracy_price_thr(eps=100.0)
                        directional_acc_quantile = error_metrics.directional_accuracy_price_quantile(q=0.8)

                        resultados.append({
                            'buffer': buffer,
                            'window': window,
                            'rmse': rmse,
                            'mape': mape,
                            'directional_accuracy_thr': directional_acc_thr,
                            'directional_accuracy_quantile': directional_acc_quantile,
                        })

                        if verbose:
                            print(f"✅ Regresión Lineal finalizada: window={window}, buffer={buffer}, rmse={rmse:.4f}") 
                            print(f"Tamaño predicción: y_pred_list={len(y_pred_list)}")
                    else:
                        print(f"❌ Regresión Lineal sin suficientes puntos: Window {window}, buffer={buffer}")

                    completed += 1

        df = pd.DataFrame(resultados).sort_values('rmse', ascending=True).reset_index(drop=True)

        return df