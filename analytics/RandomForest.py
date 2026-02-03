from sklearn.ensemble import RandomForestRegressor
from metrics.ErrorMetrics import ErrorMetrics
from sklearn.exceptions import DataConversionWarning

import pandas as pd
import numpy as np
import warnings

class RandomForest:
    '''Modelo de regresión basado en Random Forest usando solo precios.'''

    def __init__(self, window=24, buffer_size=24, sampling_rate=3, 
                 n_estimators=200, max_depth=5, min_samples_split=10):
        self.buffer_size = buffer_size
        self.window = window
        self.sampling_rate = sampling_rate  # Cada cuántos puntos usar (1 = todos, 3 = 1 de cada 3)
        self.n_estimators = n_estimators    # Número de árboles
        self.max_depth = max_depth          # Nivel de profundidad
        self.min_samples_split = min_samples_split  # Sensibilidad a cambios

        self.model = self._build_model()

        warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    def _build_model(self, warm_start=False):
        '''Inicializa el modelo Random Forest.'''
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            warm_start=warm_start,
            random_state=42
        )

        return model

    def prepare_data(self, data, window=None, horizon=1):
        '''
        Genera X e y usando únicamente el precio.
        Downsampling + ventanas + horizonte multioutput.
        '''
        X, y = [], []

        if not window:
            window = self.window

        # Downsampling: usar solo 1 de cada N precios
        sampled_data = data[::self.sampling_rate] 
        
        for i in range(len(sampled_data) - window - horizon + 1):
            X.append(sampled_data[i : i + window])
            y.append(sampled_data[i + window : i + window + horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, X_data, y_data):
        '''Entrena el modelo con las ventanas generadas.'''
        if len(X_data) > 0:
            self.model.fit(X_data, y_data)
        else:
            print("❌ Datos insuficientes después de procesar las ventanas.")

    def predict(self, window):
        '''Predice el próximo valor usando la ventana de entrada.'''
        # Soporta entrada individual (1D) o batch (iterable de ventanas)
        arr = np.array(window)

        # Entrada 1D -> devolver un escalar
        if arr.ndim == 1:
            sampled_window = arr[::self.sampling_rate][-self.window:]
            sampled_window = np.array(sampled_window).reshape(1, -1)
            prediction = self.model.predict(sampled_window)
            return prediction[0]

        # Entrada 2D -> batch
        sampled_batch = np.array([np.array(w)[::self.sampling_rate][-self.window:] for w in arr])
        preds = self.model.predict(sampled_batch)
        return preds

    def optimize(self, data, progress_callback, data_callback=None, 
                 buffer_sizes=range(100, 500, 50), window_sizes=[1, 2, 4], 
                 estimator_options=[250, 500], depth_options=[4, 5, 6], sample_split_options=[2],
                 frecuencia_reentrenamiento=24, sampling_rate=3, horizon=1, verbose=False):
        '''Optimizar hiperparámetros con división secuencial del conjunto de datos'''
        resultados = []

        completed = 0
        total = len(buffer_sizes) * len(window_sizes) * len(estimator_options) * len(depth_options) * len(sample_split_options) 

        # Para no saturar: usar máximo 2000 datos
        data_array = data['<CLOSE>'].values[-2000:]

        # División secuencial del dataset
        split_index = int(len(data_array) * 0.7)
        train_data = data_array[:split_index]
        val_data = data_array[split_index:]

        for buffer in buffer_sizes:
            for window in window_sizes:
                self.window = window

                for n_est in estimator_options:
                    self.n_estimators = n_est

                    for depth in depth_options:
                        self.max_depth = depth

                        for sample_split in sample_split_options:
                            progress_callback(completed / total, f"🔎 Optimizando Random Forest: Probando combinación {completed} de {total}")

                            self.min_samples_split = sample_split
                            self.model = self._build_model()

                            if len(train_data) < window * sampling_rate + 100:
                                print(f"❌ Random Forest sin suficientes puntos: train_data={len(train_data)}")
                                completed += 1
                                continue

                            if len(val_data) < window * sampling_rate + horizon:
                                print(f"❌ Random Forest sin suficientes puntos: val_data={len(val_data)} (warmup insuficiente)")
                                completed += 1
                                continue

                            y_pred_list = []
                            y_val_list = []

                            full_window = window * sampling_rate

                            train_size = window * sampling_rate + buffer
                            current_train_data = train_data[-train_size:]

                            # Preparar entradas y salidas de datos de entrenamiento
                            X_train, y_train = self.prepare_data(data=current_train_data, window=window)

                            # Entrenar
                            self.train(X_train, y_train)

                            # Validar
                            end_limit = len(val_data) - horizon + 1
                            for i in range(full_window, end_limit):
                                # Reentreno cada frecuencia_reentrenamiento
                                if frecuencia_reentrenamiento != 0 and i % frecuencia_reentrenamiento == 0:
                                    # Datos disponibles hasta ahora
                                    available = np.concatenate([train_data, val_data[:i]])
                                    current_train_data = available[-train_size:]

                                    X_train, y_train = self.prepare_data(data=current_train_data, window=window)
                                    
                                    self.train(X_train, y_train)

                                last_input = val_data[i - full_window : i]
                                y_true = val_data[i + horizon - 1]

                                y_pred = self.predict(last_input)
                                
                                # Guardar resultados
                                y_pred_list.append(y_pred[0] if isinstance(y_pred, np.ndarray) else y_pred)
                                y_val_list.append(y_true)

                            if len(y_pred_list) > 0 and len(y_val_list) > 0:
                                # Calcular error
                                error_metrics = ErrorMetrics(y_val_list, y_pred_list)
                                rmse = error_metrics.rmse()
                                directional_acc_thr = error_metrics.directional_accuracy_price_thr(eps=100.0)
                                directional_acc_quantile = error_metrics.directional_accuracy_price_quantile(q=0.8)

                                resultados.append({
                                    'buffer': buffer,
                                    'window': window,
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'min_samples_split': sample_split,
                                    'rmse': rmse,
                                    'directional_accuracy_thr': directional_acc_thr,
                                    'directional_accuracy_quantile': directional_acc_quantile,
                                })

                                if verbose:
                                    print(f"✅ Random Forest probada: window={window}, n_estimators={n_est}, max_depth={depth}, min_sample_split={sample_split}, rmse={rmse:.4f}")
                                    print(f"Tamaño predicción: y_pred_list={len(y_pred_list)}")
                            else:
                                rmse = float('inf')
                                print(f"❌ Random Forest sin suficientes puntos: window={window}, n_estimators={n_est}, max_depth={depth}, min_sample_split={sample_split}, rmse={rmse:.4f}")

                            completed += 1

        df = pd.DataFrame(resultados).sort_values('rmse', ascending=True).reset_index(drop=True)
 
        return df








