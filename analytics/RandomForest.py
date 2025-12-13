import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from metrics.ErrorMetrics import ErrorMetrics
from sklearn.exceptions import DataConversionWarning


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

    def _build_model(self):
        '''Inicializa el modelo Random Forest.'''
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
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
        # Asegurar que la ventana esté en el formato correcto para el modelo (downsampling)
        sampled_window = window[::self.sampling_rate][-self.window:]
        sampled_window = np.array(sampled_window).reshape(1, -1)  # Convertir la ventana en array 2D

        # Obtener la predicción para esta ventana
        prediction = self.model.predict(sampled_window)

        return prediction[0]

    def optimize(self, data, progress_callback, data_callback=None,
                 window_sizes=[8], buffer_sizes=[64],
                 estimator_options=[400], depth_options=[5], sample_split_options=[5],
                 sampling_rate=3, horizon=1, verbose=False):
        '''Optimizar hiperparámetros con división secuencial del conjunto de datos'''
        best_rmse = float('inf')
        best_params = {}

        completed = 0
        total = (len(window_sizes) * len(buffer_sizes) * 
                len(estimator_options) * len(depth_options) * len(sample_split_options)) 

        # Para no saturar: usar máximo 2000 datos
        data_array = data['<CLOSE>'].values[-2000:]

        # División secuencial del dataset
        split_index = int(len(data_array) * 0.7)
        train_data = data_array[:split_index]
        val_data = data_array[split_index:]

        for window in window_sizes:
            self.window = window

            for buffer in buffer_sizes:
                self.buffer_size = buffer

                for n_est in estimator_options:
                    self.n_estimators = n_est

                    for depth in depth_options:
                        self.max_depth = depth

                        for sample_split in sample_split_options:
                            self.min_samples_split = sample_split

                            self.model = self._build_model()

                            y_pred_list = []
                            y_test_list = []

                            # Total de pasos en la simulación en tiempo real
                            steps = len(val_data)

                            # Validar en validation pero empezando desde los ultimos training
                            for i in range(steps):
                                if len(train_data) < buffer:
                                    print(f"❌ Random Forest sin suficientes puntos: train_data={len(train_data)}, buffer={buffer}")
                                    continue
                        
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
                                X_train, y_train = self.prepare_data(data=current_train_data, window=window)

                                # Entrenar
                                self.train(X_train, y_train)

                                # Preparar ventana para predicción del siguiente valor
                                last_input = current_data[-(window * sampling_rate + horizon) : -horizon]  # Última ventana de entrada
                                y_true = current_data[-1]
                                
                                y_pred = self.predict(last_input)
                                
                                # Guardar resultados
                                y_pred_list.append(y_pred)
                                y_test_list.append(y_true)

                                # Asegurar que la ventana esté en el formato correcto para imprimir
                                sampled_window = last_input[::self.sampling_rate][-self.window:]
                                sampled_window = np.array(sampled_window).reshape(1, -1)  # Convertir la ventana en array 2D

                                # Mostrar datos de entrenamiento por pantalla
                                if verbose and i==5 and data_callback is not None:
                                    print(f" Calculando predicción con len(window)={self.window}, len(last_input)={len(last_input)}, len(current_train_data)={len(current_train_data)}")
                                    data_callback("entrenamiento", i, X_train, y_train)
                                    data_callback("validacion", i, sampled_window, y_pred)
                                    data_callback("errores (entrada=predicción, salida=reales)", i, y_pred_list, y_test_list)
                                
                            if len(y_pred_list) > 0 and len(y_test_list) > 0:
                                # Calcular error
                                error_metrics = ErrorMetrics(y_test_list, y_pred_list)
                                avg_rmse = error_metrics.rmse()

                                if avg_rmse < best_rmse:
                                    best_rmse = avg_rmse
                                    best_params = {
                                        'buffer': buffer,
                                        'window': window,
                                        'n_estimators': n_est,
                                        'max_depth': depth,
                                        'min_sample_split': sample_split,
                                        'best_rmse': best_rmse,
                                    }

                                if verbose:
                                    print(f" Random Forest probada: buffer={buffer}, window={window}, n_estimators={n_est}, max_depth={depth}, min_sample_split={sample_split}, avg_rmse={avg_rmse:.4}")
                            else:
                                avg_rmse = float('inf')
                                print(f"❌ Random Forest sin suficientes puntos: buffer={buffer}, window={window}, n_estimators={n_est}, max_depth={depth}, min_sample_split={sample_split}, avg_rmse={avg_rmse:.4}")

                            completed += 1
                            progress_callback(completed / total, f"🔎 Optimizando red neuronal: Probando combinación {completed} de {total} datos")

        print(f'🟢 Mejor Random Forest: {best_params}')

        return best_params








