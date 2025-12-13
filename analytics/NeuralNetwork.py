from tensorflow import keras
from tensorflow.keras.optimizers import Adam    # type: ignore
from tensorflow.keras.layers import Normalization   # type: ignore
from metrics.ErrorMetrics import ErrorMetrics
from tensorflow.keras.callbacks import EarlyStopping    # type: ignore

import numpy as np

class NeuralNetwork:
    def __init__(self, buffer_size=24, input_shape=24, learning_rate=0.001, hidden_neurons=64, batch_size=32, epoch=100, sampling_rate=3):
        self.buffer_size = buffer_size
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.hidden_neurons = hidden_neurons
        self.batch_size = batch_size
        self.epochs = epoch
        self.sampling_rate = sampling_rate  # Cada cuántos puntos usar (1 = todos, 3 = 1 de cada 3)
        self.early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        self.model = self._build_model()

    def prepare_data(self, data, window=None, horizon=1):
        X, y = [], []

        if not window:
            window = self.input_shape

        # Downsampling: usar solo 1 de cada N precios
        sampled_data = data[::self.sampling_rate]  
        
        for i in range(len(sampled_data) - window - horizon + 1):
            X.append(sampled_data[i : i + window])
            y.append(sampled_data[i + window : i + window + horizon])
        
        return np.array(X), np.array(y)
    
    def _build_model(self):
        optimizer = Adam(learning_rate=self.learning_rate)

        model = keras.Sequential([
            keras.layers.Dense(self.hidden_neurons, activation='elu', input_shape=(self.input_shape,)),
            keras.layers.Dense(self.hidden_neurons, activation='elu'),
            keras.layers.Dense(1)  # Salida para predicción de precios
        ])

        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def train(self, X_train, y_train):
        history = self.model.fit(
            X_train, y_train, 
            epochs=self.epochs,  
            batch_size=self.batch_size, 
            callbacks=[self.early_stop],
            verbose = 0     # type: ignore
            )

        return history

    def predict(self, X):
        '''Predice los próximos "horizon" valores usando la ventana de entrada.'''

        # Aplicar downsampling a cada subarray individualmente
        sampled_window = np.array([subarr[::self.sampling_rate] for subarr in X])

        if len(X[-1]) < self.input_shape * self.sampling_rate:
            raise ValueError(f"Longitud de window es de {len(X[-1])} puntos. Se necesitan al menos {self.input_shape * self.sampling_rate} puntos")
        
        return self.model.predict(sampled_window)

    def optimize(self, data, progress_callback, data_callback=None,
                 window_options=[1, 4], hidden_neurons_options=[16, 32, 64],
                 learning_rates_options=[0.0001, 0.0005], batch_sizes=[2, 4], buffer_sizes=[50, 100, 150],
                 horizon=1, verbose=False):
        """Optimizar hiperparámetros con división secuencial del conjunto de datos"""
        best_rmse = float('inf')
        best_params = {}

        completed = 0
        total = len(window_options) * len(hidden_neurons_options) * len(learning_rates_options) * len(batch_sizes) * len(buffer_sizes)

        # Para no saturar: usar máximo 2000 datos
        data_array = data['<CLOSE>'].values[-2000:]

        # División secuencial del dataset
        split_index = int(len(data_array) * 0.7)
        train_data = data_array[:split_index]
        val_data = data_array[split_index:]

        for window in window_options:
            self.input_shape = window

            for neurons in hidden_neurons_options:
                self.hidden_neurons = neurons

                for lr in learning_rates_options:
                    completed += (len(batch_sizes) * len(buffer_sizes))
                    progress_callback(completed / total, f"🔎 Optimizando red neuronal: Probando combinación {completed} de {total} datos")

                    self.learning_rate = lr
                    self.model = self._build_model()

                    for batch in batch_sizes:
                        self.batch_size = batch
                    
                        for buffer in buffer_sizes:

                            if len(train_data) < buffer:
                                print(f"❌ Neural Network sin suficientes puntos: train_data={len(train_data)}, buffer={buffer}")
                                continue
                    
                            # Número de pasos por delante
                            pasos_por_delante = horizon * self.sampling_rate
                    
                            # Número real de puntos a coger
                            train_size = window * self.sampling_rate + buffer

                            # Coger últimos puntos de training hasta pasos_por_delante
                            train_window = train_data[-train_size : -pasos_por_delante + 1]

                            # Preparar entradas (ventanas) sobre esos datos
                            X_train, y_train = self.prepare_data(data=train_window, window=window)

                            # Entrenar
                            self.train(X_train, y_train)

                            # Preparar ventanas de validación
                            full_data = np.concatenate([train_data[-(window * self.sampling_rate + horizon):], val_data])
                            full_window = window * self.sampling_rate
                            X_val, y_val = [], []

                            for i in range(len(full_data) - full_window):
                                X_val.append(full_data[i : i + full_window])
                                y_val.append(full_data[i + full_window : i + full_window + horizon])
                            
                            y_pred = self.predict(X_val)

                            y_pred_list = [pred[-1] for pred in y_pred]
                            y_val_list = [val[-1] for val in y_val]
                            
                            if len(y_pred_list) > 0 and len(y_val_list) > 0:
                                # Calcular error
                                error_metrics = ErrorMetrics(y_val_list, y_pred_list)
                                avg_rmse = error_metrics.rmse()

                                if avg_rmse < best_rmse:
                                    best_rmse = avg_rmse
                                    best_params = {
                                        'buffer': buffer,
                                        'window': window,
                                        'neurons': neurons,
                                        'learning_rate': lr,
                                        'batch_size': batch
                                    }

                                if verbose:
                                    print(f" Neural Network probada: buffer={buffer}, window={window}, neurons={neurons}, learning_rate={lr}, batch_size={batch}, avg_rmse={avg_rmse:.4}")
                            else:
                                avg_rmse = float('inf')
                                print(f"❌ Neural Network sin suficientes puntos: buffer={buffer}, window={window}, neurons={neurons}, learning_rate={lr}, batch_size={batch}, avg_rmse={avg_rmse:.4}")

        print(f'🟢 Mejores parámetros: {best_params}')

        return best_params








