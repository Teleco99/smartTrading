from tensorflow import keras
from tensorflow.keras.optimizers import Adam    # type: ignore
from metrics.ErrorMetrics import ErrorMetrics
from tensorflow.keras.callbacks import EarlyStopping    # type: ignore

import numpy as np

class NeuralNetwork:
    def __init__(self, buffer_size=20, input_shape=24, learning_rate=0.001, hidden_neurons=64, batch_size=32, epoch=100):
        self.buffer_size = buffer_size
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.hidden_neurons = hidden_neurons
        self.batch_size = batch_size
        self.epochs = epoch
        self.early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        self.model = self._build_model()

    def prepare_data(self, data, window=None, horizon=1):
        X, y = [], []

        if not window:
            window = self.input_shape
        
        for i in range(len(data) - window - horizon + 1):
            x_window = data[i : i + window]
            y_target = data[i + window : i + window + horizon]

            X.append(x_window)
            y.append(y_target)
        
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
            callbacks=[self.early_stop], verbose=0
            )

        return history

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, data, progress_callback, 
                 window_options=[12, 18], hidden_neurons_options=[16, 32],
                 learning_rates_options=[0.0001, 0.0005], batch_sizes=[4, 8], buffer_sizes=[12, 18],
                 horizon=1):
        """Optimizar hiperparámetros con división secuencial del conjunto de datos"""
        best_rmse = float('inf')
        best_params = {}

        completed = 0
        total = len(window_options) * len(hidden_neurons_options) * len(learning_rates_options) * len(batch_sizes) * len(buffer_sizes)

        # Para no saturar: usar máximo 100 datos
        data_array = data['<CLOSE>'].values[-200:]

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
                            # Número real de puntos a coger
                            train_size = window + buffer + horizon

                            # Coger últimos puntos de training hasta horizon
                            train_window = train_data[-train_size : -horizon]

                            # Preparar entradas (ventanas) sobre esos datos
                            X_train, y_train = self.prepare_data(data=train_window, window=window)

                            # Entrenar
                            self.train(X_train, y_train)

                            # Preparar ventanas de validación
                            full_data = np.concatenate([train_data[-(window + horizon - 1):], val_data])
                            X_val, y_val = self.prepare_data(full_data, window=window)

                            y_pred = self.predict(X_val)

                            y_pred_list = [pred[-1] for pred in y_pred]
                            y_val_list = [val[-1] for val in y_val]

                            print(f"  Optimizando modelo...")
                            print(f"  📈 X_train[-1]: {X_train[-1]}")
                            print(f"  📈 y_train[-1]: {y_train[-1]}")
                            print(f"  📈 X_val[-1]: {X_val[-1]}")
                            print(f"  🎯 y_val_list[-1]: {y_val_list[-1]}")
                            print(f"  🤖 y_pred_list[-1]: {y_pred_list[-1]}")
                            print("-" * 40)
                            
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

                                print(f"✅ Neural Network probada: buffer={buffer}, window={window}, neurons={neurons}, learning_rate={lr}, batch_size={batch}, avg_rmse={avg_rmse:.4}")
                            else:
                                avg_rmse = float('inf')
                                print(f"❌ Neural Network sin suficientes puntos: buffer={buffer}, window={window}, neurons={neurons}, learning_rate={lr}, batch_size={batch}, avg_rmse={avg_rmse:.4}")

        print(f'🟢 Mejores parámetros: {best_params}')

        return best_params








