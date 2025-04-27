from tensorflow import keras
from metrics.ErrorMetrics import ErrorMetrics
from tensorflow.keras.callbacks import EarlyStopping    # type: ignore

import numpy as np

class NeuralNetwork:
    def __init__(self, input_shape=24, hidden_neurons=64, epochs=100, batch_size=32):
        self.input_shape = input_shape
        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        self.model = self._build_model()

    def prepare_data(self, data, window=None, horizon=3):
        X, y = [], []

        if not window:
            window = self.input_shape
        
        for i in range(len(data) - window - horizon):
            x_window = data[i:i+window]
            future_value = data[i+window+horizon]

            # Predecimos el cambio respecto al último valor de la ventana
            y_target = future_value - x_window[-1]

            X.append(x_window)
            y.append(y_target)
        
        return np.array(X), np.array(y)
    
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(self.hidden_neurons, activation='relu', input_shape=(self.input_shape,)),
            keras.layers.Dense(self.hidden_neurons, activation='relu'),
            keras.layers.Dense(1)  # Salida para predicción de precios
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, y_train, epochs=None, batch_size=None):
        history = self.model.fit(
            X_train, y_train, 
            epochs=epochs if epochs else self.epochs,  
            batch_size=batch_size if batch_size else self.batch_size, 
            callbacks=[self.early_stop], verbose=0
            )

        return history

    def predict(self, X):
        return self.model.predict(X).flatten()

    def optimize(self, data, progress_callback, scaler, 
                 window_options = [4, 8, 12, 16], hidden_neurons_options=[32, 48, 64], 
                 learning_rates_options=[0.001, 0.005, 0.01], batch_sizes=[4, 8, 16], 
                 max_epochs=100, train_ratio=0.6, horizon=3):
        """Optimizar hiperparámetros con división secuencial del conjunto de datos"""
        best_rmse = float('inf')
        best_params = {}

        completed = 0
        total = len(window_options) * len(hidden_neurons_options) * len(learning_rates_options) * len(batch_sizes)

        # Para no saturar al sistema con el entrenamiento, se usan máximo los ultimos 200 datos
        data = data.iloc[-200:].copy()

        # División secuencial del dataset
        split_index = int(len(data) * train_ratio)
        train_data = data.iloc[:split_index]['<CLOSE>'].values
        val_data = data.iloc[split_index:]['<CLOSE>'].values

        # Escalar para normalizar los datos
        train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        val_data_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()

        for window in window_options:
            self.input_shape = window

            # Usar los datos escalados para preparar X e y
            X_train, y_train = self.prepare_data(train_data_scaled, window, horizon)
            X_val, y_val = self.prepare_data(val_data_scaled, window, horizon)

            # Depurar si no hay suficientes datos
            if len(X_train) <= 30 or len(X_val) <= 10:
                print("⚠️ Problema detectado:")
                print(f"- Tamaño total de data: {len(data)}")
                print(f"- Tamaño train_data: {len(train_data)}")
                print(f"- Tamaño val_data: {len(val_data)}")
                print(f"- Tamaño X_train: {len(X_train)}")
                print(f"- Tamaño X_val: {len(X_val)}")
                print(f"- Parámetros usados: window={window}, horizon={horizon}")
                print(f"- No hay suficientes datos para esta combinación, saltando...\n")
                continue

            for neurons in hidden_neurons_options:
                self.hidden_neurons = neurons
                self.model = self._build_model()

                for lr in learning_rates_options:
                    completed += len(batch_sizes)
                    progress_callback(completed / total, f"🔎 Optimizando red neuronal: Probando combinación {completed} de {total} datos")

                    for batch in batch_sizes:
                            # Modificar hiperparámetros y reconstruir modelo
                            self.learning_rate = lr
                            self.batch_size = batch
                            self.epochs = max_epochs

                            # Entrenar con datos escalados
                            history = self.train(X_train, y_train)
                            epochs = len(history.history['loss'])   # Guardar epocas usadas hasta stop

                            # Calcular predicciones con datos escalados
                            predicciones_val = self.predict(X_val)

                            # Invertir escala
                            predicciones_val_real = scaler.inverse_transform(predicciones_val.reshape(-1,1)).flatten()
                            y_val_real = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()

                            # Calcular RMSE sobre precios reales
                            error_metrics = ErrorMetrics(y_val_real, predicciones_val_real)
                            avg_rmse = error_metrics.rmse()

                            print(f"Neural Network probada: window={window}, neurons={neurons}, learning_rate={lr}, batch_size={batch}, epochs={epochs}, avg_rmse={avg_rmse:.4}")

                            if avg_rmse < best_rmse:
                                best_rmse = avg_rmse
                                best_params = {
                                    'window': window,
                                    'neurons': neurons,
                                    'learning_rate': lr,
                                    'batch_size': batch,
                                    'epochs': epochs
                                }

        print(f'Mejores parámetros: {best_params}')

        if best_params:
            # Aplicar los mejores parámetros
            self.input_shape = best_params['window']
            self.hidden_neurons = best_params['neurons']
            self.learning_rate = best_params['learning_rate']
            self.batch_size = best_params['batch_size']
            self.epochs = best_params['epochs']
            self.model = self._build_model()  # Crear modelo final con los mejores hiperparámetros

        return best_params








