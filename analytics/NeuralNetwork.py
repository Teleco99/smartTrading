from tensorflow import keras
from tensorflow.keras.optimizers import Adam    # type: ignore
from tensorflow.keras.layers import Normalization   # type: ignore
from metrics.ErrorMetrics import ErrorMetrics
from tensorflow.keras.callbacks import EarlyStopping    # type: ignore

import numpy as np
import pandas as pd
import os, random
import tensorflow as tf
from typing import Any

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

        self.set_determinism()

        self.model = self._build_model()

    def set_determinism(self, seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

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
        seed = 42
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"  # más determinismo
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        optimizer = Adam(learning_rate=self.learning_rate)

        init1: Any = keras.initializers.GlorotUniform(seed=seed)     
        init2: Any = keras.initializers.GlorotUniform(seed=seed+1)   
        init3: Any = keras.initializers.GlorotUniform(seed=seed+2)   

        model = keras.Sequential([
            keras.layers.Dense(self.hidden_neurons, activation='elu',
                            input_shape=(self.input_shape,),
                            kernel_initializer=init1,
                            bias_initializer="zeros"),
            keras.layers.Dense(self.hidden_neurons, activation='elu',
                            kernel_initializer=init2,
                            bias_initializer="zeros"),
            keras.layers.Dense(1,
                            kernel_initializer=init3,
                            bias_initializer="zeros")
        ])

        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def train(self, X_train, y_train):
        history = self.model.fit(
            X_train, y_train, 
            epochs=self.epochs,  
            batch_size=self.batch_size, 
            callbacks=[self.early_stop],
            shuffle=False,   # No mezclar datos temporales
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
                 buffer_sizes=range(100, 500, 50), learning_rates_options=[0.0001, 0.0005],
                 batch_sizes=[2, 4], hidden_neurons_options=[8, 16, 32],
                 frecuencia_reentrenamiento = 24, horizon=1, verbose=False):
        """Optimizar hiperparámetros con división secuencial del conjunto de datos"""
        resultados = []

        window = 1
        completed = 0
        total = len(buffer_sizes) * len(hidden_neurons_options) * len(learning_rates_options) * len(batch_sizes) 

        # Para no saturar: usar máximo 2000 datos
        data_array = data['<CLOSE>'].values[-2000:]

        # División secuencial del dataset
        split_index = int(len(data_array) * 0.7)
        train_data = data_array[:split_index]
        val_data = data_array[split_index:]

        self.input_shape = window
        for buffer in buffer_sizes:
            for neurons in hidden_neurons_options:
                self.hidden_neurons = neurons

                for lr in learning_rates_options:
                    self.learning_rate = lr
                    self.model = self._build_model()

                    for batch in batch_sizes:
                        self.batch_size = batch
                    
                        progress_callback(completed / total, f"🔎 Optimizando red neuronal: Probando combinación {completed} de {total}")
                            
                        if len(train_data) < window * self.sampling_rate + 100:
                            print(f"❌ Neural Network sin suficientes puntos: train_data={len(train_data)}")
                            continue

                        if len(val_data) < window * self.sampling_rate + horizon:
                            print(f"❌ Neural Network sin suficientes puntos: val_data={len(val_data)} (warmup insuficiente)")
                            continue

                        y_pred_list = []
                        y_val_list = []

                        full_window = window * self.sampling_rate

                        train_size = window * self.sampling_rate + buffer
                        current_train_data = train_data[-train_size:]

                        # Preparar entradas y salidas de datos de entrenamiento
                        X_train, y_train = self.prepare_data(data=current_train_data, window=window)
                        
                        # Entrenar
                        self.train(X_train, y_train)

                        # Validar
                        end_limit = len(val_data) - horizon + 1
                        i = full_window
                        while i < end_limit:
                            # Predicciones del tramo actual antes de reentreno
                            next_retrain = i + frecuencia_reentrenamiento if frecuencia_reentrenamiento != 0 else end_limit
                            j_end = min(next_retrain, end_limit)

                            X_val = []
                            y_val = []
                            for j in range(i, j_end):
                                X_val.append(val_data[j - full_window : j])
                                y_val.append(val_data[j + horizon - 1])

                            y_pred = self.predict(X_val)
                            y_pred_list.extend([pred[-1] for pred in y_pred])
                            y_val_list.extend(y_val)

                            # Reentreno (warm start)
                            if frecuencia_reentrenamiento != 0 and j_end < end_limit:
                                # Datos disponibles hasta ahora
                                available = np.concatenate([train_data, val_data[:i]])
                                current_train_data = available[-train_size:]

                                X_train, y_train = self.prepare_data(data=current_train_data, window=window)

                                self.train(X_train, y_train)

                            i = j_end
                        
                        if len(y_pred_list) > 0 and len(y_val_list) > 0:
                            # Calcular error
                            error_metrics = ErrorMetrics(y_val_list, y_pred_list)
                            rmse = error_metrics.rmse()
                            directional_acc_thr = error_metrics.directional_accuracy_price_thr(eps=100.0)
                            directional_acc_quantile = error_metrics.directional_accuracy_price_quantile(q=0.8)

                            resultados.append({
                                'buffer': buffer,
                                'neurons': neurons,
                                'learning_rate': lr,
                                'batch_size': batch,
                                'rmse': rmse,
                                'directional_acc_thr': directional_acc_thr,
                                'directional_acc_quantile': directional_acc_quantile
                            })

                            if verbose:
                                print(f"✅ Neural Network finalizada: window={window}, neurons={neurons}, learning_rate={lr}, batch_size={batch}, rmse={rmse:.4}")
                                print(f"Tamaño predicción: y_pred_list={len(y_pred_list)}")
                        else:
                            rmse = float('inf')
                            print(f"❌ Neural Network sin suficientes puntos: window={window}, neurons={neurons}, learning_rate={lr}, batch_size={batch}, rmse={rmse:.4}")

                        completed += 1

        df = pd.DataFrame(resultados).sort_values('rmse', ascending=True).reset_index(drop=True)

        return df








