import numpy as np
import pandas as pd

class ErrorMetrics:
    def __init__(self, actual_values, predicted_values):
        """
        Inicializa la clase con los valores reales y las predicciones.
        
        actual_values: Array o lista de los valores reales.
        predicted_values: Array o lista de los valores predichos.
        """
        if len(actual_values) != len(predicted_values):
            raise ValueError("Los conjuntos de datos deben tener el mismo tamaño.")
        
        actual_values = np.array(actual_values)
        predicted_values = np.array(predicted_values)

        # Filtrar valores None o NaN
        mask = ~np.isnan(predicted_values)  # Crea una máscara donde no haya NaN
        self.actual = actual_values[mask]
        self.predicted = predicted_values[mask]

    def rmse(self):
        """Calcular RMSE (Root Mean Squared Error)."""
        return np.sqrt(((self.actual - self.predicted) ** 2).mean())

    def mse(self):
        """Calcular MSE (Mean Squared Error)."""
        return ((self.actual - self.predicted) ** 2).mean()

    def mape(self):
        """Calcular MAPE (Mean Absolute Percentage Error)."""
        return np.mean(np.abs((self.actual - self.predicted) / self.actual)) * 100

    def smape(self):
        """Calcular SMAPE (Symmetric Mean Absolute Percentage Error)."""
        denominator = np.abs(self.actual) + np.abs(self.predicted)
        diff = np.abs(self.actual - self.predicted) / denominator
        diff[denominator == 0] = 0  # Manejo de división por cero
        return 2 * np.mean(diff) * 100
    
    def directional_accuracy_price_thr(self, H=3, eps=0.0):
        """
        Directional Accuracy con umbral eps, cuando self.actual y self.predict son PRECIOS
        del objetivo (t+H) alineados.

        H: offset real (horizon * sampling_rate)
        eps: umbral (en unidades de precio) para considerar lateral (0)
        """
        y_true_future = np.asarray(self.actual).reshape(-1)
        y_pred_future = np.asarray(self.predicted).reshape(-1)

        if len(y_true_future) != len(y_pred_future):
            raise ValueError("self.actual y self.predict deben tener la misma longitud")
        if H <= 0 or H >= len(y_true_future):
            raise ValueError("H debe ser > 0 y menor que la longitud de los arrays")

        # Precio base (t) correspondiente a cada objetivo (t+H)
        current = y_true_future[:-H]

        # Objetivo real y predicho en (t+H)
        y_true = y_true_future[H:]
        y_pred = y_pred_future[H:]

        # Movimientos (respecto a current)
        real_move = y_true - current
        pred_move = y_pred - current

        # Direcciones con umbral
        real_dir = np.where(real_move > eps, 1, np.where(real_move < -eps, -1, 0))
        pred_dir = np.where(pred_move > eps, 1, np.where(pred_move < -eps, -1, 0))

        return float(np.mean(real_dir == pred_dir))

    def directional_accuracy_price_quantile(self, H=3, q=0.8):
        y_true_future = np.asarray(self.actual).reshape(-1)
        y_pred_future = np.asarray(self.predicted).reshape(-1)

        if len(y_true_future) != len(y_pred_future):
            raise ValueError("self.actual y self.predict deben tener la misma longitud")
        if H <= 0 or H >= len(y_true_future):
            raise ValueError("H debe ser > 0 y menor que la longitud de los arrays")

        # Precio base (t) correspondiente a cada objetivo (t+H)
        current = y_true_future[:-H]

        # Objetivo real y predicho en (t+H)
        y_true = y_true_future[H:]
        y_pred = y_pred_future[H:]

        # Movimientos (respecto a current)
        real_move = y_true - current
        pred_move = y_pred - current
        
        abs_pred = np.abs(pred_move)
        threshold = np.quantile(abs_pred, q)

        mask = abs_pred >= threshold
        if mask.sum() == 0:
            return np.nan

        return np.mean(np.sign(real_move[mask]) == np.sign(pred_move[mask]))

    def get_all_metrics(self):
        """Retorna todas las métricas de error."""
        return {
            "RMSE": self.rmse(),
            "MSE": self.mse(),
            "MAPE": self.mape(),
            "SMAPE": self.smape()
        }
