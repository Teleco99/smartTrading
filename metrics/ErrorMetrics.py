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

    def get_all_metrics(self):
        """Retorna todas las métricas de error."""
        return {
            "RMSE": self.rmse(),
            "MSE": self.mse(),
            "MAPE": self.mape(),
            "SMAPE": self.smape()
        }
