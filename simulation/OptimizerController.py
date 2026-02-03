from analytics.Indicators import Indicators
from analytics.MultiOutputRegression import MultiOutputRegression
from analytics.NeuralNetwork import NeuralNetwork
from analytics.RandomForest import RandomForest

class OptimizerController:
    @staticmethod
    def optimize_rsi(data, progress_callback=None):
        return Indicators.optimize_rsi(data, progress_callback=progress_callback, verbose=False)

    @staticmethod
    def optimize_macd(data, progress_callback=None):
        return Indicators.optimize_macd(data, progress_callback=progress_callback, verbose=False)

    @staticmethod
    def optimize_regression(data, progress_callback=None, data_callback=None, frecuencia_reentrenamiento=24):
        return MultiOutputRegression.optimize(data=data, progress_callback=progress_callback, data_callback=data_callback, 
                                              frecuencia_reentrenamiento=frecuencia_reentrenamiento, verbose=False)

    @staticmethod
    def optimize_neural_network(data, progress_callback=None, data_callback=None, frecuencia_reentrenamiento=24):
        model = NeuralNetwork()
        return model.optimize(data=data, progress_callback=progress_callback, data_callback=data_callback, 
                              frecuencia_reentrenamiento=frecuencia_reentrenamiento, verbose=True)
    
    @staticmethod
    def optimize_random_forest(data, progress_callback=None, data_callback=None, frecuencia_reentrenamiento=24):
        model = RandomForest()
        return model.optimize(data=data, progress_callback=progress_callback, data_callback=data_callback, 
                              frecuencia_reentrenamiento=frecuencia_reentrenamiento, verbose=True)
