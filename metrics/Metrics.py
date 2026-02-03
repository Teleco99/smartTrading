import pandas as pd

class Metrics:
    def __init__(self, resultados, capital_por_operacion=1000):
        # Lista de operaciones simuladas y capital invertido en cada una
        self.resultados = resultados
        self.capital_por_operacion = capital_por_operacion

    def _extraer_ganancias(self):
        # Extrae las ganancias individuales de cada operación
        return [operacion['ganancia'] for operacion in self.resultados]

    def balance_inicial(self):
        # Calcula el capital total invertido (número de operaciones * capital por operación)
        return self.capital_por_operacion * self.numero_de_operaciones()

    def balance_final(self):
        # Balance final = capital invertido + beneficio neto obtenido
        return self.balance_inicial() + self.beneficio_neto()
    
    def numero_de_operaciones(self):
        # Total de operaciones simuladas
        return len(self.resultados)

    def profit_factor(self):
        # Suma de ganancias positivas
        ganancias = [g for g in self._extraer_ganancias() if g > 0]
        # Suma de pérdidas (convertidas a positivas)
        perdidas = [-g for g in self._extraer_ganancias() if g < 0]
        suma_ganancias = sum(ganancias)
        suma_perdidas = sum(perdidas)

        # Si no hay pérdidas, el PF es infinito
        if suma_perdidas == 0:
            return float("inf") if suma_ganancias > 0 else 1.0
        
        return suma_ganancias / suma_perdidas
    
    def return_factor(self):
        # Balance final = capital invertido + beneficio neto obtenido
        if self.balance_inicial() == 0 or self.balance_final() == 0:
            return 0
        else:
            return self.balance_final() / self.balance_inicial()

    def drawdown_maximo(self):
        # Ganancias acumuladas a lo largo del tiempo
        ganancias_acumuladas = pd.Series(self._extraer_ganancias()).cumsum()

        # Punto máximo acumulado hasta cada momento
        pico = ganancias_acumuladas.cummax()

        # Diferencia entre el pico y la ganancia acumulada (drawdown)
        drawdown_dinero = pico - ganancias_acumuladas

        # Mayor drawdown observado
        return drawdown_dinero.max()

    def beneficio_neto(self):
        # Suma total de las ganancias (positivas y negativas)
        return sum(self._extraer_ganancias())

    def beneficio_promedio_por_operacion(self):
        # Beneficio medio por cada operación simulada
        ganancias = self._extraer_ganancias()
        if len(ganancias) == 0:
            return 0
        return sum(ganancias) / len(ganancias)

    def porcentaje_operaciones_rentables(self):
        # Calcula el porcentaje de operaciones con ganancia positiva
        ganancias = self._extraer_ganancias()
        operaciones_ganadoras = [g for g in ganancias if g > 0]
        if len(ganancias) == 0:
            return 0
        return len(operaciones_ganadoras) / len(ganancias) * 100

    def resumen(self):
        # Devuelve todas las métricas en un diccionario resumen
        return {
            'Balance Inicial (€)': self.balance_inicial(),
            'Balance Final (€)': self.balance_final(),
            'Número de Operaciones': self.numero_de_operaciones(),
            'Return Factor': self.return_factor(),
            'Profit Factor': self.profit_factor(),
            'Drawdown Máximo (€)': self.drawdown_maximo(),
            'Beneficio Neto (€)': self.beneficio_neto(),
            'Beneficio Promedio por Operación (€)': self.beneficio_promedio_por_operacion(),
            'Porcentaje de Operaciones Rentables (%)': self.porcentaje_operaciones_rentables()
        }
