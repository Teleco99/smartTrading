import pandas as pd

class DataLoader:
    def __init__(self, file_path, freq='5min'):
        self.file_path = file_path
        self.data = None
        self.freq = freq

    def load_data(self):
        self.data = pd.read_csv(self.file_path, delimiter="\t")
        self.data['<DATETIME>'] = pd.to_datetime(self.data['<DATE>'] + ' ' + self.data['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        selected_columns = ['<DATETIME>', '<CLOSE>']
        self.data = self.data[selected_columns]
        self.data.set_index('<DATETIME>', inplace=True)

        # Crear un rango de fechas uniforme basado en el índice actual
        start = self.data.index.min()  # Primer timestamp en el índice
        end = self.data.index.max()    # Último timestamp en el índice

        # Generar un índice con intervalos regulares de 5 minutos
        regular_index = pd.date_range(start=start, end=end, freq=self.freq)

        # Reindexar y crear marcador de interpolación
        self.data = self.data.reindex(regular_index)

        self.data['Interpolado'] = self.data['<CLOSE>'].isna()

        # Interpolar los valores faltantes
        self.data = self.data.interpolate(method='linear')

    def get_filtered_data(self, start_date, end_date):
        return self.data.loc[start_date:end_date].copy()
