import plotly.graph_objects as go
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots

class Visualizer:
    @staticmethod
    def time_to_decimal(t):
        return t.hour + t.minute / 60

    @staticmethod
    def plot_interactive_price(data):
        # start_h = Visualizer.time_to_decimal(start_time)  
        # end_h = Visualizer.time_to_decimal(end_time)

        dates = data.index.normalize().unique()
        all_days = pd.date_range(dates.min(), dates.max(), freq="D")
        missing_days = all_days.difference(dates)
        nan_rows = pd.DataFrame(
            {'<CLOSE>': np.nan},
            index=missing_days
        )
        data_plot = pd.concat([data, nan_rows]).sort_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['<CLOSE>'], mode='lines', name='Precio', connectgaps=False))
        fig.update_layout(title="Visualización de precios", xaxis_title="Fecha", yaxis_title="Precio")

        # Quitamos horas fuera de mercado
        # fig.update_xaxes(
        #     rangebreaks=[
        #         dict(bounds=[end_h, start_h], pattern="hour")
        #     ]
        # )

        return fig
    
    @staticmethod
    def plot_interactive_combined(data, signals=None, operaciones=[], title="Análisis interactivo"):
        # Introducimos Nan en días festivos para evitar interpolación
        dates = data.index.normalize().unique()
        all_days = pd.date_range(dates.min(), dates.max(), freq="D")
        missing_days = all_days.difference(dates)
        nan_rows = pd.DataFrame(
            {'<CLOSE>': np.nan, 'Prediction': np.nan},
            index=missing_days
        )
        data_plot = pd.concat([data, nan_rows]).sort_index()

        # Detectar qué indicadores están presentes
        tiene_rsi = 'RSI' in data.columns and data['RSI'].notna().any()
        tiene_macd = 'MACD' in data.columns and data['MACD'].notna().any() and 'MACD_Signal' in data.columns
        
        rows = 1 + int(tiene_rsi) + int(tiene_macd)

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5] + [0.25] * (rows - 1),
            subplot_titles=["Precio"] +
                (["RSI"] if tiene_rsi else []) +
                (["MACD"] if tiene_macd else [])
        )

        row = 1

        # === Precio ===
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=data_plot['<CLOSE>'],
            name='Precio',
            line=dict(color='blue')
        ), row=row, col=1)
        
        if 'Prediction' in data_plot.columns:
            pred_series = data_plot['Prediction']
            fig.add_trace(go.Scatter(
                x=pred_series.index, 
                y=pred_series.values,
                name='Prediction',
                line=dict(color='orange', dash='dash')
            ), row=row, col=1)

        # === Señales ===
        if signals is not None and 'Operacion' in signals.columns:
            buy_ops = signals[signals['Operacion'] == 1]
            sell_ops = signals[signals['Operacion'] == -1]

            fig.add_trace(go.Scatter(
                x=buy_ops.index,
                y=data_plot.loc[buy_ops.index, '<CLOSE>'],
                mode='markers',
                name='Compra',
                marker=dict(color='green', symbol='triangle-up', size=10)
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=sell_ops.index,
                y=data_plot.loc[sell_ops.index, '<CLOSE>'],
                mode='markers',
                name='Venta',
                marker=dict(color='red', symbol='triangle-down', size=10)
            ), row=row, col=1)

        # === OPERACIONES ===
        if operaciones and len(operaciones) > 0:
            compras_x = [op['compra_fecha'] for op in operaciones]
            compras_y = [op['compra_precio'] for op in operaciones]
            ventas_x = [op['venta_fecha'] for op in operaciones]
            ventas_y = [op['venta_precio'] for op in operaciones]

            fig.add_trace(go.Scatter(
                x=compras_x,
                y=compras_y,
                mode='markers',
                name='Compra Real',
                marker=dict(color='green', symbol='star', size=12)
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=ventas_x,
                y=ventas_y,
                mode='markers',
                name='Venta Real',
                marker=dict(color='red', symbol='x', size=12)
            ), row=row, col=1)

        row += 1

        # === RSI ===
        if tiene_rsi:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['RSI'],
                name='RSI',
                line=dict(color='purple')
            ), row=row, col=1)

            # Sobrecompra y sobreventa
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
                        line=dict(color="red", dash="dash"), row=row, col=1)
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
                        line=dict(color="green", dash="dash"), row=row, col=1)
            row += 1

        # === MACD ===
        if tiene_macd:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MACD'],
                name='MACD',
                line=dict(color='blue')
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MACD_Signal'],
                name='Señal',
                line=dict(color='orange')
            ), row=row, col=1)

        # === Configuración final ===
        fig.update_layout(
            height=300 * rows,
            title=title,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )

        return fig
