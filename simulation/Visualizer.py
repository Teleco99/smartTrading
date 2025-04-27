import plotly.graph_objects as go

from plotly.subplots import make_subplots

class Visualizer:

    @staticmethod
    def plot_interactive_price(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['<CLOSE>'], mode='lines', name='Precio'))
        fig.update_layout(title="Visualización de precios", xaxis_title="Fecha", yaxis_title="Precio")

        return fig
    
    @staticmethod
    def plot_interactive_combined(data, flags, signals=None, title="Análisis interactivo"):
        rows = 1 + int(flags.get("rsi", False)) + int(flags.get("macd", False))

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5] + [0.25] * (rows - 1),
            subplot_titles=["Precio"] +
                (["RSI"] if flags.get("rsi") else []) +
                (["MACD"] if flags.get("macd") else [])
        )

        row = 1

        # === Precio ===
        fig.add_trace(go.Scatter(
            x=data.index, y=data['<CLOSE>'],
            name='Precio',
            line=dict(color='blue')
        ), row=row, col=1)

        if 'Prediction' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Prediction'],
                name='Prediction',
                line=dict(color='orange', dash='dash')
            ), row=row, col=1)

        # === Señales ===
        if signals is not None and 'Signal' in signals.columns:
            buy_ops = signals[signals['Operacion'] == 1]
            sell_ops = signals[signals['Operacion'] == -1]

            fig.add_trace(go.Scatter(
                x=buy_ops.index,
                y=data.loc[buy_ops.index, '<CLOSE>'],
                mode='markers',
                name='Compra',
                marker=dict(color='green', symbol='triangle-up', size=10)
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=sell_ops.index,
                y=data.loc[sell_ops.index, '<CLOSE>'],
                mode='markers',
                name='Venta',
                marker=dict(color='red', symbol='triangle-down', size=10)
            ), row=row, col=1)

        row += 1

        # === RSI ===
        if flags.get("rsi"):
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
        if flags.get("macd"):
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
