from bcb import sgs
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def gerar_serie_usd_brl_mensal_10_anos():
    # Data inicial: 10 anos atrás
    inicio = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    print(f"Buscando câmbio mensal USD/BRL (venda, cód. 3696) desde {inicio}...")

    # Série mensal: Dólar americano (venda) - fim de período - M (código 3696)
    df = sgs.get({'USD_BRL_mensal': 3696}, start=inicio)

    # Garante ordenação
    df = df.sort_index()

    print("\nÚltimos 5 registros mensais USD/BRL (venda):")
    print(df.tail())

    df.to_csv('usd_brl_mensal_10_anos.csv', index_label='Data')
    print("\nArquivo salvo: usd_brl_mensal_10_anos.csv")

    return df

if __name__ == "__main__":
    df_cambio = gerar_serie_usd_brl_mensal_10_anos()

    # Gráfico: quanto 1 USD vale em BRL ao longo dos meses (dólar sempre > real)
    ax = df_cambio['USD_BRL_mensal'].plot(
        figsize=(10, 5),
        title='Taxa de câmbio USD/BRL (venda, fim de período) - últimos 10 anos (mensal)'
    )
    ax.set_ylabel('Reais por 1 Dólar (R$/USD)')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("grafico_usd_brl_mensal_10_anos.png")
    print("Gráfico salvo em grafico_usd_brl_mensal_10_anos.png")
