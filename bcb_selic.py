from bcb import sgs
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

VALOR_INICIAL = 100


def gerar_serie_selic_10_anos():
    # Data inicial: 10 anos atrás
    data_inicio = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    print(f"Buscando Selic mensal (cód. 4390) desde {data_inicio}...")

    # Selic mensal (% a.m.)
    selic_mensal = sgs.get({'Selic_mensal_%': 4390}, start=data_inicio)

    # Ordena por data (garantia)
    selic_mensal = selic_mensal.sort_index()

    # Converte para fator mensal
    fatores = 1 + selic_mensal['Selic_mensal_%'] / 100.0

    # Acúmulo composto mês a mês
    fator_acumulado = fatores.cumprod()

    # Valor acumulado de R$ 100
    valor_acumulado = VALOR_INICIAL * fator_acumulado

    # Monta DataFrame final
    df = pd.DataFrame({
        'Selic_mensal_%': selic_mensal['Selic_mensal_%'],
        'Fator_acumulado': fator_acumulado,
        'Valor_acumulado': valor_acumulado
    })

    print("\nÚltimos 5 registros:")
    print(df.tail())

    # Salva para usar em gráficos (Excel, Power BI, etc. ou matplotlib)
    df.to_csv('selic_10_anos_100_reais.csv', index_label='Data')
    print("\nArquivo salvo: selic_10_anos_100_reais.csv")

    return df


if __name__ == "__main__":
    df_selic = gerar_serie_selic_10_anos()

    # Valor final (última linha)
    valor_final = df_selic['Valor_acumulado'].iloc[-1]
    print(f"\nValor final acumulado em Selic: R$ {valor_final:.2f}")

    # Plot do valor acumulado
    ax = df_selic['Valor_acumulado'].plot(
        figsize=(10, 5),
        title='R$ 100 aplicados à Selic - últimos 10 anos'
    )
    ax.set_ylabel('Valor (R$)')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("grafico_selic.png")  # << salva o gráfico
    print("Gráfico salvo em grafico_selic.png")
