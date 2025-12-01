from bcb import sgs
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

VALOR_INICIAL = 100


def gerar_serie_ipca_10_anos():
    # Data inicial: 10 anos atrás
    data_inicio = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    print(f"Buscando IPCA mensal (cód. 433) desde {data_inicio}...")

    # IPCA mensal (% a.m.)
    ipca_mensal = sgs.get({'IPCA_mensal_%': 433}, start=data_inicio)

    # Ordena por data (garantia)
    ipca_mensal = ipca_mensal.sort_index()

    # Converte para fator mensal
    fatores = 1 + ipca_mensal['IPCA_mensal_%'] / 100.0

    # Acúmulo composto mês a mês
    fator_acumulado = fatores.cumprod()

    # Valor acumulado de R$ 100 em termos NOMINAIS corrigidos pelo IPCA
    valor_acumulado = VALOR_INICIAL * fator_acumulado

    # Monta DataFrame final
    df = pd.DataFrame({
        'IPCA_mensal_%': ipca_mensal['IPCA_mensal_%'],
        'Fator_acumulado': fator_acumulado,
        'Valor_acumulado': valor_acumulado
    })

    print("\nÚltimos 5 registros (IPCA):")
    print(df.tail())

    df.to_csv('ipca_10_anos_100_reais.csv', index_label='Data')
    print("\nArquivo salvo: ipca_10_anos_100_reais.csv")

    return df


if __name__ == "__main__":
    df_ipca = gerar_serie_ipca_10_anos()

    # Valor final
    valor_final = df_ipca['Valor_acumulado'].iloc[-1]
    print(f"\nValor final corrigido pelo IPCA: R$ {valor_final:.2f}")

    # Plot do valor acumulado
    ax = df_ipca['Valor_acumulado'].plot(
        figsize=(10, 5),
        title='R$ 100 corrigidos pelo IPCA - últimos 10 anos'
    )
    ax.set_ylabel('Valor (R$ em termos nominais)')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("grafico_ipca.png")
    print("Gráfico salvo em grafico_ipca.png")
