import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

VALOR_INICIAL = 100


def gerar_serie_ibov_10_anos():
    hoje = datetime.today()

    # Data teórica de início: 10 anos atrás, mesmo mês, dia 1
    ano_inicio = hoje.year - 10
    mes_inicio = hoje.month
    inicio_teorico = datetime(ano_inicio, mes_inicio, 1)

    print("Buscando IBOVESPA (^BVSP) diário (últimos ~11 anos)...")
    dados = yf.download("^BVSP", period="11y", interval="1d", auto_adjust=False)

    if dados.empty:
        raise RuntimeError("Nenhum dado retornado para ^BVSP.")

    # Trata colunas (simples ou MultiIndex)
    if isinstance(dados.columns, pd.MultiIndex):
        close_diario = dados["Close"]["^BVSP"]
    else:
        close_diario = dados["Close"]

    close_diario = close_diario.dropna().sort_index()

    # Reamostra para fim de mês
    close_mensal = close_diario.resample("M").last()

    # Mantém só últimos 10 anos completos (a partir da data teórica)
    close_mensal = close_mensal[close_mensal.index >= pd.Timestamp(inicio_teorico)]

    # Normaliza para começar em 100
    fator_acumulado = close_mensal / close_mensal.iloc[0]
    valor_acumulado = VALOR_INICIAL * fator_acumulado

    df = pd.DataFrame({
        "IBOV_Close": close_mensal,
        "Fator_acumulado": fator_acumulado,
        "Valor_acumulado": valor_acumulado,
    })

    print("\nPrimeiros 3 registros (IBOV):")
    print(df.head(3))
    print("\nÚltimos 5 registros (IBOV):")
    print(df.tail())

    df.to_csv("ibov_10_anos_100_reais.csv", index_label="Data")
    print("\nArquivo salvo: ibov_10_anos_100_reais.csv")

    return df


if __name__ == "__main__":
    df_ibov = gerar_serie_ibov_10_anos()

    valor_final = df_ibov["Valor_acumulado"].iloc[-1]
    print(f"\nR$ {VALOR_INICIAL} aplicados no Ibovespa nos últimos 10 anos virariam: R$ {valor_final:.2f}")

    ax = df_ibov["Valor_acumulado"].plot(
        figsize=(10, 5),
        title="R$ 100 aplicados no Ibovespa - últimos 10 anos"
    )
    ax.set_ylabel("Valor (R$)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("grafico_ibov_10_anos.png")
    print("Gráfico salvo em grafico_ibov_10_anos.png")
