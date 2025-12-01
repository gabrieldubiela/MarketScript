import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

VALOR_INICIAL = 100


def gerar_serie_btc_10_anos():
    hoje = datetime.today()

    # Data teórica de início: 10 anos atrás, mesmo mês, dia 1
    ano_inicio = hoje.year - 10
    mes_inicio = hoje.month
    inicio_teorico = datetime(ano_inicio, mes_inicio, 1)

    print("Buscando Bitcoin BTC-USD diário (últimos ~11 anos)...")
    dados = yf.download("BTC-USD", period="11y", interval="1d", auto_adjust=False)

    if dados.empty:
        raise RuntimeError("Nenhum dado retornado para BTC-USD.")

    # Trata colunas (simples ou MultiIndex)
    if isinstance(dados.columns, pd.MultiIndex):
        close_diario = dados["Close"]["BTC-USD"]
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
        "BTC_Close": close_mensal,
        "Fator_acumulado": fator_acumulado,
        "Valor_acumulado": valor_acumulado,
    })

    print("\nPrimeiros 3 registros (BTC-USD):")
    print(df.head(3))
    print("\nÚltimos 5 registros (BTC-USD):")
    print(df.tail())

    df.to_csv("btc_10_anos_100_usd.csv", index_label="Data")
    print("\nArquivo salvo: btc_10_anos_100_usd.csv")

    return df


if __name__ == "__main__":
    df_btc = gerar_serie_btc_10_anos()

    valor_final = df_btc["Valor_acumulado"].iloc[-1]
    print(f"\nUS$ {VALOR_INICIAL} em Bitcoin nos últimos 10 anos virariam: US$ {valor_final:.2f}")

    ax = df_btc["Valor_acumulado"].plot(
        figsize=(10, 5),
        title="US$ 100 em Bitcoin (BTC-USD) - últimos 10 anos"
    )
    ax.set_ylabel("Valor (US$)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("grafico_btc_10_anos.png")
    print("Gráfico salvo em grafico_btc_10_anos.png")
