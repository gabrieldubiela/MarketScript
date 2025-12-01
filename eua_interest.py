from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

VALOR_INICIAL = 100

def gerar_serie_fedfunds_10_anos():
    fred = Fred(api_key="989e59a7176ae4d3333ff4978009df07")  # coloque sua chave aqui

    fim = datetime.today()
    inicio = fim - timedelta(days=365*10)

    print(f"Buscando Fed Funds Effective Rate (FEDFUNDS) de {inicio.date()} a {fim.date()}...")

    # Série DIÁRIA FEDFUNDS (% a.a.)
    fed_diario = fred.get_series('FEDFUNDS', observation_start=inicio, observation_end=fim)
    if fed_diario is None or fed_diario.empty:
        raise RuntimeError("Nenhum dado retornado para FEDFUNDS.")

    fed_diario = fed_diario.sort_index()
    fed_diario.index = pd.to_datetime(fed_diario.index)

    # 1) Taxa anual MÉDIA de cada mês (% a.a.)
    taxa_aa_mes = fed_diario.resample('M').mean()   # ainda em % a.a.

    # 2) Converte para fração anual
    taxa_aa_frac = taxa_aa_mes / 100.0

    # 3) Converte % a.a. em fator mensal equivalente: (1 + r_aa)^(1/12)
    fator_mensal = (1 + taxa_aa_frac) ** (1/12)

    # 4) Acúmulo composto mês a mês
    fator_acumulado = fator_mensal.cumprod()
    valor_acumulado = VALOR_INICIAL * fator_acumulado

    df = pd.DataFrame({
        "Taxa_aa_%_media_mes": taxa_aa_mes,
        "Fator_mensal": fator_mensal,
        "Fator_acumulado": fator_acumulado,
        "Valor_acumulado": valor_acumulado,
    }).dropna()

    print("\nÚltimos 5 registros (Fed Funds):")
    print(df.tail())

    df.to_csv("fedfunds_10_anos_100_usd.csv", index_label="Data")
    print("\nArquivo salvo: fedfunds_10_anos_100_usd.csv")

    return df


if __name__ == "__main__":
    df_fed = gerar_serie_fedfunds_10_anos()

    # Valor final
    valor_final = df_fed["Valor_acumulado"].iloc[-1]
    fator_total = df_fed["Fator_acumulado"].iloc[-1]
    taxa_media_aa = (fator_total ** (1/10) - 1) * 100

    print(f"\nUS$ {VALOR_INICIAL} aplicados nos Fed Funds por ~10 anos virariam: US$ {valor_final:.2f}")
    print(f"Taxa média anual implícita ~ {taxa_media_aa:.4f}% a.a.")

    ax = df_fed["Valor_acumulado"].plot(
        figsize=(10, 5),
        title="US$ 100 aplicados à taxa Fed Funds efetiva - últimos 10 anos"
    )
    ax.set_ylabel("Valor (US$)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("grafico_fedfunds_10_anos.png")
    print("Gráfico salvo em grafico_fedfunds_10_anos.png")
