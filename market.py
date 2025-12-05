from bcb import sgs
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from fredapi import Fred
import os
import numpy as np

INITIAL_VALUE = 100

def get_date():
    today = datetime.today()
    years_past = 10 # choose how many years in the past
    start_year = today.year - years_past
    start_month = today.month
    start_period = datetime(start_year, start_month, 1)
    return start_period

def dolar_real():
    data = sgs.get({'USD_BRL_mensal': 3696}, start=get_date())
    data = data.sort_index()
    return data

def brazilian_inflation():
    monthly_inflation = sgs.get({'monthly_inflation_%': 433}, start=get_date())
    monthly_inflation = monthly_inflation.sort_index()
    factor = 1 + monthly_inflation['monthly_inflation_%'] / 100.0
    cumulate_factor = factor.cumprod()
    cumulate_value = INITIAL_VALUE * cumulate_factor
    data = pd.DataFrame({
        'monthly_inflation_%': monthly_inflation['monthly_inflation_%'],
        'cumulate_factor': cumulate_factor,
        'cumulate_value': cumulate_value
    })
    return data

def brazilian_interest():
    monthly_interest = sgs.get({'monthly_interest_%': 4390}, start=get_date())
    monthly_interest = monthly_interest.sort_index()
    factor = 1 + monthly_interest['monthly_interest_%'] / 100.0
    cumulate_factor = factor.cumprod()
    cumulate_value = INITIAL_VALUE * cumulate_factor
    data = pd.DataFrame({
        'monthly_interest_%': monthly_interest['monthly_interest_%'],
        'cumulate_factor': cumulate_factor,
        'cumulate_value': cumulate_value
    })
    return data

def brazilian_market():
    market_data = yf.download("^BVSP", period="11y", interval="1d", auto_adjust=False)
    if isinstance(market_data.columns, pd.MultiIndex):
        daily_close = market_data["Close"]["^BVSP"]
    else:
        daily_close = market_data["Close"]
    daily_close = daily_close.dropna().sort_index()
    monthly_close = daily_close.resample("M").last()
    monthly_close = monthly_close[monthly_close.index >= pd.Timestamp(get_date())]
    cumulate_factor = monthly_close / monthly_close.iloc[0]
    cumulate_value = INITIAL_VALUE * cumulate_factor
    data = pd.DataFrame({
        "IBOV_Close": monthly_close,
        "cumulate_factor": cumulate_factor,
        "cumulate_value": cumulate_value,
    })
    return data

def bitcoin():
    market_data = yf.download("BTC-USD", period="11y", interval="1d", auto_adjust=False)
    if isinstance(market_data.columns, pd.MultiIndex):
        daily_close = market_data["Close"]["BTC-USD"]
    else:
        daily_close = market_data["Close"]
    daily_close = daily_close.dropna().sort_index()
    monthly_close = daily_close.resample("M").last()
    monthly_close = monthly_close[monthly_close.index >= pd.Timestamp(get_date())]
    cumulate_factor = monthly_close / monthly_close.iloc[0]
    cumulate_value = INITIAL_VALUE * cumulate_factor
    data = pd.DataFrame({
        "BTC_Close": monthly_close,
        "cumulate_factor": cumulate_factor,
        "cumulate_value": cumulate_value,
    })
    return data

def eua_insterest():

    fred_key = os.getenv('FRED_API_KEY', '')
    fred = Fred(api_key=fred_key)
    fed_diario = fred.get_series('FEDFUNDS', observation_start=get_date(), observation_end=datetime.today())
    fed_diario = fed_diario.sort_index()
    fed_diario.index = pd.to_datetime(fed_diario.index)
    interest_aa_month = fed_diario.resample('M').mean()  
    interest_aa_frac = interest_aa_month / 100.0
    monthly_factor = (1 + interest_aa_frac) ** (1/12)
    cumulate_factor = monthly_factor.cumprod()
    cumulate_value = INITIAL_VALUE * cumulate_factor
    data = pd.DataFrame({
        "Taxa_aa_%_media_mes": interest_aa_month,
        "monthly_factor": monthly_factor,
        "cumulate_factor": cumulate_factor,
        "cumulate_value": cumulate_value,
    }).dropna()
    return data


def eua_market():
    market_data = yf.download("^SPX", period="11y", interval="1d", auto_adjust=False)
    if isinstance(market_data.columns, pd.MultiIndex):
        daily_close = market_data["Close"]["^SPX"]
    else:
        daily_close = market_data["Close"]
    daily_close = daily_close.dropna().sort_index()
    monthly_close = daily_close.resample("M").last()
    monthly_close = monthly_close[monthly_close.index >= pd.Timestamp(get_date())]
    cumulate_factor = monthly_close / monthly_close.iloc[0]
    cumulate_value = INITIAL_VALUE * cumulate_factor
    data = pd.DataFrame({
        "SPX_Close": monthly_close,
        "cumulate_factor": cumulate_factor,
        "cumulate_value": cumulate_value,
    })
    return data

def get_all_data():
    df_consolidated = pd.DataFrame()
    
    try:
        df_btc = bitcoin()
        if not df_btc.empty and 'cumulate_value' in df_btc.columns:
            df_consolidated['BTC'] = df_btc['cumulate_value']
    except Exception:
        pass
    
    try:
        df_ibov = brazilian_market()
        if not df_ibov.empty and 'cumulate_value' in df_ibov.columns:
            df_consolidated['IBOV'] = df_ibov['cumulate_value']
    except Exception:
        pass
    
    try:
        df_spx = eua_market()
        if not df_spx.empty and 'cumulate_value' in df_spx.columns:
            df_consolidated['SPX'] = df_spx['cumulate_value']
    except Exception:
        pass
    
    try:
        df_fed = eua_insterest()
        if not df_fed.empty and 'cumulate_value' in df_fed.columns:
            df_consolidated['FED'] = df_fed['cumulate_value']
    except Exception:
        pass
    
    try:
        df_selic = brazilian_interest()
        if not df_selic.empty and 'cumulate_value' in df_selic.columns:
            df_consolidated['SELIC'] = df_selic['cumulate_value']
    except Exception:
        pass
    
    try:
        df_ipca = brazilian_inflation()
        if not df_ipca.empty and 'cumulate_value' in df_ipca.columns:
            df_consolidated['IPCA'] = df_ipca['cumulate_value']
    except Exception:
        pass
    
    try:
        df_usd = dolar_real()
        if not df_usd.empty:
            df_consolidated['USD_BRL'] = df_usd.iloc[:, 0]
    except Exception:
        pass
    
    # Ordenar por índice (data)
    df_consolidated = df_consolidated.sort_index()
    
    return df_consolidated


def plot_comparativo(df):
    ax = df.plot(
        figsize=(14, 7),
        title="Comparativo: R$ 100 investidos em diferentes ativos - últimos 10 anos"
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor (R$)")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("grafico_comparativo_10_anos.png")
    print("Gráfico salvo em grafico_comparativo_10_anos.png")


if __name__ == "__main__":
    df = get_all_data()
    
    df.to_csv("comparativo_10_anos.csv", index_label="Data")
    print("Dados salvos em comparativo_10_anos.csv\n")

    plot_comparativo(df)
