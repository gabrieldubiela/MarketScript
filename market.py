from bcb import sgs
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from fredapi import Fred
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

load_dotenv()

INITIAL_VALUE = 100

def get_date():
    today = datetime.today()
    years_past = 10
    start_year = today.year - years_past
    start_month = today.month
    if start_month == 12:
        start_period = datetime(start_year + 1, 1, 1) - timedelta(days=1)
    else:
        start_period = datetime(start_year, start_month + 1, 1) - timedelta(days=1)
    return start_period

def dollar_to_real():
    data = sgs.get({'USD_BRL_monthly': 3696}, start=get_date())
    data = data.sort_index()
    data = data.resample('ME').last()
    return data

def brazilian_inflation():
    monthly_inflation = sgs.get({'monthly_inflation_%': 433}, start=get_date())
    monthly_inflation = monthly_inflation.sort_index()
    monthly_inflation = monthly_inflation.resample('ME').last()
    factor = 1 + monthly_inflation['monthly_inflation_%'] / 100.0
    cumulative_factor = factor.cumprod()
    cumulative_value = INITIAL_VALUE * cumulative_factor
    data = pd.DataFrame({
        'monthly_inflation_%': monthly_inflation['monthly_inflation_%'],
        'cumulative_factor': cumulative_factor,
        'cumulative_value': cumulative_value
    })
    return data

def brazilian_interest():
    monthly_interest = sgs.get({'monthly_interest_%': 4390}, start=get_date())
    monthly_interest = monthly_interest.sort_index()
    monthly_interest = monthly_interest.resample('ME').last()
    factor = 1 + monthly_interest['monthly_interest_%'] / 100.0
    cumulative_factor = factor.cumprod()
    cumulative_value = INITIAL_VALUE * cumulative_factor
    data = pd.DataFrame({
        'monthly_interest_%': monthly_interest['monthly_interest_%'],
        'cumulative_factor': cumulative_factor,
        'cumulative_value': cumulative_value
    })
    return data

def brazilian_market():
    market_data = yf.download("^BVSP", period="11y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(market_data.columns, pd.MultiIndex):
        daily_close = market_data["Close"]["^BVSP"]
    else:
        daily_close = market_data["Close"]
    daily_close = daily_close.dropna().sort_index()
    monthly_close = daily_close.resample("ME").last()
    monthly_close = monthly_close[monthly_close.index >= pd.Timestamp(get_date())]
    cumulative_factor = monthly_close / monthly_close.iloc[0]
    cumulative_value = INITIAL_VALUE * cumulative_factor
    data = pd.DataFrame({
        "IBOV_Close": monthly_close,
        "cumulative_factor": cumulative_factor,
        "cumulative_value": cumulative_value,
    })
    return data

def bitcoin():
    market_data = yf.download("BTC-USD", period="11y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(market_data.columns, pd.MultiIndex):
        daily_close = market_data["Close"]["BTC-USD"]
    else:
        daily_close = market_data["Close"]
    daily_close = daily_close.dropna().sort_index()
    monthly_close = daily_close.resample("ME").last()
    monthly_close = monthly_close[monthly_close.index >= pd.Timestamp(get_date())]
    cumulative_factor = monthly_close / monthly_close.iloc[0]
    cumulative_value = INITIAL_VALUE * cumulative_factor
    data = pd.DataFrame({
        "BTC_Close": monthly_close,
        "cumulative_factor": cumulative_factor,
        "cumulative_value": cumulative_value,
    })
    return data

def us_interest():
    fred_key = os.getenv('FRED_API_KEY', '')
    fred = Fred(api_key=fred_key)
    fed_daily = fred.get_series('FEDFUNDS', observation_start=get_date(), observation_end=datetime.today())
    fed_daily = fed_daily.sort_index()
    fed_daily.index = pd.to_datetime(fed_daily.index)
    interest_yearly_avg_month = fed_daily.resample("ME").mean()  
    interest_yearly_frac = interest_yearly_avg_month / 100.0
    monthly_factor = (1 + interest_yearly_frac) ** (1/12)
    cumulative_factor = monthly_factor.cumprod()
    cumulative_value = INITIAL_VALUE * cumulative_factor
    data = pd.DataFrame({
        "Yearly_Rate_%_monthly_avg": interest_yearly_avg_month,
        "monthly_factor": monthly_factor,
        "cumulative_factor": cumulative_factor,
        "cumulative_value": cumulative_value,
    }).dropna()
    return data

def us_market():
    market_data = yf.download("^SPX", period="11y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(market_data.columns, pd.MultiIndex):
        daily_close = market_data["Close"]["^SPX"]
    else:
        daily_close = market_data["Close"]
    daily_close = daily_close.dropna().sort_index()
    monthly_close = daily_close.resample("ME").last()
    monthly_close = monthly_close[monthly_close.index >= pd.Timestamp(get_date())]
    cumulative_factor = monthly_close / monthly_close.iloc[0]
    cumulative_value = INITIAL_VALUE * cumulative_factor
    data = pd.DataFrame({
        "SPX_Close": monthly_close,
        "cumulative_factor": cumulative_factor,
        "cumulative_value": cumulative_value,
    })
    return data

def get_all_data():
    sources = [
        ('BTC', bitcoin),
        ('IBOV', brazilian_market),
        ('SPX', us_market),
        ('FED', us_interest),
        ('SELIC', brazilian_interest),
        ('IPCA', brazilian_inflation),
    ]
    
    df_consolidated = pd.DataFrame()
    
    for name, func in sources:
        try:
            df = func()
            if not df.empty and 'cumulative_value' in df.columns:
                df_consolidated[name] = df['cumulative_value']
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    try:
        df_usd = dollar_to_real()
        if not df_usd.empty:
            df_consolidated['USD_BRL'] = df_usd.iloc[:, 0]
    except Exception as e:
        print(f"Error loading USD_BRL: {e}")
    
    df_consolidated = df_consolidated.bfill().ffill()
    df_consolidated = df_consolidated.dropna(subset=['IBOV', 'SPX'])
    
    return df_consolidated.sort_index()

def plot_comparison(df):
    df_plot = df.copy()
    initial_exchange_rate = df_plot['USD_BRL'].iloc[0]
    df_plot['SPX'] = (df_plot['SPX'] / initial_exchange_rate) * df_plot['USD_BRL']
    df_plot['FED'] = (df_plot['FED'] / initial_exchange_rate) * df_plot['USD_BRL']
    df_plot = df_plot.drop(columns=['BTC', 'USD_BRL'], errors='ignore')
    
    ax = df_plot.plot(
        figsize=(14, 7),
        title="Historical Comparison: R$ 100 invested in different assets (SPX and FED adjusted by exchange rate)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (R$)")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("comparison_chart.png")

def analyze_and_predict(df):
    X = np.arange(len(df)).reshape(-1, 1)
    future_months = 120
    X_future = np.arange(len(df), len(df) + future_months).reshape(-1, 1)
    
    columns_to_predict = ['BTC', 'IBOV', 'SPX', 'FED', 'SELIC', 'IPCA', 'USD_BRL']
    predictions = {}
    
    print("=" * 60)
    print("\nANALYSIS AND FORECAST FOR THE NEXT 10 YEARS\n")
    print("=" * 60)
    
    for col in columns_to_predict:
        if col in df.columns:
            try:
                y = df[col].values
                model = LinearRegression()
                model.fit(X, y)
                y_pred_future = model.predict(X_future)
                
                initial_value = y[0]
                current_value = y[-1]
                predicted_value = y_pred_future[-1]
                
                historical_growth = ((current_value - initial_value) / initial_value) * 100
                future_growth = ((predicted_value - current_value) / current_value) * 100
                total_growth = ((predicted_value - initial_value) / initial_value) * 100
                
                future_dates = pd.date_range(start=df.index[-1], periods=future_months + 1, freq='ME')[1:]
                predictions[col] = pd.Series(y_pred_future, index=future_dates)
                
                print(f"\n{col}:")
                print(f"  Initial value (2015-12-31): R$ {initial_value:.2f}")
                print(f"  Current value ({df.index[-1].date()}): R$ {current_value:.2f}")
                print(f"  Predicted value (2035-12): R$ {predicted_value:.2f}")
                print(f"  Historical growth (10 years): {historical_growth:.2f}%")
                print(f"  Forecasted growth (next 10 years): {future_growth:.2f}%")
                print(f"  Total growth (20 years): {total_growth:.2f}%")
                print(f"  Average annual historical rate: {(historical_growth / 10):.2f}%")
                print(f"  Average annual forecasted rate: {(future_growth / 10):.2f}%")
                
            except Exception as e:
                print(f"\n{col}: Error making forecast - {e}")
    
    df_combined = pd.concat([df] + [pd.DataFrame({col: predictions[col]}) for col in predictions.keys() if col in df.columns], axis=1)
    df_combined = df_combined.sort_index()
    
    return df_combined, predictions

def plot_predictions(df, predictions):
    X = np.arange(len(df)).reshape(-1, 1)
    future_months = 120
    X_future = np.arange(len(df), len(df) + future_months).reshape(-1, 1)
    
    from sklearn.linear_model import LinearRegression
    
    if 'USD_BRL' in df.columns:
        model_usd = LinearRegression()
        y_usd = df['USD_BRL'].values
        model_usd.fit(X, y_usd)
        y_usd_future = model_usd.predict(X_future)
    else:
        y_usd_future = None

    future_dates = pd.date_range(start=df.index[-1], periods=future_months + 1, freq='ME')[1:]
    df_future = pd.DataFrame(index=future_dates)
    columns_to_plot = ['IBOV', 'SPX', 'FED', 'SELIC', 'IPCA', 'USD_BRL']

    for col in columns_to_plot:
        if col in predictions:
            df_future[col] = predictions[col].values

    if 'SPX' in df_future.columns and y_usd_future is not None:
        initial_exchange_rate = df['USD_BRL'].iloc[0]
        df_future['SPX'] = (df_future['SPX'] / initial_exchange_rate) * y_usd_future
        df_future['FED'] = (df_future['FED'] / initial_exchange_rate) * y_usd_future

    fig, ax = plt.subplots(figsize=(16, 8))
    
    for col in columns_to_plot:
        if col in df_future.columns:
            ax.plot(df_future.index, df_future[col], label=col, linewidth=2)

    ax.set_title('10-Year Forecast (Exchange Rate Considered)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value (R$)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("prediction.png", dpi=100)

def answer_questions(df, predictions):
    from math import inf

    horizons = {'2y': 24, '5y': 60, '10y': 120}
    results = {'best_growth': {}, 'volatility': {}}

    initial_exchange = df['USD_BRL'].iloc[0] if 'USD_BRL' in df.columns else None
    usd_pred = predictions.get('USD_BRL')

    for name, months in horizons.items():
        best_asset = None
        best_pct = -inf
        for col, series in predictions.items():
            if len(series) < months:
                continue
            pred_val = float(series.iloc[months - 1])
            if col not in df.columns:
                continue
            current_val = float(df[col].iloc[-1])

            if col in ('SPX', 'FED') and usd_pred is not None and initial_exchange is not None:
                usd_at_horizon = float(usd_pred.iloc[months - 1])
                current_br = (current_val / initial_exchange) * float(df['USD_BRL'].iloc[-1])
                pred_br = (pred_val / initial_exchange) * usd_at_horizon
            else:
                current_br = current_val
                pred_br = pred_val

            if current_br == 0:
                pct = None
            else:
                pct = (pred_br - current_br) / current_br * 100

            if pct is not None and pct > best_pct:
                best_pct = pct
                best_asset = col

        results['best_growth'][name] = {'asset': best_asset, 'pct_growth': None if best_pct == -inf else round(best_pct, 2)}

    safety_metrics = {}
    for col in df.columns:
        try:
            series = df[col]
            returns = series.pct_change().dropna()
            
            cumulative_return = (1 + returns).prod() - 1
            
            negative_periods = (returns < 0).sum()
            total_periods = len(returns)
            negative_ratio = negative_periods / total_periods if total_periods > 0 else 0
            
            min_value = series.min()
            max_value = series.max()
            initial_value = series.iloc[0]
            
            if max_value > 0:
                max_drawdown = (min_value - max_value) / max_value
            else:
                max_drawdown = None
                
            safety_metrics[col] = {
                'cumulative_return': cumulative_return,
                'negative_periods': negative_periods,
                'negative_ratio': negative_ratio,
                'max_drawdown': max_drawdown,
                'final_value': series.iloc[-1],
                'initial_value': initial_value
            }
        except Exception:
            safety_metrics[col] = {
                'cumulative_return': None,
                'negative_periods': None,
                'negative_ratio': None,
                'max_drawdown': None,
                'final_value': None,
                'initial_value': None
            }
    
    safest_asset = None
    best_safety_score = -inf
    best_return = -inf
    
    for col, metrics in safety_metrics.items():
        if metrics['negative_ratio'] is None:
            continue
        
        if metrics['cumulative_return'] >= 0:
            safety_score = 100 - (metrics['negative_ratio'] * 100)
        else:
            safety_score = -100
        
        cumulative_ret = metrics['cumulative_return'] if metrics['cumulative_return'] is not None else -inf
        
        if safety_score > best_safety_score or (safety_score == best_safety_score and cumulative_ret > best_return):
            best_safety_score = safety_score
            best_return = cumulative_ret
            safest_asset = col
    
    results['safety'] = {
        'safest_asset': safest_asset,
        'metrics': safety_metrics,
        'best_safety_score': best_safety_score if best_safety_score != -inf else None
    }
    
    print("\n")
    print("=" * 60)
    print("\nQUESTIONS\n")
    print("=" * 60)

    print('\nQUESTION 1 - Highest forecasted percentage growth:')
    print('-' * 60)
    for h, info in results['best_growth'].items():
        print(f"  {h}: {info['asset']} -> {info['pct_growth']}%")
    
    print('\nJUSTIFICATION FOR QUESTION 1:')
    print('-' * 60)
    print('Method: Linear regression forecasting applied to each asset.')
    print('For each time horizon (2, 5, 10 years), we:')
    print('  1. Extract the predicted value at the specified month.')
    print('  2. Calculate percentage growth from current to predicted value.')
    print('  3. Adjust SPX and FED predictions by forecasted USD/BRL exchange rate.')
    print('  4. Identify the asset with maximum growth percentage.')
    
    print('\n\nQUESTION 2 - Which asset is safest (most consistent positive returns):')
    print('-' * 60)
    if safest_asset:
        safe_metrics = safety_metrics[safest_asset]
        print(f"  Asset: {safest_asset}")
        print(f"  Cumulative return (10 years): {safe_metrics['cumulative_return']*100:.2f}%")
        print(f"  Negative periods: {safe_metrics['negative_periods']} out of {total_periods} months")
        print(f"  Months with losses: {safe_metrics['negative_ratio']*100:.1f}%")
    
    print('\nSAFETY ANALYSIS - Comparison of all assets:')
    print('-' * 60)
    print('Asset      | Cumulative Return | Months w/ Loss | Safety Score')
    print('-' * 60)
    for col in sorted(safety_metrics.keys()):
        m = safety_metrics[col]
        if m['cumulative_return'] is not None:
            ret_pct = m['cumulative_return'] * 100
            loss_pct = m['negative_ratio'] * 100
            score = 100 - loss_pct if m['cumulative_return'] >= 0 else -100
            print(f"{col:10s} | {ret_pct:16.2f}% | {loss_pct:13.1f}% | {score:6.1f}")
    
    print('\nJUSTIFICATION FOR QUESTION 2:')
    print('-' * 60)
    print('Method: Safety Score = Assets with POSITIVE returns and MINIMAL loss periods.')
    print('For each asset, we analyze:')
    print('  1. Cumulative return over 10 years (must be positive for safety).')
    print('  2. Percentage of months with negative returns (drawdowns).')
    print('  3. Safety score = 100 - (% of negative months), prioritizing consistent gains.')
    print('\nReasoning:')
    if safest_asset:
        safe_metrics = safety_metrics[safest_asset]
        print(f'  - {safest_asset} is the safest because:')
        print(f'    1. Final return: {safe_metrics["cumulative_return"]*100:.2f}% (POSITIVE and RELIABLE)')
        print(f'    2. Lost money in only {safe_metrics["negative_ratio"]*100:.1f}% of months')
        print(f'    3. Most months ({100-safe_metrics["negative_ratio"]*100:.1f}%) generated gains\n')
    
    print("=" * 60)
    print("\n")

    return results

if __name__ == "__main__":
    df = get_all_data()
    df_pred, predictions = analyze_and_predict(df)
    answer_questions(df, predictions)
    plot_comparison(df)
    plot_predictions(df, predictions)
