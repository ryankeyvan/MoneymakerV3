# scanner.py
#!/usr/bin/env python3
import os, json, pickle
import yfinance as yf
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# load
models = {
    h: pickle.load(open(os.path.join(MODELS_DIR, f'breakout_model_{h}.pkl'),'rb'))
    for h in ['1w','1m','3m']
}
thresholds = json.load(open(os.path.join(MODELS_DIR, 'thresholds.json'),'r'))

def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return table['Symbol'].str.replace('.', '-', regex=False).tolist()

def extract_features(df):
    return df[['Open','High','Low','Close']].iloc[-1].values

def scan_tickers(tickers=None):
    if tickers is None:
        tickers = get_sp500_tickers()
    results, errors = [], []
    for t in tqdm(tickers, desc="Scanning…"):
        try:
            df = yf.download(t, period='6mo', interval='1d', progress=False)
            if df.empty: raise ValueError("No data")
            df = df[['Open','High','Low','Close']].dropna()
            feats = extract_features(df)
            current = float(df['Close'].iloc[-1])
            rec = {'ticker':t,'current_price':round(current,2)}
            for h, clf in models.items():
                prob = clf.predict_proba([feats])[0][1]
                rec[f'score_{h}'] = round(prob,4)
                rec[f'decision_{h}'] = 'BUY' if prob >= thresholds[h] else 'HOLD'
                rec[f'target_{h}']   = round(current*(1+thresholds[h]),2)
            results.append(rec)
        except Exception as e:
            errors.append({'ticker':t,'error':str(e)})
    return results, errors

def top_breakouts(results, period_key, n=5):
    buys = [r for r in results if r[f'decision_{period_key}']=='BUY']
    buys.sort(key=lambda x: x[f'score_{period_key}'], reverse=True)
    return buys[:n]

if __name__ == "__main__":
    res, errs = scan_tickers()
    print("📈 1-Week Top 5:")
    for r in top_breakouts(res,'1w'): print(r)
    print("📈 1-Month Top 5:")
    for r in top_breakouts(res,'1m'): print(r)
    print("📈 3-Month Top 5:")
    for r in top_breakouts(res,'3m'): print(r)
    if errs: print("⚠️ Errors:", errs)
