"""
Forecast Q1 2027 Incremental Revenue by Channel (Upper Funnel, Lower Funnel, Unpaid)
Using historical data from LL Historical Funnel and Platform.xlsx (2023-2026)

Given Q1 2027 planned spend:
  - Upper Funnel: $1,019,420
  - Lower Funnel: $1,337,392
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
df = pd.read_excel('LL Historical Funnel and Platform.xlsx')
df['Quarter'] = df['Day'].dt.to_period('Q')
df['Year'] = df['Day'].dt.year

# Aggregate quarterly by funnel level
quarterly = df.groupby(['Quarter', 'Funnel Level']).agg({
    'Spend': 'sum',
    'Revenue': 'sum',
    'Orders': 'sum'
}).reset_index()

quarterly['iROAS'] = quarterly['Revenue'] / quarterly['Spend'].replace(0, np.nan)
quarterly['Quarter_str'] = quarterly['Quarter'].astype(str)
quarterly['Year'] = quarterly['Quarter'].dt.year
quarterly['Q'] = quarterly['Quarter'].dt.quarter

print("=" * 80)
print("HISTORICAL QUARTERLY DATA BY FUNNEL LEVEL")
print("=" * 80)

for funnel in ['Upper Funnel', 'Lower Funnel', 'Unpaid']:
    subset = quarterly[quarterly['Funnel Level'] == funnel].sort_values('Quarter')
    print(f"\n--- {funnel} ---")
    for _, row in subset.iterrows():
        roas_str = f"  iROAS: {row['iROAS']:.2f}" if pd.notna(row['iROAS']) else "  iROAS: N/A"
        print(f"  {row['Quarter_str']}  Spend: ${row['Spend']:>12,.2f}  Revenue: ${row['Revenue']:>14,.2f}{roas_str}")

# =============================================================================
# 2. FORECAST UPPER FUNNEL & LOWER FUNNEL (Spend -> Revenue relationship)
# =============================================================================
print("\n" + "=" * 80)
print("FORECASTING METHODOLOGY")
print("=" * 80)

q1_2027_spend = {
    'Upper Funnel': 1_019_420,
    'Lower Funnel': 1_337_392
}

results = {}

for funnel in ['Upper Funnel', 'Lower Funnel']:
    subset = quarterly[quarterly['Funnel Level'] == funnel].sort_values('Quarter')
    # Filter to quarters with meaningful spend (> $1000)
    subset_with_spend = subset[subset['Spend'] > 1000].copy()

    print(f"\n--- {funnel} ---")
    print(f"  Data points with meaningful spend: {len(subset_with_spend)}")

    X = subset_with_spend['Spend'].values.reshape(-1, 1)
    y = subset_with_spend['Revenue'].values

    # Method 1: Linear Regression (Spend vs Revenue)
    lr = LinearRegression()
    lr.fit(X, y)
    lr_pred = lr.predict([[q1_2027_spend[funnel]]])[0]
    lr_r2 = lr.score(X, y)
    print(f"  Method 1 - Linear Regression: R² = {lr_r2:.4f}")
    print(f"    Revenue = {lr.coef_[0]:.4f} * Spend + {lr.intercept_:,.2f}")
    print(f"    Forecast: ${lr_pred:,.2f}")

    # Method 2: Log-Log Regression (captures diminishing returns)
    log_X = np.log(X)
    log_y = np.log(y)
    lr_log = LinearRegression()
    lr_log.fit(log_X, log_y)
    loglog_pred = np.exp(lr_log.predict(np.log([[q1_2027_spend[funnel]]]))[0])
    loglog_r2 = lr_log.score(log_X, log_y)
    print(f"  Method 2 - Log-Log Regression: R² = {loglog_r2:.4f}")
    print(f"    log(Revenue) = {lr_log.coef_[0]:.4f} * log(Spend) + {lr_log.intercept_:.4f}")
    print(f"    Elasticity: {lr_log.coef_[0]:.4f}")
    print(f"    Forecast: ${loglog_pred:,.2f}")

    # Method 3: iROAS trend approach - use recent quarters' iROAS
    recent_q1s = subset_with_spend[subset_with_spend['Q'] == 1]
    recent_qs = subset_with_spend.tail(4)  # Last 4 quarters

    if len(recent_q1s) > 0:
        q1_iroas = recent_q1s['iROAS'].mean()
        print(f"  Method 3a - Historical Q1 iROAS (avg): {q1_iroas:.2f}")
        print(f"    Forecast: ${q1_iroas * q1_2027_spend[funnel]:,.2f}")

    recent_iroas = recent_qs['iROAS'].mean()
    print(f"  Method 3b - Recent 4Q iROAS (avg): {recent_iroas:.2f}")
    print(f"    Forecast: ${recent_iroas * q1_2027_spend[funnel]:,.2f}")

    # Weighted average of methods (favor log-log for diminishing returns + recent iROAS)
    weights = {'linear': 0.25, 'loglog': 0.40, 'recent_iroas': 0.35}
    iroas_pred = recent_iroas * q1_2027_spend[funnel]

    weighted_forecast = (
        weights['linear'] * lr_pred +
        weights['loglog'] * loglog_pred +
        weights['recent_iroas'] * iroas_pred
    )

    implied_iroas = weighted_forecast / q1_2027_spend[funnel]

    results[funnel] = {
        'spend': q1_2027_spend[funnel],
        'linear': lr_pred,
        'loglog': loglog_pred,
        'iroas_based': iroas_pred,
        'weighted': weighted_forecast,
        'implied_iroas': implied_iroas
    }

# =============================================================================
# 3. FORECAST UNPAID (Time Series - no spend driver)
# =============================================================================
print(f"\n--- Unpaid ---")
unpaid = quarterly[quarterly['Funnel Level'] == 'Unpaid'].sort_values('Quarter').copy()
unpaid['Quarter_idx'] = range(len(unpaid))

# Method 1: Q1-specific historical average (seasonality)
unpaid_q1 = unpaid[unpaid['Q'] == 1]
print(f"  Historical Q1 Revenue:")
for _, row in unpaid_q1.iterrows():
    print(f"    {row['Quarter_str']}: ${row['Revenue']:>14,.2f}")

q1_avg = unpaid_q1['Revenue'].mean()
q1_recent_avg = unpaid_q1.tail(2)['Revenue'].mean()  # Last 2 Q1s

# Method 2: Linear trend on Q1s
if len(unpaid_q1) >= 2:
    q1_years = unpaid_q1['Year'].values.reshape(-1, 1)
    q1_rev = unpaid_q1['Revenue'].values
    lr_q1 = LinearRegression()
    lr_q1.fit(q1_years, q1_rev)
    trend_pred = lr_q1.predict([[2027]])[0]
    trend_r2 = lr_q1.score(q1_years, q1_rev)
    print(f"  Q1 Linear Trend: R² = {trend_r2:.4f}, Forecast: ${trend_pred:,.2f}")

# Method 3: Overall trend + seasonality
# Calculate seasonal factors
overall_avg = unpaid['Revenue'].mean()
seasonal_factors = {}
for q in [1, 2, 3, 4]:
    q_data = unpaid[unpaid['Q'] == q]['Revenue']
    seasonal_factors[q] = q_data.mean() / overall_avg if overall_avg > 0 else 1

print(f"  Seasonal factors: Q1={seasonal_factors[1]:.3f}, Q2={seasonal_factors[2]:.3f}, Q3={seasonal_factors[3]:.3f}, Q4={seasonal_factors[4]:.3f}")

# Trend on deseasonalized data
unpaid_deseas = unpaid.copy()
unpaid_deseas['deseas_rev'] = unpaid_deseas.apply(lambda r: r['Revenue'] / seasonal_factors[r['Q']], axis=1)
lr_trend = LinearRegression()
lr_trend.fit(unpaid_deseas['Quarter_idx'].values.reshape(-1, 1), unpaid_deseas['deseas_rev'].values)
# Q1 2027 would be index len(unpaid) + quarters_ahead
# Last quarter in data is 2026Q1 (index len-1). Q1 2027 = 4 quarters later
next_idx = len(unpaid) + 3  # 2026Q2, Q3, Q4, then 2027Q1
trend_deseas_pred = lr_trend.predict([[next_idx]])[0] * seasonal_factors[1]
print(f"  Trend + Seasonality Forecast: ${trend_deseas_pred:,.2f}")

# Weighted forecast for unpaid
unpaid_weighted = 0.30 * q1_recent_avg + 0.30 * trend_pred + 0.40 * trend_deseas_pred

results['Unpaid'] = {
    'spend': 0,
    'q1_avg': q1_avg,
    'q1_recent_avg': q1_recent_avg,
    'trend': trend_pred,
    'trend_seasonal': trend_deseas_pred,
    'weighted': unpaid_weighted,
}

# =============================================================================
# 4. FINAL RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("Q1 2027 INCREMENTAL REVENUE FORECAST")
print("=" * 80)

total_spend = 0
total_revenue = 0

print(f"\n{'Channel':<16} {'Planned Spend':>15} {'Forecast Revenue':>18} {'Implied iROAS':>15}")
print("-" * 68)

for funnel in ['Upper Funnel', 'Lower Funnel', 'Unpaid']:
    r = results[funnel]
    spend = r['spend']
    rev = r['weighted']
    iroas = rev / spend if spend > 0 else 'N/A'

    total_spend += spend
    total_revenue += rev

    iroas_str = f"{iroas:.2f}x" if isinstance(iroas, float) else iroas
    spend_str = f"${spend:>13,.0f}" if spend > 0 else f"{'$0':>14}"
    print(f"{funnel:<16} {spend_str} {f'${rev:>16,.2f}'} {iroas_str:>14}")

print("-" * 68)
total_iroas = total_revenue / total_spend if total_spend > 0 else 'N/A'
total_iroas_str = f"{total_iroas:.2f}x" if isinstance(total_iroas, float) else total_iroas
print(f"{'TOTAL':<16} ${total_spend:>13,.0f} ${total_revenue:>16,.2f} {total_iroas_str:>14}")

print(f"\n{'Total Media Spend':>30}: ${total_spend:>14,.0f}")
print(f"{'Total Forecast Revenue':>30}: ${total_revenue:>14,.2f}")
print(f"{'Overall Blended iROAS':>30}: {total_iroas:.2f}x")

# =============================================================================
# 5. DETAIL BREAKDOWN BY METHOD
# =============================================================================
print("\n" + "=" * 80)
print("FORECAST DETAIL BY METHOD")
print("=" * 80)

for funnel in ['Upper Funnel', 'Lower Funnel']:
    r = results[funnel]
    print(f"\n--- {funnel} (Spend: ${r['spend']:,.0f}) ---")
    print(f"  Linear Regression:     ${r['linear']:>14,.2f}  (iROAS: {r['linear']/r['spend']:.2f}x)")
    print(f"  Log-Log Regression:    ${r['loglog']:>14,.2f}  (iROAS: {r['loglog']/r['spend']:.2f}x)")
    print(f"  Recent iROAS-based:    ${r['iroas_based']:>14,.2f}  (iROAS: {r['iroas_based']/r['spend']:.2f}x)")
    print(f"  Weighted Average:      ${r['weighted']:>14,.2f}  (iROAS: {r['implied_iroas']:.2f}x)")
    print(f"  Weights: Linear=25%, Log-Log=40%, Recent iROAS=35%")

r = results['Unpaid']
print(f"\n--- Unpaid (No spend) ---")
print(f"  Q1 Historical Average: ${r['q1_avg']:>14,.2f}")
print(f"  Recent Q1 Average:     ${r['q1_recent_avg']:>14,.2f}")
print(f"  Q1 Linear Trend:       ${r['trend']:>14,.2f}")
print(f"  Trend + Seasonality:   ${r['trend_seasonal']:>14,.2f}")
print(f"  Weighted Average:      ${r['weighted']:>14,.2f}")
print(f"  Weights: Recent Q1 Avg=30%, Trend=30%, Trend+Seasonal=40%")

# =============================================================================
# 6. CONFIDENCE INTERVALS (using historical variance)
# =============================================================================
print("\n" + "=" * 80)
print("CONFIDENCE RANGES (based on historical variability)")
print("=" * 80)

for funnel in ['Upper Funnel', 'Lower Funnel']:
    subset = quarterly[(quarterly['Funnel Level'] == funnel) & (quarterly['Spend'] > 1000)]
    iroas_std = subset['iROAS'].std()
    iroas_mean = subset['iROAS'].mean()
    cv = iroas_std / iroas_mean if iroas_mean > 0 else 0.2

    r = results[funnel]
    low = r['weighted'] * (1 - cv)
    high = r['weighted'] * (1 + cv)
    print(f"  {funnel}: ${low:,.0f} - ${high:,.0f} (±{cv*100:.0f}%)")

# Unpaid confidence range
unpaid_q1_rev = unpaid_q1['Revenue'].values
unpaid_cv = unpaid_q1_rev.std() / unpaid_q1_rev.mean() if unpaid_q1_rev.mean() > 0 else 0.15
r = results['Unpaid']
low = r['weighted'] * (1 - unpaid_cv)
high = r['weighted'] * (1 + unpaid_cv)
print(f"  Unpaid: ${low:,.0f} - ${high:,.0f} (±{unpaid_cv*100:.0f}%)")
