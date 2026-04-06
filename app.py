"""
Streamlit App: Incremental Revenue Forecaster
Forecast quarterly revenue by channel based on planned spend inputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="LiftLab Revenue Forecaster", layout="wide")

# =============================================================================
# DATA LOADING & MODEL TRAINING (cached)
# =============================================================================
@st.cache_data
def load_and_train():
    filepath = os.path.join(os.path.dirname(__file__), 'LL Historical Funnel and Platform.xlsx')
    df = pd.read_excel(filepath)
    df['Quarter'] = df['Day'].dt.to_period('Q')
    df['Year'] = df['Day'].dt.year

    quarterly = df.groupby(['Quarter', 'Funnel Level']).agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Orders': 'sum'
    }).reset_index()

    quarterly['iROAS'] = quarterly['Revenue'] / quarterly['Spend'].replace(0, np.nan)
    quarterly['Quarter_str'] = quarterly['Quarter'].astype(str)
    quarterly['Year'] = quarterly['Quarter'].apply(lambda x: x.year)
    quarterly['Q'] = quarterly['Quarter'].apply(lambda x: x.quarter)

    models = {}

    # Train models for Upper Funnel and Lower Funnel
    for funnel in ['Upper Funnel', 'Lower Funnel']:
        subset = quarterly[quarterly['Funnel Level'] == funnel].sort_values('Quarter')
        subset_with_spend = subset[subset['Spend'] > 1000].copy()

        X = subset_with_spend['Spend'].values.reshape(-1, 1)
        y = subset_with_spend['Revenue'].values

        # Linear
        lr = LinearRegression().fit(X, y)

        # Log-Log
        lr_log = LinearRegression().fit(np.log(X), np.log(y))

        # Recent iROAS by quarter
        q_iroas = {}
        for q in [1, 2, 3, 4]:
            q_data = subset_with_spend[subset_with_spend['Q'] == q]
            if len(q_data) > 0:
                q_iroas[q] = q_data['iROAS'].mean()
            else:
                q_iroas[q] = subset_with_spend['iROAS'].mean()

        recent_4q_iroas = subset_with_spend.tail(4)['iROAS'].mean()

        # Historical variability for confidence intervals (last 5 quarters only
        # to reflect current spend regime without mixing in earlier eras)
        recent_5q = subset_with_spend.tail(5)
        iroas_std = recent_5q['iROAS'].std()
        iroas_mean = recent_5q['iROAS'].mean()
        cv = iroas_std / iroas_mean if iroas_mean > 0 else 0.2

        models[funnel] = {
            'linear': lr,
            'loglog': lr_log,
            'q_iroas': q_iroas,
            'recent_4q_iroas': recent_4q_iroas,
            'cv': cv,
            'r2_linear': lr.score(X, y),
            'r2_loglog': lr_log.score(np.log(X), np.log(y)),
            'elasticity': lr_log.coef_[0],
            'historical': subset_with_spend,
        }

    # Unpaid model (time series)
    unpaid = quarterly[quarterly['Funnel Level'] == 'Unpaid'].sort_values('Quarter').copy()
    unpaid['Quarter_idx'] = range(len(unpaid))

    # Seasonal factors from stable pre-2025 era (unaffected by 2025 dip)
    pre2025 = unpaid[unpaid['Year'] < 2025]
    pre2025_avg = pre2025['Revenue'].mean()
    seasonal_factors = {}
    for q in [1, 2, 3, 4]:
        q_data = pre2025[pre2025['Q'] == q]['Revenue']
        if len(q_data) > 0:
            seasonal_factors[q] = q_data.mean() / pre2025_avg
        else:
            seasonal_factors[q] = 1.0

    # Recovery trend: 2025Q1 onward shows clear recovery trajectory
    # Use this to project forward rather than full history which gets
    # dragged down by the 2025 dip
    recovery = unpaid[unpaid['Year'] >= 2025].copy()
    recovery['recovery_idx'] = range(len(recovery))
    recovery_deseas = recovery.copy()
    recovery_deseas['deseas_rev'] = recovery_deseas.apply(
        lambda r: r['Revenue'] / seasonal_factors[r['Q']], axis=1
    )
    lr_recovery = LinearRegression().fit(
        recovery_deseas['recovery_idx'].values.reshape(-1, 1),
        recovery_deseas['deseas_rev'].values
    )

    # Full history trend + seasonality (as a secondary signal)
    unpaid_deseas = unpaid.copy()
    unpaid_deseas['deseas_rev'] = unpaid_deseas.apply(
        lambda r: r['Revenue'] / seasonal_factors[r['Q']], axis=1
    )
    lr_trend = LinearRegression().fit(
        unpaid_deseas['Quarter_idx'].values.reshape(-1, 1),
        unpaid_deseas['deseas_rev'].values
    )

    # Q1 trend models per quarter
    unpaid_q_trends = {}
    for q in [1, 2, 3, 4]:
        q_data = unpaid[unpaid['Q'] == q]
        if len(q_data) >= 2:
            lr_q = LinearRegression().fit(
                q_data['Year'].values.reshape(-1, 1),
                q_data['Revenue'].values
            )
            unpaid_q_trends[q] = lr_q

    # CV from recent quarters only (2025+) to avoid mixing eras
    recent_unpaid = unpaid[unpaid['Year'] >= 2025]
    recent_unpaid_deseas = recent_unpaid.apply(
        lambda r: r['Revenue'] / seasonal_factors[r['Q']], axis=1
    )
    unpaid_cv = recent_unpaid_deseas.std() / recent_unpaid_deseas.mean() if recent_unpaid_deseas.mean() > 0 else 0.15

    models['Unpaid'] = {
        'trend': lr_trend,
        'recovery_trend': lr_recovery,
        'recovery_start_n': len(recovery),  # number of quarters in recovery set
        'seasonal_factors': seasonal_factors,
        'q_trends': unpaid_q_trends,
        'n_quarters': len(unpaid),
        'last_quarter_idx': len(unpaid) - 1,  # 2026Q1
        'cv': unpaid_cv,
        'historical': unpaid,
    }

    return quarterly, models


def forecast_paid(models, funnel, spend, quarter):
    m = models[funnel]

    # Method 1: Linear
    linear_pred = m['linear'].predict([[spend]])[0]

    # Method 2: Log-Log
    loglog_pred = np.exp(m['loglog'].predict(np.log([[spend]]))[0])

    # Method 3: Recent 4-quarter iROAS (reflects current performance levels)
    iroas_pred = m['recent_4q_iroas'] * spend

    # Weighted blend
    weighted = 0.25 * linear_pred + 0.40 * loglog_pred + 0.35 * iroas_pred

    cv = m['cv']
    return {
        'linear': max(0, linear_pred),
        'loglog': max(0, loglog_pred),
        'iroas': max(0, iroas_pred),
        'weighted': max(0, weighted),
        'low': max(0, weighted * (1 - cv)),
        'high': weighted * (1 + cv),
        'implied_iroas': weighted / spend if spend > 0 else 0,
    }


def forecast_unpaid(models, quarter_num, year=2027):
    m = models['Unpaid']
    sf = m['seasonal_factors']

    # Method 1: Recovery trend (2025Q1+) — captures upward trajectory post-dip
    # Recovery data starts at 2025Q1 (idx 0). 2026Q1 = idx 4.
    # Target quarter offset from recovery start:
    quarters_from_2025q1 = (year - 2025) * 4 + (quarter_num - 1)
    recovery_pred = m['recovery_trend'].predict([[quarters_from_2025q1]])[0] * sf[quarter_num]

    # Method 2: Quarter-specific year trend (all history)
    trend_pred = 0
    if quarter_num in m['q_trends']:
        trend_pred = m['q_trends'][quarter_num].predict([[year]])[0]

    # Method 3: Full history trend + seasonality (conservative baseline)
    quarters_from_2026q1 = (year - 2026) * 4 + (quarter_num - 1)
    target_idx = m['last_quarter_idx'] + quarters_from_2026q1
    trend_seasonal = m['trend'].predict([[target_idx]])[0] * sf[quarter_num]

    # Weighted blend — recovery trend weighted highest since it reflects
    # the current trajectory without being dragged down by the 2025 dip
    if trend_pred > 0:
        weighted = 0.50 * recovery_pred + 0.30 * trend_pred + 0.20 * trend_seasonal
    else:
        weighted = 0.65 * recovery_pred + 0.35 * trend_seasonal

    cv = m['cv']
    return {
        'trend': max(0, trend_pred),
        'trend_seasonal': max(0, trend_seasonal),
        'weighted': max(0, weighted),
        'low': max(0, weighted * (1 - cv)),
        'high': weighted * (1 + cv),
    }


# =============================================================================
# APP UI
# =============================================================================
st.title("LiftLab Incremental Revenue Forecaster")
st.markdown("Forecast 2027 quarterly incremental revenue based on planned media spend.")

quarterly, models = load_and_train()

st.divider()

# Spend inputs
st.header("Planned Media Spend (2027)")

cols = st.columns(4)
quarter_labels = ['Q1 2027', 'Q2 2027', 'Q3 2027', 'Q4 2027']

spend_inputs = {}
for i, (col, label) in enumerate(zip(cols, quarter_labels)):
    q = i + 1
    with col:
        st.subheader(label)
        # Q1 defaults based on Q1 2026 actual spend levels
        spend_inputs[(q, 'Upper Funnel')] = st.number_input(
            f"Upper Funnel Spend",
            min_value=0, value=1_019_420 if q == 1 else 800_000,
            step=10_000, key=f"uf_{q}", format="%d"
        )
        spend_inputs[(q, 'Lower Funnel')] = st.number_input(
            f"Lower Funnel Spend",
            min_value=0, value=1_337_392 if q == 1 else 1_200_000,
            step=10_000, key=f"lf_{q}", format="%d"
        )

st.divider()

# Run forecast
if st.button("Run Forecast", type="primary", use_container_width=True):

    all_results = []

    for q in [1, 2, 3, 4]:
        for funnel in ['Upper Funnel', 'Lower Funnel']:
            spend = spend_inputs[(q, funnel)]
            if spend > 0:
                r = forecast_paid(models, funnel, spend, q)
            else:
                r = {'linear': 0, 'loglog': 0, 'iroas': 0, 'weighted': 0,
                     'low': 0, 'high': 0, 'implied_iroas': 0}
            all_results.append({
                'Quarter': f"Q{q} 2027",
                'Q': q,
                'Channel': funnel,
                'Spend': spend,
                **r
            })

        # Unpaid
        r = forecast_unpaid(models, q)
        all_results.append({
            'Quarter': f"Q{q} 2027",
            'Q': q,
            'Channel': 'Unpaid',
            'Spend': 0,
            'linear': 0, 'loglog': 0, 'iroas': 0,
            'implied_iroas': 0,
            **r
        })

    results_df = pd.DataFrame(all_results)

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    st.header("Forecast Results")

    # Quarterly summary
    for q in [1, 2, 3, 4]:
        q_data = results_df[results_df['Q'] == q]
        total_spend = q_data['Spend'].sum()
        total_rev = q_data['weighted'].sum()

        st.subheader(f"Q{q} 2027")

        summary_rows = []
        for _, row in q_data.iterrows():
            iroas_str = f"{row['implied_iroas']:.2f}x" if row['Spend'] > 0 else "N/A"
            summary_rows.append({
                'Channel': row['Channel'],
                'Planned Spend': f"${row['Spend']:,.0f}" if row['Spend'] > 0 else "$0",
                'Forecast Revenue': f"${row['weighted']:,.0f}",
                'iROAS': iroas_str,
                'Low Estimate': f"${row['low']:,.0f}",
                'High Estimate': f"${row['high']:,.0f}",
            })

        blended_iroas = total_rev / total_spend if total_spend > 0 else 0
        summary_rows.append({
            'Channel': 'TOTAL',
            'Planned Spend': f"${total_spend:,.0f}",
            'Forecast Revenue': f"${total_rev:,.0f}",
            'iROAS': f"{blended_iroas:.2f}x",
            'Low Estimate': f"${q_data['low'].sum():,.0f}",
            'High Estimate': f"${q_data['high'].sum():,.0f}",
        })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # ANNUAL SUMMARY
    # =========================================================================
    st.divider()
    st.header("2027 Annual Summary")

    annual = results_df.groupby('Channel').agg({
        'Spend': 'sum', 'weighted': 'sum', 'low': 'sum', 'high': 'sum'
    }).reset_index()

    annual_rows = []
    for _, row in annual.iterrows():
        iroas_str = f"{row['weighted']/row['Spend']:.2f}x" if row['Spend'] > 0 else "N/A"
        annual_rows.append({
            'Channel': row['Channel'],
            'Total Spend': f"${row['Spend']:,.0f}",
            'Total Forecast Revenue': f"${row['weighted']:,.0f}",
            'Blended iROAS': iroas_str,
            'Low Estimate': f"${row['low']:,.0f}",
            'High Estimate': f"${row['high']:,.0f}",
        })

    grand_spend = annual['Spend'].sum()
    grand_rev = annual['weighted'].sum()
    annual_rows.append({
        'Channel': 'GRAND TOTAL',
        'Total Spend': f"${grand_spend:,.0f}",
        'Total Forecast Revenue': f"${grand_rev:,.0f}",
        'Blended iROAS': f"{grand_rev/grand_spend:.2f}x" if grand_spend > 0 else "N/A",
        'Low Estimate': f"${annual['low'].sum():,.0f}",
        'High Estimate': f"${annual['high'].sum():,.0f}",
    })

    st.dataframe(pd.DataFrame(annual_rows), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Media Spend", f"${grand_spend:,.0f}")
    col2.metric("Total Forecast Revenue", f"${grand_rev:,.0f}")
    col3.metric("Blended iROAS", f"{grand_rev/grand_spend:.2f}x" if grand_spend > 0 else "N/A")

    # =========================================================================
    # CHARTS
    # =========================================================================
    st.divider()
    st.header("Visualizations")

    # Chart 1: Revenue by quarter and channel (stacked bar)
    fig1 = go.Figure()
    colors = {'Upper Funnel': '#636EFA', 'Lower Funnel': '#EF553B', 'Unpaid': '#00CC96'}

    for channel in ['Upper Funnel', 'Lower Funnel', 'Unpaid']:
        ch_data = results_df[results_df['Channel'] == channel]
        fig1.add_trace(go.Bar(
            name=channel,
            x=ch_data['Quarter'],
            y=ch_data['weighted'],
            marker_color=colors[channel],
            text=[f"${v:,.0f}" for v in ch_data['weighted']],
            textposition='inside',
        ))

    fig1.update_layout(
        barmode='stack',
        title='Forecast Revenue by Quarter & Channel',
        yaxis_title='Incremental Revenue ($)',
        yaxis_tickformat='$,.0f',
        height=500
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Spend vs Revenue with confidence bands
    col1, col2 = st.columns(2)

    with col1:
        fig2 = go.Figure()
        for channel in ['Upper Funnel', 'Lower Funnel']:
            ch_data = results_df[results_df['Channel'] == channel]
            fig2.add_trace(go.Bar(
                name=f"{channel} Spend", x=ch_data['Quarter'],
                y=ch_data['Spend'], marker_color=colors[channel], opacity=0.4,
            ))
            fig2.add_trace(go.Bar(
                name=f"{channel} Revenue", x=ch_data['Quarter'],
                y=ch_data['weighted'], marker_color=colors[channel],
            ))
        fig2.update_layout(
            barmode='group', title='Spend vs Forecast Revenue (Paid Channels)',
            yaxis_title='$', yaxis_tickformat='$,.0f', height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        for channel in ['Upper Funnel', 'Lower Funnel']:
            ch_data = results_df[results_df['Channel'] == channel]
            fig3.add_trace(go.Scatter(
                name=channel, x=ch_data['Quarter'],
                y=[r['implied_iroas'] for _, r in ch_data.iterrows()],
                mode='lines+markers', marker=dict(size=10),
                line=dict(color=colors[channel]),
            ))
        fig3.update_layout(
            title='Implied iROAS by Quarter',
            yaxis_title='iROAS (x)', height=450
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 3: Historical + forecast timeline
    st.subheader("Historical + Forecast Timeline")

    hist_quarterly = quarterly.groupby(['Quarter_str', 'Funnel Level']).agg({
        'Revenue': 'sum'
    }).reset_index()

    fig4 = go.Figure()
    for channel in ['Upper Funnel', 'Lower Funnel', 'Unpaid']:
        hist = hist_quarterly[hist_quarterly['Funnel Level'] == channel]
        fig4.add_trace(go.Scatter(
            name=f"{channel} (Historical)", x=hist['Quarter_str'],
            y=hist['Revenue'], mode='lines+markers',
            line=dict(color=colors[channel]),
        ))

        fore = results_df[results_df['Channel'] == channel]
        fig4.add_trace(go.Scatter(
            name=f"{channel} (Forecast)", x=fore['Quarter'],
            y=fore['weighted'], mode='lines+markers',
            line=dict(color=colors[channel], dash='dash'),
            showlegend=False,
        ))

    fig4.update_layout(
        title='Revenue Timeline: Historical vs Forecast',
        yaxis_title='Revenue ($)', yaxis_tickformat='$,.0f',
        height=500
    )
    st.plotly_chart(fig4, use_container_width=True)

    # =========================================================================
    # MODEL DETAILS (expandable)
    # =========================================================================
    with st.expander("Model Details & Methodology"):
        st.markdown("""
        ### Paid Channels (Upper Funnel & Lower Funnel)
        Three methods are blended with a weighted average:

        | Method | Weight | Description |
        |--------|--------|-------------|
        | **Linear Regression** | 25% | Spend vs Revenue linear fit |
        | **Log-Log Regression** | 40% | Captures diminishing returns at higher spend |
        | **Recent iROAS** | 35% | Recent 4-quarter historical iROAS average |

        ### Unpaid Channel
        Three methods are blended (seasonal factors derived from stable pre-2025 era):

        | Method | Weight | Description |
        |--------|--------|-------------|
        | **Recovery Trend (2025+)** | 50% | Captures upward trajectory post-2025 dip |
        | **Quarter-Specific Trend** | 30% | Linear trend on same-quarter historical data |
        | **Full History Trend + Seasonality** | 20% | Conservative baseline using all data |

        ### Confidence Ranges
        Based on the coefficient of variation (CV) of historical iROAS for paid channels,
        and revenue variability for unpaid.
        """)

        for funnel in ['Upper Funnel', 'Lower Funnel']:
            m = models[funnel]
            st.markdown(f"**{funnel}**: Linear R²={m['r2_linear']:.3f}, "
                        f"Log-Log R²={m['r2_loglog']:.3f}, "
                        f"Elasticity={m['elasticity']:.3f}, "
                        f"CV=±{m['cv']*100:.0f}%")

        st.markdown(f"**Unpaid**: CV=±{models['Unpaid']['cv']*100:.0f}%")
