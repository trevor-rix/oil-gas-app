import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="O&G Economics Engine", layout="wide")

st.title("🛢️ Custom Oil & Gas Economics Engine")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Forecast Parameters")

# Product Toggle
primary_product = st.sidebar.radio("Select Primary Product", ["Oil", "Gas"])

# Forecast Inputs
st.sidebar.subheader("Arps Decline Settings")
forecast_months = st.sidebar.number_input("Forecast Duration (Months)", value=360, step=12)

if primary_product == "Oil":
    qi_label = "Initial Rate (qi) [bbl/d]"
    qi_default = 500.0
else:
    qi_label = "Initial Rate (qi) [Mcf/d]"
    qi_default = 2500.0

qi = st.sidebar.number_input(qi_label, value=qi_default)
b_factor = st.sidebar.number_input("b-factor", value=1.2, min_value=0.0, max_value=2.0, step=0.1)
di_eff = st.sidebar.number_input("Initial Effective Annual Decline (%)", value=65.0, min_value=0.0, max_value=99.9) / 100
d_lim_eff = st.sidebar.number_input("Terminal Decline (Exponential) (%)", value=8.0, min_value=0.0, max_value=99.9) / 100

# Secondary Phase Logic
st.sidebar.subheader("Secondary Phase Ratios")
if primary_product == "Oil":
    gor = st.sidebar.number_input("GOR (SCF/bbl)", value=1000.0)
    # Visual placeholder for Gas
    lgr = 0
else:
    lgr = st.sidebar.number_input("LGR (bbl/MMcf)", value=40.0)
    # Visual placeholder for Oil
    gor = 0

st.sidebar.markdown("---")
st.sidebar.header("2. Economic Assumptions")

# Pricing
col1, col2 = st.sidebar.columns(2)
with col1:
    oil_price = st.number_input("Oil Price ($/bbl)", value=75.0)
    oil_diff = st.number_input("Oil Diff ($/bbl)", value=-5.0)
with col2:
    gas_price = st.number_input("Gas Price ($/Mcf)", value=3.50)
    gas_diff = st.number_input("Gas Diff ($/Mcf)", value=-0.25)

royalty_rate = st.sidebar.number_input("Royalty Rate (%)", value=12.5) / 100

# Costs
st.sidebar.subheader("OPEX & CAPEX")
fixed_cost = st.sidebar.number_input("Fixed Cost ($/Month)", value=2500.0)
var_oil_cost = st.sidebar.number_input("Variable Oil Cost ($/bbl)", value=2.0)
var_gas_cost = st.sidebar.number_input("Variable Gas Cost ($/Mcf)", value=0.25)

capex_mm = st.sidebar.number_input("Capital Cost (Time 0) ($MM)", value=5.0)
capex = capex_mm * 1_000_000

discount_rate = st.sidebar.number_input("Discount Rate (%/Year)", value=10.0) / 100

# --- CALCULATION ENGINE ---

def calculate_arps_modified(qi, b, di_eff, d_lim_eff, months):
    """
    Calculates monthly volumes based on Arps Modified Hyperbolic.
    """
    # Convert Effective Annual to Nominal Monthly
    if b == 0:
        ai = -np.log(1 - di_eff)
    else:
        term = (1 - di_eff) ** (-b)
        ai = (term - 1) / b
        
    di_nom_m = ai / 12  # Nominal Monthly Initial Decline
    
    # Terminal Decline Nominal Monthly
    d_lim_nom_m = -np.log(1 - d_lim_eff) / 12
    
    # Time vector
    t = np.arange(1, months + 1)
    
    # Switch time calculation
    if b > 0:
        if di_nom_m > d_lim_nom_m:
            t_switch = (di_nom_m / d_lim_nom_m - 1) / (b * di_nom_m)
        else:
            t_switch = 0
    else:
        t_switch = 0 # Exponential forever

    q_switch = 0
    if t_switch > 0:
         q_switch = qi / ((1 + b * di_nom_m * t_switch) ** (1 / b))

    # Vectorized calculation
    hyp_mask = t <= t_switch
    exp_mask = t > t_switch
    
    rate_hyp = np.zeros_like(t, dtype=float)
    rate_exp = np.zeros_like(t, dtype=float)

    if b > 0:
        rate_hyp[hyp_mask] = qi / ((1 + b * di_nom_m * t[hyp_mask]) ** (1 / b))
        if np.any(exp_mask):
            rate_exp[exp_mask] = q_switch * np.exp(-d_lim_nom_m * (t[exp_mask] - t_switch))
    else:
        rate_hyp = qi * np.exp(-di_nom_m * t)

    q_final = rate_hyp + rate_exp
    vol_monthly = q_final * 30.4167
    
    return t, vol_monthly, q_final

# 1. Forecast Volumes
t, prim_vol, prim_rate = calculate_arps_modified(qi, b_factor, di_eff, d_lim_eff, int(forecast_months))

if primary_product == "Oil":
    oil_vol = prim_vol
    oil_rate = prim_rate
    gas_vol = (oil_vol * gor) / 1000
    gas_rate = (oil_rate * gor) / 1000
else:
    gas_vol = prim_vol
    gas_rate = prim_rate
    oil_vol = (gas_vol / 1000) * lgr
    oil_rate = (gas_rate / 1000) * lgr

# 2. Create DataFrame
df = pd.DataFrame({
    "Month": t,
    "Oil_Vol_bbl": oil_vol,
    "Gas_Vol_Mcf": gas_vol,
    "Oil_Rate_bpd": oil_rate,
    "Gas_Rate_Mcfd": gas_rate
})

# 3. Economics Calculation
realized_oil = oil_price + oil_diff
realized_gas = gas_price + gas_diff

df["Gross_Rev_Oil"] = df["Oil_Vol_bbl"] * realized_oil
df["Gross_Rev_Gas"] = df["Gas_Vol_Mcf"] * realized_gas
df["Total_Gross_Rev"] = df["Gross_Rev_Oil"] + df["Gross_Rev_Gas"]

df["Royalty_Paid"] = df["Total_Gross_Rev"] * royalty_rate
df["Net_Rev"] = df["Total_Gross_Rev"] - df["Royalty_Paid"]

df["Fixed_Opex"] = fixed_cost
df["Var_Opex"] = (df["Oil_Vol_bbl"] * var_oil_cost) + (df["Gas_Vol_Mcf"] * var_gas_cost)
df["Total_Opex"] = df["Fixed_Opex"] + df["Var_Opex"]

df["Op_Cashflow"] = df["Net_Rev"] - df["Total_Opex"]

disc_monthly = (1 + discount_rate)**(1/12) - 1
df["Disc_Factor"] = 1 / ((1 + disc_monthly) ** df["Month"])
df["Disc_Cashflow"] = df["Op_Cashflow"] * df["Disc_Factor"]

# --- METRICS ---
total_pv = df["Disc_Cashflow"].sum()
npv = total_pv - capex
pir = total_pv / capex if capex > 0 else 0
cf_stream = np.insert(df["Op_Cashflow"].values, 0, -capex)
try:
    irr = npf.irr(cf_stream) * 12
except:
    irr = np.nan

# --- DASHBOARD LAYOUT ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("NPV", f"${npv/1_000_000:,.2f} MM")
m2.metric("IRR", f"{irr*100:.1f}%" if not np.isnan(irr) else "N/A")
m3.metric("PIR (ROI)", f"{pir:.2f}x")
m4.metric("Payout", f"{(capex / df['Op_Cashflow'].mean()):.1f} Months (Avg)" )

tab1, tab2 = st.tabs(["📊 Production Plots", "💰 Cash Flow Analysis"])

with tab1:
    fig_prod = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prod.add_trace(go.Scatter(x=df["Month"], y=df["Oil_Rate_bpd"], name="Oil Rate (bpd)", line=dict(color='green')), secondary_y=False)
    fig_prod.add_trace(go.Scatter(x=df["Month"], y=df["Gas_Rate_Mcfd"], name="Gas Rate (Mcfd)", line=dict(color='red', dash='dot')), secondary_y=True)
    fig_prod.update_layout(title="Wellhead Production", xaxis_title="Month", yaxis_type="log", height=500)
    st.plotly_chart(fig_prod, use_container_width=True)

with tab2:
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(x=df["Month"], y=df["Op_Cashflow"], name="Monthly Op Cashflow", marker_color='blue'))
    fig_cf.add_trace(go.Scatter(x=df["Month"], y=df["Op_Cashflow"].cumsum() - capex, name="Cumulative Net Cashflow", line=dict(color='black')))
    fig_cf.add_hline(y=0, line_dash="dash", line_color="red")
    fig_cf.update_layout(title="Cash Flow & Payout", xaxis_title="Month", height=500)
    st.plotly_chart(fig_cf, use_container_width=True)
