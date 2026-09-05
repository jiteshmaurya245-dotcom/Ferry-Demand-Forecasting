import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page setup & visual identity
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Toronto Island Ferry — Demand Forecast", layout="wide")

NAVY   = "#0B3D4C"   # actual demand / headers
TEAL   = "#147D82"   # predicted demand / accent
AMBER  = "#E8A33D"   # uncertainty band
INK    = "#1F2937"
PAPER  = "#F7F9FA"
MUTED  = "#6B7280"
BORDER = "#E2E8F0"
GRID   = "#EEF1F3"

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap">
<style>
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; color: {INK} !important; }}
.stApp {{ background-color: {PAPER}; }}
section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E7F1F2 !important; }}
/* The rule above also lightened text INSIDE the input boxes below, which
   still have a white background (unlike the dark sidebar around them) --
   that made the selected date/time/model value nearly invisible. Force
   those specific value areas back to dark text since they sit on white. */
section[data-testid="stSidebar"] div[data-testid="stDateInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTimeInput"] input,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] * {{ color: {INK} !important; }}
section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stDateInput label, section[data-testid="stSidebar"] .stTimeInput label {{ font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(231,241,242,0.15); }}
.sidebar-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: white !important; margin-bottom: 0.1rem; }}
div[data-testid="stMetric"] {{ background-color: white; border: 1px solid {BORDER}; border-left: 4px solid {TEAL}; border-radius: 10px; padding: 1rem 1.1rem; box-shadow: 0 1px 2px rgba(15,23,42,0.05); transition: box-shadow 0.15s ease, transform 0.15s ease; }}
div[data-testid="stMetric"]:hover {{ box-shadow: 0 6px 16px rgba(15,23,42,0.10); transform: translateY(-1px); }}
div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
div[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; color: {NAVY}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px !important; border-color: {BORDER} !important; background-color: white; box-shadow: 0 1px 3px rgba(15,23,42,0.04); }}
.accent-bar {{ height: 4px; width: 64px; background: linear-gradient(90deg, {TEAL}, {AMBER}); border-radius: 3px; margin: 0.3rem 0 1.1rem 0; }}
.section-label {{ font-weight: 600; font-size: 0.95rem; color: {NAVY}; margin-bottom: 0.3rem; }}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("app_predictions.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

df = load_data()

HORIZONS = ["15 min", "30 min", "1 hour", "2 hours"]
MODELS = {
    "Gradient Boosting (recommended)": "gbm_pred",
    "Linear Regression": "lr_pred",
    "Moving Average": "ma_pred",
    "Naive (last known value)": "naive_pred",
}
MODEL_SHORT = {"gbm_pred": "Gradient Boosting", "lr_pred": "Linear reg", "ma_pred": "Moving avg", "naive_pred": "Naive"}

# ---------------------------------------------------------------------------
# Sidebar — the operator's controls
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">⛴️ Ferry Forecast</div>', unsafe_allow_html=True)
st.sidebar.caption("Toronto Island Park · Parks, Forestry & Recreation")
st.sidebar.markdown("---")

min_date, max_date = df["Timestamp"].min().date(), df["Timestamp"].max().date()
sel_date = st.sidebar.date_input("Date", value=min_date, min_value=min_date, max_value=max_date)
sel_time = st.sidebar.time_input("Time", value=pd.Timestamp("12:00").time(), step=900)  # 15-min steps
horizon = st.sidebar.selectbox("How far ahead to forecast", HORIZONS, index=0)
model_label = st.sidebar.selectbox("Model", list(MODELS.keys()), index=0)
model_col = MODELS[model_label]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Forecasts shown here are evaluated on a held-out test period "
    f"({df['Timestamp'].min():%b %d, %Y} – {df['Timestamp'].max():%b %d, %Y}) "
    "the models never trained on — this is a backtest, not a live feed."
)

sel_dt = pd.Timestamp.combine(sel_date, sel_time)
hz = df[df["horizon"] == horizon].sort_values("Timestamp").reset_index(drop=True)

idx_nearest = (hz["Timestamp"] - sel_dt).abs().idxmin()
anchor_ts = hz.loc[idx_nearest, "Timestamp"]
day_window = hz[hz["Timestamp"].dt.date == anchor_ts.date()]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"### Short-term ticket demand — {horizon} ahead")
st.caption(f"Showing {anchor_ts:%A, %B %d %Y} · model: {model_label}")
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
mae = (hz["target"] - hz[model_col]).abs().mean()
rmse = ((hz["target"] - hz[model_col]) ** 2).mean() ** 0.5
nonzero = hz["target"] > 0
mape = (np.abs((hz.loc[nonzero, "target"] - hz.loc[nonzero, model_col]) / hz.loc[nonzero, "target"])).mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg. error (MAE)", f"{mae:.1f}", help="Average absolute miss, in tickets per 15-min interval.")
c2.metric("RMSE", f"{rmse:.1f}", help="Like MAE, but penalizes big misses more heavily.")
c3.metric("Typical % error", f"{mape:.0f}%", help="Unreliable when actual counts are near zero — treat MAE/RMSE as primary.")
if model_col == "gbm_pred":
    coverage = ((hz["target"] >= hz["gbm_lower"]) & (hz["target"] <= hz["gbm_upper"])).mean() * 100
    band = (hz["gbm_upper"] - hz["gbm_lower"]).mean()
    c4.metric("Band width (±)", f"{band/2:.0f}", help=f"80% confidence interval captured the true value {coverage:.0f}% of the time on the test set.")
else:
    c4.metric("Band width (±)", "n/a", help="Only Gradient Boosting produces an uncertainty range in this dashboard.")

st.write("")

# ---------------------------------------------------------------------------
# Forecast chart + model comparison
# ---------------------------------------------------------------------------
left, right = st.columns([2.3, 1])

with left:
    with st.container(border=True):
        st.markdown('<div class="section-label">Predicted vs. actual — selected day</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if model_col == "gbm_pred":
            fig.add_trace(go.Scatter(x=day_window["Timestamp"], y=day_window["gbm_upper"], mode="lines",
                                      line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=day_window["Timestamp"], y=day_window["gbm_lower"], mode="lines",
                                      line=dict(width=0), fill="tonexty", fillcolor="rgba(232,163,61,0.28)",
                                      name="Confidence band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=day_window["Timestamp"], y=day_window["target"], mode="lines",
                                  line=dict(color=NAVY, width=2.4), name="Actual sales",
                                  hovertemplate="%{y:.0f} tickets<extra>Actual</extra>"))
        fig.add_trace(go.Scatter(x=day_window["Timestamp"], y=day_window[model_col], mode="lines",
                                  line=dict(color=TEAL, width=2.4, dash="dash"), name="Forecast",
                                  hovertemplate="%{y:.0f} tickets<extra>Forecast</extra>"))
        fig.add_vline(x=anchor_ts.to_pydatetime(), line_dash="dot", line_color=MUTED, line_width=1.3)
        fig.add_annotation(x=anchor_ts.to_pydatetime(), y=1, yref="paper", yanchor="bottom",
                            text="Selected time", showarrow=False, font=dict(size=11, color=MUTED))
        fig.update_layout(
            template="plotly_white", hovermode="x unified",
            font=dict(family="Inter, sans-serif", color=INK, size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=45, b=10), height=420,
            plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=None, tickformat="%H:%M", showgrid=False),
            yaxis=dict(title="Tickets / 15-min interval", gridcolor=GRID, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    with st.container(border=True):
        st.markdown('<div class="section-label">Model comparison</div>', unsafe_allow_html=True)
        comp = hz.melt(id_vars=["target"], value_vars=["naive_pred", "ma_pred", "lr_pred", "gbm_pred"],
                        var_name="model", value_name="pred")
        comp["abs_err"] = (comp["target"] - comp["pred"]).abs()
        comp_summary = comp.groupby("model")["abs_err"].mean().rename(index=MODEL_SHORT)
        comp_summary = comp_summary.sort_values(ascending=True)
        best_model = comp_summary.idxmin()
        colors = [TEAL if m == best_model else "#C7D2D6" for m in comp_summary.index]

        fig2 = go.Figure(go.Bar(
            x=comp_summary.values, y=comp_summary.index, orientation="h",
            marker_color=colors, text=[f"{v:.1f}" for v in comp_summary.values], textposition="outside",
            hovertemplate="%{y}: %{x:.1f} avg. error<extra></extra>",
        ))
        fig2.update_layout(
            template="plotly_white", font=dict(family="Inter, sans-serif", color=INK, size=12),
            margin=dict(l=10, r=25, t=10, b=10), height=340,
            plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Avg. error (MAE)", gridcolor=GRID),
            yaxis=dict(title=None, autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.caption("Lower is better · computed on the full test period at this horizon.")

# ---------------------------------------------------------------------------
# Predicted vs actual table
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="section-label">Interval-by-interval detail</div>', unsafe_allow_html=True)
    table = day_window[["Timestamp", "target", model_col]].copy()
    table.columns = ["Time", "Actual", "Predicted"]
    table["Predicted"] = table["Predicted"].round(1)
    table["Error"] = (table["Actual"] - table["Predicted"]).round(1)
    table["Time"] = table["Time"].dt.strftime("%H:%M")
    st.dataframe(
        table, width="stretch", height=260, hide_index=True,
        column_config={
            "Actual": st.column_config.NumberColumn("Actual", format="%d"),
            "Predicted": st.column_config.NumberColumn("Predicted", format="%.1f"),
            "Error": st.column_config.NumberColumn("Error", format="%.1f"),
        },
    )
