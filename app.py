import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Big Mac Index Analytics",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global dark theme */
    .stApp {
        background-color: #0a0a0f;
        color: #e0e0e0;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(255, 183, 77, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 50%, rgba(255, 183, 77, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFB74D, #FF8A65, #FFD54F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-family: 'Outfit', sans-serif;
        color: #8899aa;
        font-size: 1rem;
        font-weight: 300;
    }

    /* KPI cards */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .kpi-card {
        background: linear-gradient(145deg, #12121f, #1a1a2e);
        border: 1px solid rgba(255, 183, 77, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        flex: 1;
        min-width: 160px;
        text-align: center;
        transition: border-color 0.3s;
    }
    .kpi-card:hover { border-color: rgba(255, 183, 77, 0.5); }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFB74D;
    }
    .kpi-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        color: #7788aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFB74D;
        border-left: 4px solid #FF8A65;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .section-sub {
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem;
        color: #7788aa;
        margin-bottom: 1rem;
        padding-left: 1.2rem;
    }

    /* Analysis type badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-right: 0.4rem;
    }
    .badge-desc { background: rgba(76, 175, 80, 0.2); color: #81C784; border: 1px solid rgba(76, 175, 80, 0.3); }
    .badge-diag { background: rgba(33, 150, 243, 0.2); color: #64B5F6; border: 1px solid rgba(33, 150, 243, 0.3); }
    .badge-pred { background: rgba(156, 39, 176, 0.2); color: #BA68C8; border: 1px solid rgba(156, 39, 176, 0.3); }
    .badge-pres { background: rgba(255, 152, 0, 0.2); color: #FFB74D; border: 1px solid rgba(255, 152, 0, 0.3); }

    /* Insight box */
    .insight-box {
        background: linear-gradient(145deg, #0d1b2a, #1b2838);
        border: 1px solid rgba(100, 181, 246, 0.25);
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin: 0.8rem 0;
        font-family: 'Outfit', sans-serif;
        font-size: 0.88rem;
        color: #b0c4de;
        line-height: 1.6;
    }
    .insight-box strong { color: #64B5F6; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a, #12121f);
        border-right: 1px solid rgba(255, 183, 77, 0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
    }

    /* Plotly chart container */
    .chart-container {
        background: #12121f;
        border: 1px solid rgba(255, 183, 77, 0.1);
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────────────
DARK_BG = "#0a0a0f"
CARD_BG = "#12121f"
GRID_COLOR = "rgba(255,255,255,0.05)"
ACCENT = "#FFB74D"
ACCENT2 = "#FF8A65"
ACCENT3 = "#64B5F6"
ACCENT4 = "#81C784"
ACCENT5 = "#BA68C8"

COLOR_PALETTE = [
    "#FFB74D", "#64B5F6", "#81C784", "#FF8A65", "#BA68C8",
    "#4DD0E1", "#FFD54F", "#E57373", "#AED581", "#7986CB",
    "#F06292", "#4DB6AC", "#DCE775", "#FF8A80", "#80DEEA",
    "#FFCC80", "#B39DDB", "#C5E1A5", "#FFAB91", "#80CBC4",
]

DIVERGING_COLORS = [
    [0, "#E53935"], [0.25, "#FF8A65"], [0.5, "#FAFAFA"],
    [0.75, "#64B5F6"], [1, "#1E88E5"]
]

def dark_layout(fig, title="", height=500, showlegend=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Outfit, sans-serif", color="#c0c0c0", size=12),
        title=dict(text=title, font=dict(size=16, color=ACCENT, family="Outfit, sans-serif"), x=0.02),
        height=height,
        showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("big_mac_v2.csv")
    df.columns = df.columns.str.strip()
    # Shorten column names for convenience
    df = df.rename(columns={
        "observation_date_january_first": "date",
        "country_iso3_code": "iso3",
        "country_currency_iso_code": "currency_code",
        "country_name": "country",
        "big_mac_price_in_local_currency": "local_price",
        "local_currency_units_per_us_dollar_exchange_rate": "exchange_rate",
        "big_mac_price_converted_to_us_dollars": "usd_price",
        "currency_misvaluation_vs_usd_raw_big_mac_index": "raw_index_usd",
        "currency_misvaluation_vs_euro_raw_big_mac_index": "raw_index_eur",
        "currency_misvaluation_vs_gbp_raw_big_mac_index": "raw_index_gbp",
        "currency_misvaluation_vs_jpy_raw_big_mac_index": "raw_index_jpy",
        "currency_misvaluation_vs_cny_raw_big_mac_index": "raw_index_cny",
        "gdp_per_capita_in_us_dollars_used_for_adjusted_big_mac_index": "gdp_per_capita",
        "big_mac_price_adjusted_for_gdp_per_capita": "gdp_adj_price",
        "currency_misvaluation_vs_usd_gdp_adjusted_big_mac_index": "adj_index_usd",
        "currency_misvaluation_vs_euro_gdp_adjusted_big_mac_index": "adj_index_eur",
        "currency_misvaluation_vs_gbp_gdp_adjusted_big_mac_index": "adj_index_gbp",
        "currency_misvaluation_vs_jpy_gdp_adjusted_big_mac_index": "adj_index_jpy",
        "currency_misvaluation_vs_cny_gdp_adjusted_big_mac_index": "adj_index_cny",
        "observation_year": "year",
        "consumer_price_index_world_bank_2010_base": "cpi",
        "annual_consumer_price_index_inflation_rate": "inflation",
        "inflation_adjusted_big_mac_price_in_local_currency": "real_local_price",
        "inflation_adjusted_big_mac_price_in_us_dollars": "real_usd_price",
        "purchasing_power_parity_currency_misvaluation_vs_usd": "ppp_misval",
    })
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)

    # Region mapping
    region_map = {
        "United States": "North America", "Canada": "North America", "Mexico": "North America",
        "Guatemala": "Central America", "Honduras": "Central America", "Costa Rica": "Central America", "Nicaragua": "Central America",
        "Brazil": "South America", "Argentina": "South America", "Chile": "South America",
        "Colombia": "South America", "Peru": "South America", "Uruguay": "South America",
        "Britain": "Europe", "Euro area": "Europe", "Switzerland": "Europe", "Sweden": "Europe",
        "Norway": "Europe", "Denmark": "Europe", "Poland": "Europe", "Czech Republic": "Europe",
        "Hungary": "Europe", "Romania": "Europe", "Moldova": "Europe", "Russia": "Europe",
        "Ukraine": "Europe", "Turkey": "Europe",
        "China": "East Asia", "Japan": "East Asia", "South Korea": "East Asia",
        "Hong Kong": "East Asia", "Taiwan": "East Asia",
        "India": "South Asia", "Pakistan": "South Asia", "Sri Lanka": "South Asia",
        "Thailand": "Southeast Asia", "Malaysia": "Southeast Asia", "Singapore": "Southeast Asia",
        "Indonesia": "Southeast Asia", "Philippines": "Southeast Asia", "Vietnam": "Southeast Asia",
        "Australia": "Oceania", "New Zealand": "Oceania",
        "UAE": "Middle East", "Saudi Arabia": "Middle East", "Kuwait": "Middle East",
        "Qatar": "Middle East", "Bahrain": "Middle East", "Oman": "Middle East",
        "Jordan": "Middle East", "Israel": "Middle East", "Lebanon": "Middle East",
        "Egypt": "Africa", "South Africa": "Africa",
        "Azerbaijan": "Central Asia",
    }
    df["region"] = df["country"].map(region_map).fillna("Other")

    # Income classification based on GDP per capita
    def classify_income(gdp):
        if pd.isna(gdp): return "Unknown"
        if gdp > 40000: return "High Income"
        elif gdp > 12000: return "Upper-Middle"
        elif gdp > 4000: return "Lower-Middle"
        else: return "Low Income"
    df["income_group"] = df["gdp_per_capita"].apply(classify_income)

    return df


df = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🍔 Big Mac Index — Advanced Analytics Dashboard</h1>
    <p>Descriptive · Diagnostic · Predictive · Prescriptive — 55 Countries · 2001–2025</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters & Controls")
    st.markdown("---")

    year_range = st.slider(
        "📅 Year Range",
        int(df["year"].min()), int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())),
    )

    all_countries = sorted(df["country"].unique())
    selected_countries = st.multiselect(
        "🌍 Select Countries",
        all_countries,
        default=["United States", "China", "India", "Britain", "Switzerland", "Brazil", "Japan", "Australia"],
        help="Choose countries to compare"
    )

    all_regions = sorted(df["region"].unique())
    selected_regions = st.multiselect(
        "🗺️ Filter by Region",
        all_regions,
        default=all_regions,
    )

    base_currency = st.selectbox(
        "💱 Base Currency for Misvaluation",
        ["USD", "EUR", "GBP", "JPY", "CNY"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📖 Analysis Types")
    st.markdown("""
    <span class="badge badge-desc">Descriptive</span> What happened?<br>
    <span class="badge badge-diag">Diagnostic</span> Why did it happen?<br>
    <span class="badge badge-pred">Predictive</span> What will happen?<br>
    <span class="badge badge-pres">Prescriptive</span> What should we do?
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#556677;text-align:center;'>SP Jain School of Global Management<br>MBA Applied Research Project</p>",
        unsafe_allow_html=True,
    )

# Map base currency to column suffix
currency_col_map = {"USD": "usd", "EUR": "eur", "GBP": "gbp", "JPY": "jpy", "CNY": "cny"}
raw_col = f"raw_index_{currency_col_map[base_currency]}"
adj_col = f"adj_index_{currency_col_map[base_currency]}"

# Apply filters
fdf = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1]) & (df["region"].isin(selected_regions))]
sdf = fdf[fdf["country"].isin(selected_countries)]  # selected countries subset

# ──────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────────────────────────────────────
latest_year = fdf["year"].max()
latest = fdf[fdf["year"] == latest_year]

avg_usd = latest["usd_price"].mean()
most_expensive = latest.loc[latest["usd_price"].idxmax(), "country"] if len(latest) > 0 else "N/A"
cheapest = latest.loc[latest["usd_price"].idxmin(), "country"] if len(latest) > 0 else "N/A"
most_overval = latest.loc[latest[raw_col].idxmax(), "country"] if len(latest) > 0 else "N/A"
most_underval = latest.loc[latest[raw_col].idxmin(), "country"] if len(latest) > 0 else "N/A"

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-value">${avg_usd:.2f}</div>
        <div class="kpi-label">Avg Big Mac (USD) — {latest_year}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{len(latest)}</div>
        <div class="kpi-label">Countries Tracked</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="font-size:1.1rem;">{most_expensive}</div>
        <div class="kpi-label">Most Expensive</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="font-size:1.1rem;">{cheapest}</div>
        <div class="kpi-label">Cheapest</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="font-size:1.1rem;color:#81C784;">{most_overval}</div>
        <div class="kpi-label">Most Overvalued ({base_currency})</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="font-size:1.1rem;color:#E57373;">{most_underval}</div>
        <div class="kpi-label">Most Undervalued ({base_currency})</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 1: CURRENCY MISVALUATION ASSESSMENT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Objective 1 — Currency Misvaluation Assessment</div>
<div class="section-sub">
    <span class="badge badge-desc">Descriptive</span>
    <span class="badge badge-diag">Diagnostic</span>
    Assess currency misvaluation using the Big Mac PPP Index (2001–2025)
</div>
""", unsafe_allow_html=True)

tab1a, tab1b, tab1c, tab1d = st.tabs(["🌍 Global Map", "🍩 Drill-Down Donut", "📊 Ranked Bar", "📈 Time Series"])

with tab1a:
    map_year = st.select_slider("Select Year for Map", options=sorted(fdf["year"].unique()), value=latest_year, key="map_yr1")
    map_data = fdf[fdf["year"] == map_year].copy()
    map_data["misval_pct"] = map_data[raw_col] * 100

    fig_map = px.choropleth(
        map_data, locations="iso3", color="misval_pct",
        hover_name="country",
        hover_data={"usd_price": ":.2f", "misval_pct": ":.1f", "iso3": False},
        color_continuous_scale=[[0, "#B71C1C"], [0.3, "#E57373"], [0.5, "#FAFAFA"], [0.7, "#64B5F6"], [1, "#0D47A1"]],
        color_continuous_midpoint=0,
        labels={"misval_pct": f"Misvaluation vs {base_currency} (%)"},
    )
    fig_map = dark_layout(fig_map, f"Currency Misvaluation vs {base_currency} — {map_year}", height=520)
    fig_map.update_geos(
        bgcolor=CARD_BG, landcolor="#1a1a2e", oceancolor="#0a0a14",
        showframe=False, coastlinecolor="#333",
        projection_type="natural earth",
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab1b:
    st.markdown("**Click a region slice to drill down into individual countries:**")
    donut_year = st.select_slider("Year", options=sorted(fdf["year"].unique()), value=latest_year, key="donut_yr1")
    donut_data = fdf[fdf["year"] == donut_year].copy()
    donut_data["abs_misval"] = donut_data[raw_col].abs() * 100
    donut_data["status"] = donut_data[raw_col].apply(lambda x: "Overvalued" if x > 0 else "Undervalued")

    # Region-level donut
    region_agg = donut_data.groupby("region").agg(
        avg_misval=(raw_col, "mean"),
        count=("country", "count"),
        countries=("country", lambda x: ", ".join(sorted(x))),
    ).reset_index()
    region_agg["avg_misval_pct"] = region_agg["avg_misval"] * 100
    region_agg["abs_misval"] = region_agg["avg_misval_pct"].abs()
    region_agg["status"] = region_agg["avg_misval_pct"].apply(lambda x: "Overvalued" if x > 0 else "Undervalued")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig_donut_region = px.sunburst(
            donut_data, path=["region", "country"], values="abs_misval",
            color="status", color_discrete_map={"Overvalued": "#64B5F6", "Undervalued": "#E57373"},
            hover_data={raw_col: ":.3f"},
        )
        fig_donut_region = dark_layout(fig_donut_region, f"Misvaluation by Region → Country ({donut_year})", height=520)
        st.plotly_chart(fig_donut_region, use_container_width=True)

    with col_d2:
        # Overvalued vs Undervalued split donut
        ov_count = (donut_data[raw_col] > 0).sum()
        uv_count = (donut_data[raw_col] <= 0).sum()
        fig_split = go.Figure(data=[go.Pie(
            labels=["Overvalued", "Undervalued"],
            values=[ov_count, uv_count],
            hole=0.55,
            marker=dict(colors=["#64B5F6", "#E57373"], line=dict(color=CARD_BG, width=3)),
            textinfo="label+percent+value",
            textfont=dict(size=13),
        )])
        fig_split = dark_layout(fig_split, f"Over vs Undervalued Countries ({donut_year})", height=520)
        fig_split.update_layout(
            annotations=[dict(text=f"{donut_year}", x=0.5, y=0.5, font_size=24, font_color=ACCENT, showarrow=False)]
        )
        st.plotly_chart(fig_split, use_container_width=True)

with tab1c:
    bar_year = st.select_slider("Year", options=sorted(fdf["year"].unique()), value=latest_year, key="bar_yr1")
    bar_data = fdf[fdf["year"] == bar_year].copy()
    bar_data["misval_pct"] = bar_data[raw_col] * 100
    bar_data = bar_data.sort_values("misval_pct")
    bar_data["color"] = bar_data["misval_pct"].apply(lambda x: "#64B5F6" if x > 0 else "#E57373")

    fig_bar = go.Figure(go.Bar(
        x=bar_data["misval_pct"], y=bar_data["country"], orientation="h",
        marker_color=bar_data["color"],
        text=bar_data["misval_pct"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>Misvaluation: %{x:.1f}%<extra></extra>",
    ))
    fig_bar = dark_layout(fig_bar, f"Currency Misvaluation Ranking vs {base_currency} — {bar_year}", height=max(600, len(bar_data) * 18))
    fig_bar.update_layout(xaxis_title=f"Misvaluation vs {base_currency} (%)", yaxis=dict(tickfont=dict(size=10)))
    fig_bar.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab1d:
    if len(selected_countries) > 0:
        fig_ts = go.Figure()
        for i, c in enumerate(selected_countries):
            cdata = sdf[sdf["country"] == c]
            fig_ts.add_trace(go.Scatter(
                x=cdata["year"], y=cdata[raw_col] * 100,
                name=c, mode="lines+markers",
                line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                marker=dict(size=5),
            ))
        fig_ts.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_ts = dark_layout(fig_ts, f"Currency Misvaluation Trend vs {base_currency}", height=500)
        fig_ts.update_layout(xaxis_title="Year", yaxis_title=f"Misvaluation vs {base_currency} (%)")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Select countries in the sidebar to view time series.")

# Descriptive insight
desc_latest = fdf[fdf["year"] == latest_year]
avg_misval = desc_latest[raw_col].mean() * 100
std_misval = desc_latest[raw_col].std() * 100
overval_pct = (desc_latest[raw_col] > 0).mean() * 100
st.markdown(f"""
<div class="insight-box">
    <strong>📊 Descriptive Insight ({latest_year}):</strong> The average currency misvaluation vs {base_currency} across {len(desc_latest)} countries
    is <strong>{avg_misval:+.1f}%</strong> (σ = {std_misval:.1f}%). Approximately <strong>{overval_pct:.0f}%</strong> of currencies appear overvalued
    relative to {base_currency} purchasing power parity, suggesting the US dollar remains generally strong in Big Mac terms.<br><br>
    <strong>🔍 Diagnostic Insight:</strong> Countries with persistent undervaluation (e.g., emerging markets) often reflect
    lower labor costs and non-tradable input prices (rent, utilities) that make Big Mac production cheaper — a structural
    feature not captured by exchange rates alone. Overvalued currencies in Nordic/Swiss economies reflect higher
    wage levels and cost structures rather than pure currency distortion.
</div>
""", unsafe_allow_html=True)

# CSV export for Objective 1
with st.expander("📥 Download Objective 1 Data"):
    export1 = fdf[["year", "country", "region", "iso3", "usd_price", raw_col, adj_col]].copy()
    export1.columns = ["Year", "Country", "Region", "ISO3", "USD Price", f"Raw Misval vs {base_currency}", f"GDP-Adj Misval vs {base_currency}"]
    st.download_button("Download CSV", export1.to_csv(index=False), "obj1_misvaluation.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 2: INFLATION & BIG MAC PRICE CHANGES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Objective 2 — Inflation & Big Mac Price Dynamics</div>
<div class="section-sub">
    <span class="badge badge-desc">Descriptive</span>
    <span class="badge badge-diag">Diagnostic</span>
    Analyze inflation vs. Big Mac price changes across countries over time
</div>
""", unsafe_allow_html=True)

# Exclude countries/years with missing CPI
inf_df = fdf.dropna(subset=["inflation", "cpi"]).copy()
# Compute year-over-year Big Mac price change
inf_df = inf_df.sort_values(["country", "year"])
inf_df["price_change_pct"] = inf_df.groupby("country")["local_price"].pct_change() * 100

tab2a, tab2b, tab2c = st.tabs(["📈 Scatter Analysis", "🔥 Heatmap", "📊 Comparison"])

with tab2a:
    scatter_year = st.select_slider("Year", options=sorted(inf_df["year"].unique()), value=inf_df["year"].max(), key="sc_yr2")
    sc_data = inf_df[inf_df["year"] == scatter_year].dropna(subset=["price_change_pct"])

    if len(sc_data) > 2:
        fig_sc = px.scatter(
            sc_data, x="inflation", y="price_change_pct",
            color="region", size="usd_price",
            hover_name="country",
            color_discrete_sequence=COLOR_PALETTE,
            labels={"inflation": "CPI Inflation Rate (%)", "price_change_pct": "Big Mac Price Change (%)"},
            trendline="ols",
        )
        fig_sc = dark_layout(fig_sc, f"Inflation vs Big Mac Price Change — {scatter_year}", height=500)
        # Add 45-degree reference line
        max_val = max(sc_data["inflation"].abs().max(), sc_data["price_change_pct"].abs().max()) * 1.1
        fig_sc.add_trace(go.Scatter(
            x=[-max_val, max_val], y=[-max_val, max_val],
            mode="lines", line=dict(dash="dot", color="rgba(255,255,255,0.2)"),
            name="1:1 Line", showlegend=True,
        ))
        st.plotly_chart(fig_sc, use_container_width=True)

        # Compute correlation
        valid = sc_data.dropna(subset=["inflation", "price_change_pct"])
        if len(valid) > 2:
            r, p = stats.pearsonr(valid["inflation"], valid["price_change_pct"])
            st.markdown(f"""
            <div class="insight-box">
                <strong>📊 Descriptive:</strong> Pearson correlation between inflation and Big Mac price change in {scatter_year}:
                <strong>r = {r:.3f}</strong> (p = {p:.4f}). {"Statistically significant" if p < 0.05 else "Not statistically significant"} at 5% level.<br>
                <strong>🔍 Diagnostic:</strong> {"Strong positive correlation suggests Big Mac prices closely track CPI inflation — consistent with cost-push pricing where input costs (beef, wheat, labor) rise with general inflation." if r > 0.5 else "Moderate/weak correlation indicates Big Mac prices don't perfectly mirror inflation — McDonald's pricing power, local competition, and menu strategy create deviations from pure inflation pass-through."}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Insufficient data for selected year. Try a different year.")

with tab2b:
    # Inflation heatmap over time for selected countries
    heat_countries = selected_countries if selected_countries else all_countries[:10]
    heat_data = inf_df[inf_df["country"].isin(heat_countries)].pivot_table(
        index="country", columns="year", values="inflation", aggfunc="mean"
    )
    if not heat_data.empty:
        fig_heat = px.imshow(
            heat_data, color_continuous_scale="RdYlBu_r",
            labels=dict(x="Year", y="Country", color="Inflation %"),
            aspect="auto",
        )
        fig_heat = dark_layout(fig_heat, "Inflation Rate Heatmap by Country & Year", height=max(400, len(heat_countries) * 30))
        st.plotly_chart(fig_heat, use_container_width=True)

with tab2c:
    # Side by side: inflation vs price change over time
    if len(selected_countries) > 0:
        comp_country = st.selectbox("Select country for deep dive", selected_countries, key="comp2")
        comp_data = inf_df[inf_df["country"] == comp_country].dropna(subset=["price_change_pct"])

        if len(comp_data) > 1:
            fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
            fig_comp.add_trace(go.Bar(
                x=comp_data["year"], y=comp_data["inflation"],
                name="CPI Inflation", marker_color="rgba(100,181,246,0.6)",
            ), secondary_y=False)
            fig_comp.add_trace(go.Scatter(
                x=comp_data["year"], y=comp_data["price_change_pct"],
                name="Big Mac Price Change", mode="lines+markers",
                line=dict(color=ACCENT, width=2.5), marker=dict(size=7),
            ), secondary_y=True)
            fig_comp = dark_layout(fig_comp, f"Inflation vs Big Mac Price Change — {comp_country}", height=450)
            fig_comp.update_yaxes(title_text="CPI Inflation (%)", secondary_y=False)
            fig_comp.update_yaxes(title_text="Big Mac Price Change (%)", secondary_y=True)
            st.plotly_chart(fig_comp, use_container_width=True)

with st.expander("📥 Download Objective 2 Data"):
    export2 = inf_df[["year", "country", "inflation", "cpi", "local_price", "price_change_pct"]].dropna()
    st.download_button("Download CSV", export2.to_csv(index=False), "obj2_inflation.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 3: REAL PURCHASING POWER COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Objective 3 — Real Purchasing Power Comparison</div>
<div class="section-sub">
    <span class="badge badge-desc">Descriptive</span>
    <span class="badge badge-diag">Diagnostic</span>
    Compare real purchasing power using inflation-adjusted Big Mac prices
</div>
""", unsafe_allow_html=True)

real_df = fdf.dropna(subset=["real_usd_price"]).copy()

tab3a, tab3b, tab3c = st.tabs(["🏆 Rankings", "🍩 Regional Drill-Down", "📈 Trend"])

with tab3a:
    rank_year = st.select_slider("Year", options=sorted(real_df["year"].unique()), value=real_df["year"].max(), key="rank_yr3")
    rank_data = real_df[real_df["year"] == rank_year].sort_values("real_usd_price", ascending=True)

    fig_rank = go.Figure(go.Bar(
        x=rank_data["real_usd_price"], y=rank_data["country"], orientation="h",
        marker=dict(
            color=rank_data["real_usd_price"],
            colorscale=[[0, "#E57373"], [0.5, "#FFB74D"], [1, "#81C784"]],
            colorbar=dict(title="Real USD"),
        ),
        text=rank_data["real_usd_price"].apply(lambda x: f"${x:.2f}"),
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>Real Price: $%{x:.2f}<extra></extra>",
    ))
    fig_rank = dark_layout(fig_rank, f"Inflation-Adjusted Big Mac Price Ranking — {rank_year}", height=max(600, len(rank_data) * 18))
    fig_rank.update_layout(xaxis_title="Real Big Mac Price (USD, Inflation-Adjusted)")
    st.plotly_chart(fig_rank, use_container_width=True)

with tab3b:
    sun_year = st.select_slider("Year", options=sorted(real_df["year"].unique()), value=real_df["year"].max(), key="sun_yr3")
    sun_data = real_df[real_df["year"] == sun_year].copy()
    sun_data["purchasing_power"] = 1 / sun_data["real_usd_price"]  # higher = more purchasing power
    sun_data["pp_label"] = sun_data["purchasing_power"].apply(lambda x: f"{x:.3f}")

    fig_sun = px.sunburst(
        sun_data, path=["region", "country"], values="purchasing_power",
        color="real_usd_price",
        color_continuous_scale=[[0, "#81C784"], [0.5, "#FFB74D"], [1, "#E57373"]],
        hover_data={"real_usd_price": ":.2f"},
    )
    fig_sun = dark_layout(fig_sun, f"Purchasing Power by Region → Country ({sun_year})", height=550)
    st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📊 Descriptive:</strong> Countries with <em>lower</em> inflation-adjusted Big Mac prices have higher relative
        purchasing power — meaning a dollar buys more locally. The sunburst shows purchasing power (inverse of real price)
        grouped by region, letting you drill down from region-level patterns to individual countries.<br>
        <strong>🔍 Diagnostic:</strong> Purchasing power disparities persist due to the Balassa-Samuelson effect:
        countries with lower productivity in non-tradables (services) have lower overall price levels, making
        Big Macs structurally cheaper even after inflation adjustment. Wage arbitrage across borders is the root driver.
    </div>
    """, unsafe_allow_html=True)

with tab3c:
    if selected_countries:
        fig_pp = go.Figure()
        for i, c in enumerate(selected_countries):
            cdata = real_df[real_df["country"] == c]
            fig_pp.add_trace(go.Scatter(
                x=cdata["year"], y=cdata["real_usd_price"],
                name=c, mode="lines+markers",
                line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                marker=dict(size=5),
            ))
        fig_pp = dark_layout(fig_pp, "Inflation-Adjusted Big Mac Price Trends", height=480)
        fig_pp.update_layout(xaxis_title="Year", yaxis_title="Real Big Mac Price (USD)")
        st.plotly_chart(fig_pp, use_container_width=True)

with st.expander("📥 Download Objective 3 Data"):
    export3 = real_df[["year", "country", "region", "real_usd_price", "real_local_price", "cpi"]].copy()
    st.download_button("Download CSV", export3.to_csv(index=False), "obj3_purchasing_power.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 4: PRICE CONVERGENCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Objective 4 — Big Mac Price Convergence Testing</div>
<div class="section-sub">
    <span class="badge badge-desc">Descriptive</span>
    <span class="badge badge-diag">Diagnostic</span>
    <span class="badge badge-pred">Predictive</span>
    Examine long-term price convergence in the global market
</div>
""", unsafe_allow_html=True)

tab4a, tab4b, tab4c = st.tabs(["📉 Sigma Convergence", "📈 Beta Convergence", "🌐 Dispersion Analysis"])

with tab4a:
    # Sigma convergence: coefficient of variation over time
    sigma_data = fdf.groupby("year").agg(
        mean_usd=("usd_price", "mean"),
        std_usd=("usd_price", "std"),
        count=("usd_price", "count"),
    ).reset_index()
    sigma_data["cv"] = sigma_data["std_usd"] / sigma_data["mean_usd"] * 100
    sigma_data["range"] = fdf.groupby("year")["usd_price"].apply(lambda x: x.max() - x.min()).values

    fig_sigma = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sigma.add_trace(go.Scatter(
        x=sigma_data["year"], y=sigma_data["cv"],
        name="Coefficient of Variation (%)", mode="lines+markers",
        line=dict(color=ACCENT, width=3), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(255,183,77,0.1)",
    ), secondary_y=False)
    fig_sigma.add_trace(go.Bar(
        x=sigma_data["year"], y=sigma_data["range"],
        name="Price Range (Max - Min USD)", marker_color="rgba(100,181,246,0.4)",
    ), secondary_y=True)
    fig_sigma = dark_layout(fig_sigma, "Sigma Convergence: Price Dispersion Over Time", height=480)
    fig_sigma.update_yaxes(title_text="CV (%)", secondary_y=False)
    fig_sigma.update_yaxes(title_text="Price Range (USD)", secondary_y=True)
    st.plotly_chart(fig_sigma, use_container_width=True)

    # Trend analysis
    slope, intercept, r_val, p_val, std_err = stats.linregress(sigma_data["year"], sigma_data["cv"])
    convergence = "converging (sigma convergence confirmed)" if slope < 0 else "diverging (no sigma convergence)"
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 Descriptive:</strong> The coefficient of variation (CV) measures price dispersion. A declining CV indicates
        sigma convergence — prices are becoming more similar globally.<br>
        <strong>🔍 Diagnostic:</strong> CV trend slope = <strong>{slope:.3f}</strong> per year (p = {p_val:.4f}).
        Big Mac prices are <strong>{convergence}</strong>. {"Globalization, supply chain standardization, and increasing economic integration have narrowed price gaps." if slope < 0 else "Persistent structural differences in input costs, exchange rate volatility, and diverging inflation rates prevent price equalization."}
    </div>
    """, unsafe_allow_html=True)

with tab4b:
    # Beta convergence: initial price vs subsequent growth
    initial_year = fdf["year"].min()
    final_year = fdf["year"].max()
    initial_prices = fdf[fdf["year"] == initial_year][["country", "usd_price"]].rename(columns={"usd_price": "initial_price"})
    final_prices = fdf[fdf["year"] == final_year][["country", "usd_price"]].rename(columns={"usd_price": "final_price"})
    beta_df = initial_prices.merge(final_prices, on="country")
    beta_df["growth_rate"] = ((beta_df["final_price"] / beta_df["initial_price"]) - 1) * 100
    beta_df = beta_df.merge(fdf[["country", "region"]].drop_duplicates(), on="country")

    fig_beta = px.scatter(
        beta_df, x="initial_price", y="growth_rate",
        color="region", hover_name="country",
        color_discrete_sequence=COLOR_PALETTE,
        labels={"initial_price": f"Initial USD Price ({initial_year})", "growth_rate": f"Price Growth {initial_year}–{final_year} (%)"},
        trendline="ols",
    )
    fig_beta = dark_layout(fig_beta, f"Beta Convergence: Initial Price vs Growth ({initial_year}–{final_year})", height=500)
    st.plotly_chart(fig_beta, use_container_width=True)

    if len(beta_df) > 2:
        r_b, p_b = stats.pearsonr(beta_df["initial_price"], beta_df["growth_rate"])
        st.markdown(f"""
        <div class="insight-box">
            <strong>📊 Descriptive:</strong> Beta convergence tests whether initially cheaper countries see faster price growth
            (catch-up effect). Correlation: <strong>r = {r_b:.3f}</strong> (p = {p_b:.4f}).<br>
            <strong>🔍 Diagnostic:</strong> {"Negative correlation confirms beta convergence — cheaper countries are catching up, consistent with economic development and integration." if r_b < 0 else "No beta convergence detected — initially cheap countries are not systematically catching up, suggesting persistent structural barriers to price equalization."}
        </div>
        """, unsafe_allow_html=True)

with tab4c:
    # Regional dispersion analysis
    reg_disp = fdf.groupby(["year", "region"]).agg(cv=("usd_price", lambda x: x.std() / x.mean() * 100)).reset_index()

    fig_disp = px.line(
        reg_disp, x="year", y="cv", color="region",
        color_discrete_sequence=COLOR_PALETTE,
        labels={"cv": "Coefficient of Variation (%)", "year": "Year"},
    )
    fig_disp = dark_layout(fig_disp, "Intra-Regional Price Dispersion Over Time", height=480)
    st.plotly_chart(fig_disp, use_container_width=True)

with st.expander("📥 Download Objective 4 Data"):
    st.download_button("Download Sigma Data", sigma_data.to_csv(index=False), "obj4_sigma_convergence.csv", "text/csv")
    st.download_button("Download Beta Data", beta_df.to_csv(index=False), "obj4_beta_convergence.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 5: GDP IMPACT ON ADJUSTED INDEX
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Objective 5 — GDP Impact on Adjusted Big Mac Index</div>
<div class="section-sub">
    <span class="badge badge-desc">Descriptive</span>
    <span class="badge badge-diag">Diagnostic</span>
    <span class="badge badge-pred">Predictive</span>
    Evaluate GDP per capita's influence on currency valuation
</div>
""", unsafe_allow_html=True)

gdp_df = fdf.dropna(subset=["gdp_per_capita", "adj_index_usd"]).copy()

tab5a, tab5b, tab5c = st.tabs(["🫧 Bubble Chart", "📈 Regression Deep-Dive", "🍩 Income Group Donut"])

with tab5a:
    bubble_year = st.select_slider("Year", options=sorted(gdp_df["year"].unique()), value=gdp_df["year"].max(), key="bub_yr5")
    bub_data = gdp_df[gdp_df["year"] == bubble_year]

    fig_bub = px.scatter(
        bub_data, x="gdp_per_capita", y=adj_col,
        size="usd_price", color="region", hover_name="country",
        size_max=30, color_discrete_sequence=COLOR_PALETTE,
        labels={"gdp_per_capita": "GDP per Capita (USD)", adj_col: f"GDP-Adj Misvaluation vs {base_currency}"},
        log_x=True,
    )
    fig_bub.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig_bub = dark_layout(fig_bub, f"GDP per Capita vs GDP-Adjusted Misvaluation — {bubble_year}", height=520)
    st.plotly_chart(fig_bub, use_container_width=True)

with tab5b:
    # Full regression: GDP per capita vs raw misvaluation
    reg_data = gdp_df.dropna(subset=["gdp_per_capita", raw_col])
    if len(reg_data) > 10:
        X = np.log(reg_data["gdp_per_capita"].values).reshape(-1, 1)
        y = reg_data[raw_col].values

        model = sm.OLS(y, sm.add_constant(X)).fit()

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=reg_data["gdp_per_capita"], y=reg_data[raw_col] * 100,
            mode="markers", marker=dict(color=ACCENT3, size=5, opacity=0.4),
            name="All Observations",
            hovertemplate="GDP: $%{x:,.0f}<br>Misval: %{y:.1f}%<extra></extra>",
        ))

        x_pred = np.linspace(X.min(), X.max(), 100)
        y_pred = model.predict(sm.add_constant(x_pred))
        fig_reg.add_trace(go.Scatter(
            x=np.exp(x_pred), y=y_pred * 100,
            mode="lines", line=dict(color=ACCENT, width=3),
            name=f"OLS (R² = {model.rsquared:.3f})",
        ))
        fig_reg = dark_layout(fig_reg, "GDP per Capita (log) vs Raw Misvaluation — All Years", height=480)
        fig_reg.update_layout(xaxis_title="GDP per Capita (USD, log scale)", yaxis_title=f"Raw Misvaluation vs {base_currency} (%)", xaxis_type="log")
        st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
            <strong>📊 Descriptive:</strong> R² = <strong>{model.rsquared:.3f}</strong> — GDP per capita explains {model.rsquared*100:.1f}% of the
            variance in raw currency misvaluation. Coefficient = {model.params[1]:.4f} (p = {model.pvalues[1]:.4f}).<br>
            <strong>🔍 Diagnostic:</strong> The positive relationship confirms the Balassa-Samuelson hypothesis: richer countries
            have structurally higher price levels. The GDP-adjusted index corrects for this, providing a fairer benchmark for
            currency valuation. The remaining unexplained variance comes from trade barriers, subsidies, and local competitive dynamics.<br>
            <strong>🔮 Predictive:</strong> As developing nations' GDP per capita rises, we expect their Big Mac prices to increase
            and raw undervaluation to shrink — a "convergence toward fair value" trajectory.
        </div>
        """, unsafe_allow_html=True)

with tab5c:
    inc_year = st.select_slider("Year", options=sorted(gdp_df["year"].unique()), value=gdp_df["year"].max(), key="inc_yr5")
    inc_data = gdp_df[gdp_df["year"] == inc_year]

    inc_agg = inc_data.groupby("income_group").agg(
        avg_raw=(raw_col, "mean"),
        avg_adj=(adj_col, "mean"),
        count=("country", "count"),
    ).reset_index()

    fig_inc = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                            subplot_titles=[f"Raw Misvaluation by Income Group", f"GDP-Adj Misvaluation by Income Group"])
    fig_inc.add_trace(go.Pie(
        labels=inc_agg["income_group"], values=inc_agg["count"],
        marker=dict(colors=["#E57373", "#FFB74D", "#81C784", "#64B5F6"]),
        hole=0.5, textinfo="label+percent",
        customdata=inc_agg["avg_raw"] * 100,
        hovertemplate="<b>%{label}</b><br>Countries: %{value}<br>Avg Misval: %{customdata:.1f}%<extra></extra>",
    ), 1, 1)
    fig_inc.add_trace(go.Pie(
        labels=inc_agg["income_group"], values=inc_agg["count"],
        marker=dict(colors=["#E57373", "#FFB74D", "#81C784", "#64B5F6"]),
        hole=0.5, textinfo="label+percent",
        customdata=inc_agg["avg_adj"] * 100,
        hovertemplate="<b>%{label}</b><br>Countries: %{value}<br>Avg GDP-Adj Misval: %{customdata:.1f}%<extra></extra>",
    ), 1, 2)
    fig_inc = dark_layout(fig_inc, f"Income Group Distribution & Misvaluation — {inc_year}", height=450)
    st.plotly_chart(fig_inc, use_container_width=True)

with st.expander("📥 Download Objective 5 Data"):
    export5 = gdp_df[["year", "country", "gdp_per_capita", "income_group", raw_col, adj_col, "usd_price"]].copy()
    st.download_button("Download CSV", export5.to_csv(index=False), "obj5_gdp_impact.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 6: EXCHANGE RATE INFLUENCE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Objective 6 — Exchange Rate Influence on USD Big Mac Price</div>
<div class="section-sub">
    <span class="badge badge-desc">Descriptive</span>
    <span class="badge badge-diag">Diagnostic</span>
    Investigate FX movements' impact on USD-converted Big Mac prices
</div>
""", unsafe_allow_html=True)

fx_df = fdf[fdf["country"] != "United States"].copy()
fx_df = fx_df.sort_values(["country", "year"])
fx_df["fx_change_pct"] = fx_df.groupby("country")["exchange_rate"].pct_change() * 100
fx_df["usd_price_change_pct"] = fx_df.groupby("country")["usd_price"].pct_change() * 100

tab6a, tab6b, tab6c = st.tabs(["📈 Correlation Scatter", "🔥 Correlation Heatmap", "📊 Country Deep-Dive"])

with tab6a:
    fx_scatter = fx_df.dropna(subset=["fx_change_pct", "usd_price_change_pct"])
    # Remove extreme outliers for cleaner visualization
    q_low = fx_scatter["fx_change_pct"].quantile(0.02)
    q_high = fx_scatter["fx_change_pct"].quantile(0.98)
    fx_scatter = fx_scatter[(fx_scatter["fx_change_pct"] >= q_low) & (fx_scatter["fx_change_pct"] <= q_high)]

    fig_fx = px.scatter(
        fx_scatter, x="fx_change_pct", y="usd_price_change_pct",
        color="region", hover_name="country",
        color_discrete_sequence=COLOR_PALETTE,
        labels={"fx_change_pct": "Exchange Rate Change (%)", "usd_price_change_pct": "USD Big Mac Price Change (%)"},
        opacity=0.5,
        trendline="ols",
    )
    fig_fx = dark_layout(fig_fx, "Exchange Rate Change vs USD Price Change (All Countries, All Years)", height=500)
    st.plotly_chart(fig_fx, use_container_width=True)

    valid_fx = fx_scatter.dropna(subset=["fx_change_pct", "usd_price_change_pct"])
    if len(valid_fx) > 5:
        r_fx, p_fx = stats.pearsonr(valid_fx["fx_change_pct"], valid_fx["usd_price_change_pct"])
        st.markdown(f"""
        <div class="insight-box">
            <strong>📊 Descriptive:</strong> Correlation between exchange rate changes and USD Big Mac price changes:
            <strong>r = {r_fx:.3f}</strong> (p = {p_fx:.2e}). A depreciating local currency (positive FX change) generally
            leads to a <em>decrease</em> in the USD-converted Big Mac price.<br>
            <strong>🔍 Diagnostic:</strong> Exchange rate movements dominate short-term USD price changes because local
            Big Mac prices adjust slowly (menu price stickiness), while FX rates move daily. Countries with
            pegged or managed currencies (GCC states, Hong Kong) show minimal FX-driven price changes,
            while floating-rate economies (Turkey, Argentina) exhibit the strongest FX pass-through.
        </div>
        """, unsafe_allow_html=True)

with tab6b:
    # Per-country correlation between FX change and USD price change
    corr_list = []
    for c in fx_df["country"].unique():
        cdata = fx_df[fx_df["country"] == c].dropna(subset=["fx_change_pct", "usd_price_change_pct"])
        if len(cdata) > 3:
            r, _ = stats.pearsonr(cdata["fx_change_pct"], cdata["usd_price_change_pct"])
            corr_list.append({"country": c, "correlation": r})
    corr_df = pd.DataFrame(corr_list).sort_values("correlation")

    fig_corr_bar = go.Figure(go.Bar(
        x=corr_df["correlation"], y=corr_df["country"], orientation="h",
        marker=dict(
            color=corr_df["correlation"],
            colorscale=[[0, "#E57373"], [0.5, "#FAFAFA"], [1, "#64B5F6"]],
            cmid=0,
        ),
        text=corr_df["correlation"].apply(lambda x: f"{x:.2f}"),
        textposition="outside", textfont=dict(size=9),
    ))
    fig_corr_bar = dark_layout(fig_corr_bar, "FX-to-USD Price Correlation by Country", height=max(500, len(corr_df) * 16))
    fig_corr_bar.update_layout(xaxis_title="Pearson Correlation")
    fig_corr_bar.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    st.plotly_chart(fig_corr_bar, use_container_width=True)

with tab6c:
    fx_countries = [c for c in selected_countries if c != "United States"]
    if fx_countries:
        fx_deep_country = st.selectbox("Select country", fx_countries, key="fx_deep6")
        fx_deep = fx_df[fx_df["country"] == fx_deep_country]

        fig_fxd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fxd.add_trace(go.Scatter(
            x=fx_deep["year"], y=fx_deep["exchange_rate"],
            name="Exchange Rate (local/USD)", mode="lines+markers",
            line=dict(color=ACCENT3, width=2), marker=dict(size=5),
        ), secondary_y=False)
        fig_fxd.add_trace(go.Scatter(
            x=fx_deep["year"], y=fx_deep["usd_price"],
            name="Big Mac USD Price", mode="lines+markers",
            line=dict(color=ACCENT, width=2.5), marker=dict(size=6),
        ), secondary_y=True)
        fig_fxd = dark_layout(fig_fxd, f"Exchange Rate & USD Big Mac Price — {fx_deep_country}", height=450)
        fig_fxd.update_yaxes(title_text="Exchange Rate", secondary_y=False)
        fig_fxd.update_yaxes(title_text="USD Price", secondary_y=True)
        st.plotly_chart(fig_fxd, use_container_width=True)

with st.expander("📥 Download Objective 6 Data"):
    export6 = fx_df[["year", "country", "exchange_rate", "usd_price", "fx_change_pct", "usd_price_change_pct"]].dropna()
    st.download_button("Download CSV", export6.to_csv(index=False), "obj6_fx_influence.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 7: RAW VS GDP-ADJUSTED MISVALUATION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Objective 7 — Raw vs GDP-Adjusted Misvaluation</div>
<div class="section-sub">
    <span class="badge badge-desc">Descriptive</span>
    <span class="badge badge-diag">Diagnostic</span>
    <span class="badge badge-pres">Prescriptive</span>
    Compare raw vs GDP-adjusted PPP misvaluation to identify structural economic differences
</div>
""", unsafe_allow_html=True)

adj_df = fdf.dropna(subset=[raw_col, adj_col]).copy()
adj_df["divergence"] = (adj_df[adj_col] - adj_df[raw_col]) * 100

tab7a, tab7b, tab7c, tab7d = st.tabs(["📊 Side-by-Side", "🍩 Drill-Down Donut", "📈 Divergence Trend", "🔍 Scatter Comparison"])

with tab7a:
    comp7_year = st.select_slider("Year", options=sorted(adj_df["year"].unique()), value=adj_df["year"].max(), key="comp7_yr")
    comp7 = adj_df[adj_df["year"] == comp7_year].sort_values(raw_col)

    fig_comp7 = go.Figure()
    fig_comp7.add_trace(go.Bar(
        y=comp7["country"], x=comp7[raw_col] * 100,
        name="Raw Index", orientation="h",
        marker_color="rgba(255,183,77,0.7)",
    ))
    fig_comp7.add_trace(go.Bar(
        y=comp7["country"], x=comp7[adj_col] * 100,
        name="GDP-Adjusted Index", orientation="h",
        marker_color="rgba(100,181,246,0.7)",
    ))
    fig_comp7 = dark_layout(fig_comp7, f"Raw vs GDP-Adjusted Misvaluation — {comp7_year}", height=max(600, len(comp7) * 20))
    fig_comp7.update_layout(barmode="group", xaxis_title=f"Misvaluation vs {base_currency} (%)")
    fig_comp7.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    st.plotly_chart(fig_comp7, use_container_width=True)

with tab7b:
    donut7_year = st.select_slider("Year", options=sorted(adj_df["year"].unique()), value=adj_df["year"].max(), key="donut7_yr")
    d7_data = adj_df[adj_df["year"] == donut7_year].copy()
    d7_data["adj_status"] = d7_data[adj_col].apply(lambda x: "Overvalued (GDP-Adj)" if x > 0 else "Undervalued (GDP-Adj)")
    d7_data["raw_status"] = d7_data[raw_col].apply(lambda x: "Overvalued (Raw)" if x > 0 else "Undervalued (Raw)")
    d7_data["abs_div"] = d7_data["divergence"].abs()

    # Sunburst: status → region → country
    fig_sun7 = px.sunburst(
        d7_data, path=["adj_status", "region", "country"],
        values="abs_div",
        color="divergence",
        color_continuous_scale=[[0, "#E57373"], [0.5, "#FAFAFA"], [1, "#64B5F6"]],
        color_continuous_midpoint=0,
    )
    fig_sun7 = dark_layout(fig_sun7, f"GDP Adjustment Impact: Divergence Drill-Down ({donut7_year})", height=550)
    st.plotly_chart(fig_sun7, use_container_width=True)

    # Flipped countries
    flipped = d7_data[(d7_data[raw_col] * d7_data[adj_col]) < 0]
    if len(flipped) > 0:
        st.markdown(f"""
        <div class="insight-box">
            <strong>🔍 Key Finding:</strong> <strong>{len(flipped)} countries</strong> flip their valuation status after GDP adjustment
            in {donut7_year}: <strong>{', '.join(flipped['country'].tolist())}</strong>. These are countries where the GDP-adjustment
            reveals that what appeared to be currency misvaluation is actually explained by their income level —
            a crucial distinction for policymakers and investors.
        </div>
        """, unsafe_allow_html=True)

with tab7c:
    if selected_countries:
        fig_div = go.Figure()
        for i, c in enumerate(selected_countries):
            cdata = adj_df[adj_df["country"] == c]
            fig_div.add_trace(go.Scatter(
                x=cdata["year"], y=cdata["divergence"],
                name=c, mode="lines+markers",
                line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                marker=dict(size=5),
            ))
        fig_div.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_div = dark_layout(fig_div, "Raw vs GDP-Adjusted Divergence Over Time", height=480)
        fig_div.update_layout(xaxis_title="Year", yaxis_title="Divergence (GDP-Adj − Raw) in pp")
        st.plotly_chart(fig_div, use_container_width=True)

with tab7d:
    sc7_year = st.select_slider("Year", options=sorted(adj_df["year"].unique()), value=adj_df["year"].max(), key="sc7_yr")
    sc7_data = adj_df[adj_df["year"] == sc7_year]

    fig_sc7 = px.scatter(
        sc7_data, x=raw_col, y=adj_col,
        color="region", hover_name="country",
        color_discrete_sequence=COLOR_PALETTE,
        labels={raw_col: f"Raw Misvaluation vs {base_currency}", adj_col: f"GDP-Adj Misvaluation vs {base_currency}"},
        size="gdp_per_capita", size_max=25,
    )
    # Add 45-degree line
    lim = max(sc7_data[raw_col].abs().max(), sc7_data[adj_col].abs().max()) * 1.1
    fig_sc7.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim],
        mode="lines", line=dict(dash="dot", color="rgba(255,255,255,0.2)"),
        name="No Adjustment Effect", showlegend=True,
    ))
    fig_sc7.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig_sc7.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig_sc7 = dark_layout(fig_sc7, f"Raw vs GDP-Adjusted Misvaluation — {sc7_year}", height=520)
    st.plotly_chart(fig_sc7, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 Descriptive:</strong> Points above the 45° line = GDP adjustment makes the currency appear <em>more</em> overvalued
        (or less undervalued). Points below = adjustment reduces perceived overvaluation. Points far from the line = large GDP adjustment effect.<br>
        <strong>🔍 Diagnostic:</strong> Developing countries (India, Vietnam, Indonesia) cluster in the "below the line" zone — their
        raw undervaluation is partially or fully explained by lower GDP per capita, not currency manipulation. Wealthy outliers
        (Switzerland, Norway) remain above the line, confirming genuine high-cost structures.<br>
        <strong>💡 Prescriptive:</strong> For currency policy analysis, always prefer GDP-adjusted indices when evaluating
        emerging markets — raw indices systematically overstate undervaluation for low-income countries.
        For developed economies, the raw and adjusted indices largely agree, making either suitable.
    </div>
    """, unsafe_allow_html=True)

with st.expander("📥 Download Objective 7 Data"):
    export7 = adj_df[["year", "country", "region", raw_col, adj_col, "divergence", "gdp_per_capita", "income_group"]].copy()
    st.download_button("Download CSV", export7.to_csv(index=False), "obj7_raw_vs_adjusted.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# BONUS: PREDICTIVE & PRESCRIPTIVE SECTION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">Bonus — Predictive & Prescriptive Analytics</div>
<div class="section-sub">
    <span class="badge badge-pred">Predictive</span>
    <span class="badge badge-pres">Prescriptive</span>
    Trend forecasting & policy implications
</div>
""", unsafe_allow_html=True)

tab_pa, tab_pb = st.tabs(["🔮 Price Trend Forecast", "💡 Policy Prescriptions"])

with tab_pa:
    pred_country = st.selectbox("Select country for forecast", all_countries, index=all_countries.index("India") if "India" in all_countries else 0, key="pred_c")
    pred_data = fdf[fdf["country"] == pred_country].sort_values("year")

    if len(pred_data) > 5:
        X_p = pred_data["year"].values.reshape(-1, 1)
        y_p = pred_data["usd_price"].values

        # Linear trend
        lr = LinearRegression().fit(X_p, y_p)
        # Polynomial trend
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_p)
        pr = LinearRegression().fit(X_poly, y_p)

        future_years = np.arange(pred_data["year"].max() + 1, pred_data["year"].max() + 6).reshape(-1, 1)
        all_years = np.concatenate([X_p, future_years])

        y_lr = lr.predict(all_years)
        y_pr = pr.predict(poly.transform(all_years))

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=pred_data["year"], y=pred_data["usd_price"],
            name="Actual", mode="lines+markers",
            line=dict(color=ACCENT, width=2.5), marker=dict(size=7),
        ))
        fig_pred.add_trace(go.Scatter(
            x=all_years.flatten(), y=y_lr,
            name=f"Linear Trend (R²={lr.score(X_p, y_p):.3f})",
            mode="lines", line=dict(color=ACCENT3, width=2, dash="dash"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=all_years.flatten(), y=y_pr,
            name=f"Polynomial Trend (R²={pr.score(X_poly, y_p):.3f})",
            mode="lines", line=dict(color=ACCENT5, width=2, dash="dot"),
        ))
        fig_pred.add_vrect(
            x0=pred_data["year"].max() + 0.5, x1=future_years.max() + 0.5,
            fillcolor="rgba(255,183,77,0.08)", line_width=0,
            annotation_text="Forecast", annotation_position="top",
        )
        fig_pred = dark_layout(fig_pred, f"Big Mac USD Price Forecast — {pred_country}", height=480)
        fig_pred.update_layout(xaxis_title="Year", yaxis_title="USD Price")
        st.plotly_chart(fig_pred, use_container_width=True)

        # CAGR
        if pred_data["usd_price"].iloc[0] > 0:
            n_years = pred_data["year"].max() - pred_data["year"].min()
            cagr = ((pred_data["usd_price"].iloc[-1] / pred_data["usd_price"].iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
            st.markdown(f"""
            <div class="insight-box">
                <strong>🔮 Predictive:</strong> {pred_country}'s Big Mac USD price has a CAGR of <strong>{cagr:.2f}%</strong> over
                {n_years} years. Linear forecast projects ~${y_lr[-1]:.2f} by {int(future_years[-1][0])}.
                Polynomial model captures acceleration/deceleration effects, projecting ~${y_pr[-1]:.2f}.<br>
                <em>Caveat: These are simple trend extrapolations and don't account for structural breaks, policy changes, or black swan events.</em>
            </div>
            """, unsafe_allow_html=True)

with tab_pb:
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Prescriptive Insights from the Big Mac Index Analysis:</strong><br><br>
        <strong>1. For Central Banks & Currency Policy:</strong> Countries with persistent GDP-adjusted overvaluation
        should monitor competitiveness risks. The gap between raw and adjusted indices can signal whether
        currency intervention is addressing real misalignment or merely structural price differences.<br><br>
        <strong>2. For International Investors:</strong> Use the GDP-adjusted index as a long-term mean-reversion signal.
        Currencies that are undervalued even after GDP adjustment (rare) represent stronger value opportunities
        than those that are only raw-undervalued.<br><br>
        <strong>3. For Trade Policy:</strong> Countries showing convergence in Big Mac prices (declining CV) are
        becoming more economically integrated. This can guide trade agreement priorities and tariff negotiations.<br><br>
        <strong>4. For Inflation Management:</strong> Countries where Big Mac price growth consistently exceeds CPI
        should investigate food-specific inflation drivers — agricultural policy, import dependencies, or
        supply chain inefficiencies in the fast-food sector.<br><br>
        <strong>5. For Development Economics:</strong> The Balassa-Samuelson relationship (GDP vs. price level) provides
        a benchmark for expected price levels. Countries deviating significantly from this line may have
        distorted price structures worth investigating.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1.5rem; color: #556677; font-family: 'Outfit', sans-serif; font-size: 0.85rem;">
    🍔 Big Mac Index Analytics Dashboard — SP Jain School of Global Management<br>
    MBA (Global) — Applied Research Project<br>
    Data Source: The Economist's Big Mac Index (2001–2025) | World Bank CPI & GDP Data<br>
    <span style="color:#FFB74D;">Descriptive</span> · <span style="color:#64B5F6;">Diagnostic</span> ·
    <span style="color:#BA68C8;">Predictive</span> · <span style="color:#FFB74D;">Prescriptive</span>
</div>
""", unsafe_allow_html=True)
