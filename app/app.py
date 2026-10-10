import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error, r2_score
)

# ==================================================
# 1. PAGE SETUP AND LOOK
# ==================================================
st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Professional palette: blue, teal, amber, violet, red, green
BLUE = "#4C8DFF"
GREEN = "#2DD4BF"
AMBER = "#F5A524"
VIOLET = "#8B7CF6"
PALETTE = [BLUE, GREEN, AMBER, VIOLET, "#F0616D", "#34C38F"]

st.markdown(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0E1117;
  --panel: #151B27;
  --panel2: #1A2231;
  --line: #263043;
  --txt: #E5E9F0;
  --mut: #8B97A9;
  --blue: #4C8DFF;
  --teal: #2DD4BF;
  --amber: #F5A524;
}

/* ---------- Page ---------- */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse at 50% -10%, rgba(76,141,255,0.10), transparent 55%);
}
header[data-testid="stHeader"] {background: transparent;}
.block-container, div[data-testid="stMainBlockContainer"] {
    padding-top: 1.4rem !important; max-width: 1400px;
}
::selection {background: rgba(76,141,255,0.45);}
::-webkit-scrollbar {width: 10px; height: 10px;}
::-webkit-scrollbar-track {background: var(--bg);}
::-webkit-scrollbar-thumb {background: #2A3550; border-radius: 6px;}
::-webkit-scrollbar-thumb:hover {background: #3A4A6E;}
hr {border-color: var(--line) !important;}

/* ---------- Typography ---------- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-baseweb="select"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {color: var(--txt);}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
    font-weight: 600; letter-spacing: -0.01em;
}
/* Small blue bar before each section heading */
.stApp h2::before, .stApp h3::before {
    content: ""; display: inline-block; width: 4px; height: 1.05em;
    background: var(--blue); border-radius: 2px; margin-right: 10px;
    vertical-align: -0.15em;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0B0F17; border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    position: absolute; top: 0; right: 0; z-index: 5;
    height: auto; padding: 0.4rem; background: transparent;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0.8rem !important;
}

.sb-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 14px; margin-bottom: 14px; border-radius: 10px;
    background: var(--panel); border: 1px solid var(--line);
}
.sb-brand .logo {
    width: 38px; height: 38px; border-radius: 9px; display: flex;
    align-items: center; justify-content: center; font-size: 0.85rem;
    background: linear-gradient(135deg, #4C8DFF, #2DD4BF); color: #fff;
    font-weight: 700; letter-spacing: 0.02em;
}
.sb-brand .name {font-weight: 700; font-size: 0.9rem;
                 letter-spacing: 0.04em; color: var(--txt);}
.sb-brand .tag {font-size: 0.72rem; color: var(--mut);}

.sb-title {font-size: 0.78rem; font-weight: 600; color: var(--mut);
           text-transform: uppercase; letter-spacing: 0.1em;}
.sb-sub {color: var(--mut); font-size: 0.78rem; margin-bottom: 12px;}

span[data-baseweb="tag"] {
    background-color: var(--blue) !important; color: #fff !important;
    border-radius: 4px !important; font-size: 0.76rem; font-weight: 500;
}
span[data-baseweb="tag"] span {color: #fff !important;}
span[data-baseweb="tag"] svg {fill: #fff !important;}

section[data-testid="stSidebar"] details {
    background: var(--panel); border: 1px solid var(--line) !important;
    border-radius: 8px; margin-bottom: 10px;
}
section[data-testid="stSidebar"] summary {font-weight: 600;}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: var(--panel) !important; border-color: var(--line) !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
    border-color: #3A4A66 !important;
}
div[data-baseweb="input"] > div:focus-within, div[data-baseweb="base-input"]:focus-within {
    border-color: var(--blue) !important;
}

.sb-card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 14px 16px; margin-top: 8px;
}
.sb-card .big {font-size: 1.3rem; font-weight: 600; color: var(--txt);
               font-family: 'JetBrains Mono', Consolas, monospace;}
.sb-card .small {color: var(--mut); font-size: 0.74rem;}
.sb-bar {height: 6px; border-radius: 3px; background: #222C40;
         margin: 8px 0 6px 0; overflow: hidden;}
.sb-fill {height: 100%; border-radius: 3px;
          background: linear-gradient(90deg, var(--blue), var(--teal));}

/* ---------- Buttons ---------- */
button[data-testid="stBaseButton-primary"], button[kind="primary"],
button[kind="primaryFormSubmit"] {
    background: var(--blue) !important; color: #fff !important;
    border: none !important; font-weight: 600 !important;
    border-radius: 6px !important;
    transition: background 0.15s ease;
}
button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover {background: #3D7BEA !important;}
button[data-testid="stBaseButton-primary"] *, button[kind="primary"] *,
button[kind="primaryFormSubmit"] * {color: #fff !important;}
button[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: var(--panel) !important; border-radius: 6px !important;
    border: 1px solid var(--line) !important; color: var(--txt) !important;
}
button[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
    border-color: var(--blue) !important; color: #BBD3FF !important;
}

/* ---------- Navigation ---------- */
div[role="radiogroup"] {gap: 8px; flex-wrap: wrap;}
div[role="radiogroup"] > label {
    background: var(--panel); border: 1px solid var(--line);
    padding: 7px 16px; border-radius: 6px; cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
}
div[role="radiogroup"] > label:hover {
    border-color: #3A4A66; background: var(--panel2);
}
div[role="radiogroup"] > label > div:first-child {display: none;}
div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(76,141,255,0.16); border-color: var(--blue);
}
div[role="radiogroup"] > label:has(input:checked) * {
    color: #BBD3FF !important; font-weight: 600;
}

/* ---------- KPI and About cards ---------- */
.kpi, .card {
    position: relative; overflow: hidden;
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px;
}
.kpi {padding: 18px 18px 14px 18px;}
.card {padding: 18px 22px; margin-bottom: 14px;}
.kpi::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--blue), var(--teal));
}
.kpi:hover {border-color: #3A4A66;}
.kpi .label {color: var(--mut); font-size: 0.72rem; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.08em;}
.kpi .value {color: var(--txt); font-size: 1.65rem; font-weight: 600;
             margin-top: 2px;
             font-family: 'JetBrains Mono', Consolas, monospace;}
.kpi .note {color: var(--mut); font-size: 0.78rem;}
.card h4 {color: var(--txt) !important; margin-top: 0;}

/* Streamlit's own metric boxes */
div[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px 16px;
}

/* ---------- Explanation boxes ---------- */
.howto {
    background: rgba(76,141,255,0.07); border-left: 3px solid var(--blue);
    border-radius: 4px 8px 8px 4px; padding: 10px 14px; margin: 4px 0 8px 0;
    color: #B7C3D6; font-size: 0.88rem;
}
.insight {
    background: rgba(45,212,191,0.07); border-left: 3px solid var(--teal);
    border-radius: 4px 8px 8px 4px; padding: 10px 14px; margin: 6px 0 18px 0;
    color: #BDEFE8; font-size: 0.9rem;
}
.insight b {color: #E5E9F0;}
.result {
    background: linear-gradient(135deg, rgba(76,141,255,0.20), rgba(45,212,191,0.10));
    border: 1px solid rgba(76,141,255,0.55);
    padding: 20px 24px; border-radius: 10px;
}
.result div {color: #BBD3FF !important;}
.result .big {font-size: 2.1rem; font-weight: 600; color: #fff !important;
              font-family: 'JetBrains Mono', Consolas, monospace;}

/* ---------- Tables and alerts ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--line); border-radius: 8px;
}
div[data-testid="stAlert"] {border-radius: 8px;}
</style>
    """,
    unsafe_allow_html=True
)


def money(x):
    return f"₹{x:,.0f}"


def howto(text):
    st.markdown(f'<div class="howto">📖 <b>How to read:</b> {text}</div>',
                unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">👉 <b>Insight:</b> {text}</div>',
                unsafe_allow_html=True)


def kpi(label, value, note):
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="note">{note}</div></div>')


def clean(fig, height=380):
    """Same simple look for every chart."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif",
                  color="#C7D0DC", size=12),
        hoverlabel=dict(bgcolor="#1A2231", bordercolor="#4C8DFF",
                        font=dict(family="Inter, Segoe UI, sans-serif",
                                  color="#E5E9F0")),
        showlegend=False
    )
    fig.update_xaxes(showgrid=False, linecolor="#263043")
    fig.update_yaxes(gridcolor="#1E2638", linecolor="#263043")
    return fig


# ==================================================
# 2. PATHS AND LOADING FUNCTIONS
# ==================================================
PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
DATA_PATH = os.path.join(
    PROJECT_DIR, "data", "processed", "Apple_Business_Cleaned.csv"
)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

REVENUE_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_revenue.pkl")
REVENUE_FEATURES_PATH = os.path.join(MODELS_DIR, "revenue_features.pkl")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "random_forest_profit_classifier.pkl")
CLASSIFICATION_FEATURES_PATH = os.path.join(
    MODELS_DIR, "classification_features.pkl"
)

# The 17 inputs used when the models were trained
FEATURE_COLUMNS = [
    "Units_Sold", "Units_Produced", "Units_Returned", "Units_Damaged",
    "Inventory_Remaining", "Cost_Price", "Selling_Price", "Discount_Percent",
    "Marketing_Cost", "Transportation_Cost", "Customer_Segment", "Region",
    "Sales_Channel", "Demand_Level", "Product_Category", "Year", "Month"
]
TEXT_FEATURES = ["Customer_Segment", "Region", "Sales_Channel",
                 "Demand_Level", "Product_Category"]


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

# ==================================================
# 3. DATA VALIDATION AND PREPARATION
# ==================================================
required_columns = ["Product_Category", "Region", "Revenue", "Profit",
                    "Units_Sold"]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

for col in ["Revenue", "Profit", "Units_Sold"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

for col in ["Product_Category", "Region"]:
    df[col] = (
        df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )

has_dates = "Sales_Date" in df.columns
if has_dates:
    df["Sales_Date"] = pd.to_datetime(df["Sales_Date"], errors="coerce")
    has_dates = df["Sales_Date"].notna().any()


# ==================================================
# 4. PREDICTION AND MODEL HELPER FUNCTIONS
# ==================================================
def create_prediction_form(prefix, categories_list):
    """Collect the 17 input features used during training."""
    with st.form(f"{prefix}_form"):
        col1, col2 = st.columns(2)

        with col1:
            units_sold = st.number_input(
                "Units Sold", min_value=0, value=100, step=1,
                key=f"{prefix}_sold")
            units_produced = st.number_input(
                "Units Produced", min_value=0, value=120, step=1,
                key=f"{prefix}_produced")
            units_returned = st.number_input(
                "Units Returned", min_value=0, value=2, step=1,
                key=f"{prefix}_returned")
            units_damaged = st.number_input(
                "Units Damaged", min_value=0, value=1, step=1,
                key=f"{prefix}_damaged")
            inventory = st.number_input(
                "Inventory Remaining", min_value=0, value=20, step=1,
                key=f"{prefix}_inventory")
            cost_price = st.number_input(
                "Cost Price (₹)", min_value=0.0, value=50000.0,
                step=1000.0, key=f"{prefix}_cost")
            selling_price = st.number_input(
                "Selling Price (₹)", min_value=0.0, value=60000.0,
                step=1000.0, key=f"{prefix}_selling")
            discount = st.number_input(
                "Discount (%)", min_value=0.0, max_value=100.0,
                value=5.0, step=1.0, key=f"{prefix}_discount")

        with col2:
            marketing_cost = st.number_input(
                "Marketing Cost (₹)", min_value=0.0, value=10000.0,
                step=1000.0, key=f"{prefix}_marketing")
            transport_cost = st.number_input(
                "Transportation Cost (₹)", min_value=0.0, value=5000.0,
                step=500.0, key=f"{prefix}_transport")
            customer_segment = st.selectbox(
                "Customer Segment",
                ["Consumer", "Corporate", "Small Business"],
                key=f"{prefix}_segment")
            region = st.selectbox(
                "Region", ["Central", "East", "North", "South", "West"],
                key=f"{prefix}_region")
            sales_channel = st.selectbox(
                "Sales Channel", ["Online", "Retail", "Distributor"],
                key=f"{prefix}_channel")
            demand_level = st.selectbox(
                "Demand Level", ["Low", "Medium", "High"],
                key=f"{prefix}_demand")
            product_category = st.selectbox(
                "Product Category", categories_list,
                key=f"{prefix}_category")
            year = st.number_input(
                "Year", min_value=2000, max_value=2100, value=2025,
                step=1, key=f"{prefix}_year")
            month = st.selectbox(
                "Month", list(range(1, 13)), key=f"{prefix}_month")

        submitted = st.form_submit_button(
            "🚀 Run Prediction", type="primary", use_container_width=True
        )

    input_row = {
        "Units_Sold": units_sold,
        "Units_Produced": units_produced,
        "Units_Returned": units_returned,
        "Units_Damaged": units_damaged,
        "Inventory_Remaining": inventory,
        "Cost_Price": cost_price,
        "Selling_Price": selling_price,
        "Discount_Percent": discount,
        "Marketing_Cost": marketing_cost,
        "Transportation_Cost": transport_cost,
        "Customer_Segment": customer_segment,
        "Region": region,
        "Sales_Channel": sales_channel,
        "Demand_Level": demand_level,
        "Product_Category": product_category,
        "Year": year,
        "Month": month
    }
    return submitted, pd.DataFrame([input_row])


def encode_for_model(input_data, feature_columns):
    """Align one-hot encoded input to the saved training columns."""
    encoded = pd.get_dummies(input_data)
    return encoded.reindex(columns=list(feature_columns), fill_value=0)


def grouped_importance(model, feature_columns):
    """Add up one-hot columns (Region_East, Region_West...) per original input."""
    if not hasattr(model, "feature_importances_"):
        return None
    names = list(feature_columns)
    if len(names) != len(model.feature_importances_):
        return None

    totals = {}
    for name, value in zip(names, model.feature_importances_):
        group = name
        for text_col in TEXT_FEATURES:
            if name.startswith(text_col + "_"):
                group = text_col
                break
        totals[group] = totals.get(group, 0) + float(value)

    result = pd.Series(totals).sort_values(ascending=False)
    return result * 100


def prepare_check_data(path):
    """Load the dataset and make sure Year and Month columns exist."""
    data = load_data(path).copy()
    if "Sales_Date" in data.columns:
        dates = pd.to_datetime(data["Sales_Date"], errors="coerce")
        if "Year" not in data.columns:
            data["Year"] = dates.dt.year
        if "Month" not in data.columns:
            data["Month"] = dates.dt.month
    for col in ["Revenue", "Profit"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


@st.cache_data
def check_revenue_model(model_path, features_path, data_path):
    """R2 and MAE of the revenue model on a sample of the dataset."""
    model = load_model(model_path)
    features = load_model(features_path)
    data = prepare_check_data(data_path)

    needed = FEATURE_COLUMNS + ["Revenue"]
    if any(c not in data.columns for c in needed):
        return None

    data = data[needed].dropna()
    data = data.sample(min(len(data), 5000), random_state=42)

    x = encode_for_model(data[FEATURE_COLUMNS], features)
    predicted = model.predict(x)

    return {
        "r2": r2_score(data["Revenue"], predicted),
        "mae": mean_absolute_error(data["Revenue"], predicted),
        "rows": len(data)
    }


@st.cache_data
def check_classifier(model_path, features_path, data_path):
    """Accuracy and confusion matrix of the classifier on a sample."""
    model = load_model(model_path)
    features = load_model(features_path)
    data = prepare_check_data(data_path)

    needed = FEATURE_COLUMNS + ["Profit"]
    if any(c not in data.columns for c in needed):
        return None

    class_names = [str(c).strip().lower() for c in model.classes_]
    if sorted(class_names) != ["loss", "profit"]:
        return None

    data = data[needed].dropna()
    data = data.sample(min(len(data), 5000), random_state=42)

    # Assumption: a record is "profit" when its Profit value is above zero
    actual = np.where(data["Profit"] > 0, "profit", "loss")

    x = encode_for_model(data[FEATURE_COLUMNS], features)
    predicted = pd.Series(model.predict(x)).astype(str).str.strip().str.lower()

    labels = ["profit", "loss"]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Profit", "Actual Loss"],
        columns=["Predicted Profit", "Predicted Loss"]
    )

    return {
        "accuracy": accuracy_score(actual, predicted),
        "matrix": matrix_df,
        "rows": len(data)
    }


# ==================================================
# 5. SIDEBAR FILTERS
# ==================================================
st.sidebar.markdown(
    """
    <div class="sb-brand">
        <div class="logo">BA</div>
        <div>
            <div class="name">BUSINESS ANALYTICS</div>
            <div class="tag">Data Engineering · ML</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown('<div class="sb-title">🔎 Filters</div>'
                    '<div class="sb-sub">Applies to every page</div>',
                    unsafe_allow_html=True)

categories = sorted(df["Product_Category"].unique())
regions = sorted(df["Region"].unique())

# Reset button: clears the saved filter choices
if st.sidebar.button("🔄 Reset filters", use_container_width=True):
    for key in ["all_cats", "f_cat", "all_regs", "f_reg", "f_date"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Product category filter (tick "All" or pick a few)
with st.sidebar.expander("📦 Product Category", expanded=False):
    all_cats = st.checkbox(f"All categories ({len(categories)})",
                           value=True, key="all_cats")
    if all_cats:
        selected_categories = categories
    else:
        selected_categories = st.multiselect(
            "Choose categories", categories,
            default=categories[:3], key="f_cat"
        )

# Region filter (tick "All" or pick a few)
with st.sidebar.expander("🌍 Region", expanded=False):
    all_regs = st.checkbox(f"All regions ({len(regions)})",
                           value=True, key="all_regs")
    if all_regs:
        selected_regions = regions
    else:
        selected_regions = st.multiselect(
            "Choose regions", regions,
            default=regions[:2], key="f_reg"
        )

filtered = df[
    df["Product_Category"].isin(selected_categories)
    & df["Region"].isin(selected_regions)
].copy()

date_text = "All dates"
if has_dates:
    min_date = df["Sales_Date"].min().date()
    max_date = df["Sales_Date"].max().date()
    date_range = st.sidebar.date_input(
        "📅 Date Range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date, key="f_date"
    )
    # Only one date is returned while the user is still picking
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered[
            (filtered["Sales_Date"] >= pd.to_datetime(date_range[0]))
            & (filtered["Sales_Date"] <= pd.to_datetime(date_range[1]))
        ]
        date_text = f"{date_range[0]:%d %b %Y} to {date_range[1]:%d %b %Y}"

# Summary card: how much of the data is currently shown
share_shown = (len(filtered) / len(df) * 100) if len(df) else 0
cat_text = ("All" if len(selected_categories) == len(categories)
            else f"{len(selected_categories)} of {len(categories)}")
reg_text = ("All" if len(selected_regions) == len(regions)
            else f"{len(selected_regions)} of {len(regions)}")

st.sidebar.markdown(
    f"""
    <div class="sb-card">
        <div class="small">RECORDS IN VIEW</div>
        <div class="big">{len(filtered):,}
            <span class="small">of {len(df):,}</span></div>
        <div class="sb-bar"><div class="sb-fill"
             style="width:{share_shown:.0f}%"></div></div>
        <div class="small">Categories: {cat_text} &nbsp;·&nbsp;
             Regions: {reg_text}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# 6. HEADER AND NAVIGATION
# ==================================================
# ---------- Project header with the data pipeline ----------
models_found = sum(os.path.exists(p) for p in
                   [REVENUE_MODEL_PATH, CLASSIFIER_PATH])
model_state = "" if models_found == 2 else "warn"

period_text = "n/a"
if has_dates:
    period_text = (f"{df['Sales_Date'].min():%b %Y} - "
                   f"{df['Sales_Date'].max():%b %Y}")

BANNER_HTML = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
  html, body {margin: 0; background: transparent;}
  .hdr {font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        background: linear-gradient(135deg, #171E2B, #121823);
        border: 1px solid #263043; border-radius: 12px;
        padding: 20px 26px; color: #E5E9F0;}
  .top {display: flex; justify-content: space-between; align-items: flex-start;
        gap: 20px; flex-wrap: wrap;}
  .eyebrow {color: #4C8DFF; font-size: 11px; font-weight: 600;
            letter-spacing: 0.14em; text-transform: uppercase;}
  .title {font-size: 25px; font-weight: 700; letter-spacing: -0.01em;
          margin-top: 5px;}
  .sub {color: #8B97A9; font-size: 13px; margin-top: 3px;}
  .stats {display: flex; gap: 10px; flex-wrap: wrap;}
  .stat {background: #0E1117; border: 1px solid #263043; border-radius: 8px;
         padding: 7px 14px; min-width: 84px;}
  .stat b {display: block; font-family: 'JetBrains Mono', Consolas, monospace;
           font-size: 15px; font-weight: 600;}
  .stat span {color: #8B97A9; font-size: 10px; text-transform: uppercase;
              letter-spacing: 0.08em;}
  .pipe {display: flex; align-items: center; margin-top: 18px;
         flex-wrap: wrap; row-gap: 8px;}
  .step {display: flex; align-items: center; gap: 10px; background: #0E1117;
         border: 1px solid #263043; border-radius: 8px; padding: 8px 12px;
         white-space: nowrap;}
  .num {width: 22px; height: 22px; border-radius: 50%;
        background: rgba(76,141,255,0.18); color: #8DB4FF; font-size: 11px;
        font-weight: 700; display: flex; align-items: center;
        justify-content: center;}
  .step b {display: block; font-size: 12px; font-weight: 600;}
  .step small {display: block; color: #8B97A9; font-size: 10.5px;}
  .dot {width: 7px; height: 7px; border-radius: 50%; background: #34C38F;
        margin-left: 4px;}
  .dot.warn {background: #F5A524;}
  .link {flex: 1; min-width: 14px; height: 2px; margin: 0 6px;
         background: linear-gradient(90deg, #263043, #4C8DFF, #263043);
         background-size: 200% 100%; animation: flow 4s linear infinite;}
  @keyframes flow {0% {background-position: 0% 50%;}
                   100% {background-position: 200% 50%;}}
</style>
<div class="hdr">
  <div class="top">
    <div>
      <div class="eyebrow">Data engineering console</div>
      <div class="title">Business Data Analytics</div>
      <div class="sub">Visualization &amp; Machine Learning Prediction System</div>
    </div>
    <div class="stats">
      <div class="stat"><b>__ROWS__</b><span>Rows</span></div>
      <div class="stat"><b>__COLS__</b><span>Columns</span></div>
      <div class="stat"><b>__PERIOD__</b><span>Period</span></div>
      <div class="stat"><b>__MODELS__ / 2</b><span>Models loaded</span></div>
    </div>
  </div>
  <div class="pipe">
    <div class="step"><div class="num">1</div>
      <div><b>Raw data</b><small>Source CSV</small></div></div>
    <div class="link"></div>
    <div class="step"><div class="num">2</div>
      <div><b>Cleaning</b><small>Validated &amp; typed</small></div></div>
    <div class="link"></div>
    <div class="step"><div class="num">3</div>
      <div><b>Features</b><small>__FEATS__ model inputs</small></div></div>
    <div class="link"></div>
    <div class="step"><div class="num">4</div>
      <div><b>ML models</b><small>Random Forest x2</small></div>
      <span class="dot __MODEL_STATE__"></span></div>
    <div class="link"></div>
    <div class="step"><div class="num">5</div>
      <div><b>Dashboard</b><small>Streamlit app</small></div>
      <span class="dot"></span></div>
  </div>
</div>
"""
components.html(
    BANNER_HTML.replace("__ROWS__", f"{len(df):,}")
               .replace("__COLS__", str(df.shape[1]))
               .replace("__PERIOD__", period_text)
               .replace("__MODELS__", str(models_found))
               .replace("__FEATS__", str(len(FEATURE_COLUMNS)))
               .replace("__MODEL_STATE__", model_state),
    height=215
)

P_OVERVIEW = "🏠 Overview"
P_TREND = "📈 Monthly Trend"
P_TABLE = "🗂️ Data Table"
P_REVENUE = "🤖 Revenue Prediction"
P_CLASS = "🎯 Profit / Loss Prediction"
P_MODEL = "🧠 Model Insights"
P_ABOUT = "ℹ️ About Project"

page = st.radio(
    "Go to",
    [P_OVERVIEW, P_TREND, P_TABLE, P_REVENUE, P_CLASS, P_MODEL, P_ABOUT],
    horizontal=True, label_visibility="collapsed"
)
st.write("")

# The About and Model pages do not depend on the filters
needs_data = page in [P_OVERVIEW, P_TREND, P_TABLE]
if needs_data and filtered.empty:
    st.warning("No records match your filters. Change the sidebar selections.")
    st.stop()

total_revenue = filtered["Revenue"].sum()
total_profit = filtered["Profit"].sum()
total_units = filtered["Units_Sold"].sum()
margin = (total_profit / total_revenue * 100) if total_revenue else 0

categories_for_prediction = sorted(df["Product_Category"].unique().tolist())

# ==================================================
# PAGE: OVERVIEW
# ==================================================
if page == P_OVERVIEW:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Total Revenue", money(total_revenue),
                    "All money earned from sales"), unsafe_allow_html=True)
    k2.markdown(kpi("Total Profit", money(total_profit),
                    "Money left after costs"), unsafe_allow_html=True)
    k3.markdown(kpi("Profit Margin", f"{margin:.1f}%",
                    "Profit out of every ₹100 of revenue"),
                unsafe_allow_html=True)
    k4.markdown(kpi("Units Sold", f"{total_units:,.0f}",
                    "Total items sold"), unsafe_allow_html=True)
    st.write("")

    # ---- Chart 1: revenue by category ----
    st.subheader("1. Which product category earns the most?")
    howto("Each bar is one product category. A longer bar means more revenue.")

    revenue_category = (
        filtered.groupby("Product_Category")["Revenue"]
        .sum().sort_values(ascending=True)
    )
    fig = px.bar(
        revenue_category.rename("Revenue (₹)").reset_index(),
        x="Revenue (₹)", y="Product_Category", orientation="h",
        color_discrete_sequence=[BLUE]
    )
    fig.update_yaxes(title="")
    st.plotly_chart(clean(fig, 450), use_container_width=True)

    best_cat = revenue_category.idxmax()
    best_cat_share = (
        revenue_category.max() / revenue_category.sum() * 100
        if revenue_category.sum() else 0
    )
    insight(f"**{best_cat}** earns the most revenue, about "
            f"{best_cat_share:.0f}% of the total.")

    # ---- Chart 2: profit margin by category ----
    st.subheader("2. Which category keeps the most profit?")
    howto("Margin = profit kept out of every ₹100 of revenue. "
          "A category can sell a lot but still have a low margin.")

    cat_money = filtered.groupby("Product_Category")[
        ["Revenue", "Profit"]].sum()
    cat_money = cat_money[cat_money["Revenue"] != 0]
    cat_margin = (cat_money["Profit"] / cat_money["Revenue"] * 100
                  ).sort_values(ascending=True)

    if cat_margin.empty:
        st.info("Not enough data to calculate margins.")
    else:
        fig = px.bar(
            cat_margin.rename("Profit margin (%)").reset_index(),
            x="Profit margin (%)", y="Product_Category", orientation="h",
            color_discrete_sequence=[VIOLET]
        )
        fig.update_yaxes(title="")
        st.plotly_chart(clean(fig, 450), use_container_width=True)
        insight(f"**{cat_margin.idxmax()}** has the best margin "
                f"({cat_margin.max():.1f}%). "
                f"**{cat_margin.idxmin()}** has the lowest "
                f"({cat_margin.min():.1f}%).")

    # ---- Chart 3 and 4: region revenue and profit ----
    region_revenue = filtered.groupby("Region")["Revenue"].sum().sort_values(
        ascending=False)
    region_profit = filtered.groupby("Region")["Profit"].sum().sort_values(
        ascending=False)

    left, right = st.columns(2)

    with left:
        st.subheader("3. Revenue by region")
        howto("Each bar is a region. Taller bar = more revenue.")
        fig = px.bar(
            region_revenue.rename("Revenue (₹)").reset_index(),
            x="Region", y="Revenue (₹)", color_discrete_sequence=[GREEN]
        )
        st.plotly_chart(clean(fig), use_container_width=True)
        insight(f"**{region_revenue.idxmax()}** is the top region for revenue.")

    with right:
        st.subheader("4. Profit by region")
        howto("Same regions, but showing profit instead of revenue.")
        fig = px.bar(
            region_profit.rename("Profit (₹)").reset_index(),
            x="Region", y="Profit (₹)", color_discrete_sequence=[AMBER]
        )
        st.plotly_chart(clean(fig), use_container_width=True)
        insight(f"**{region_profit.idxmax()}** makes the most profit.")

    # ---- Chart 5: donut of revenue share ----
    st.subheader("5. How is revenue split between regions?")
    howto("The full circle is 100% of revenue. Each slice is one region's share.")
    fig = px.pie(
        region_revenue.rename("Revenue").reset_index(),
        names="Region", values="Revenue", hole=0.5,
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(clean(fig, 400), use_container_width=True)
    insight(f"{region_revenue.idxmax()} holds the biggest slice; "
            f"{region_revenue.idxmin()} holds the smallest.")

    # ---- Summary report download ----
    st.subheader("📄 Download summary report")
    summary_text = (
        "BUSINESS ANALYTICS SUMMARY\n"
        "==========================\n"
        f"Date range : {date_text}\n"
        f"Regions    : {', '.join(selected_regions)}\n"
        f"Categories : {len(selected_categories)} selected\n"
        f"Records    : {len(filtered):,}\n\n"
        f"Total revenue : {money(total_revenue)}\n"
        f"Total profit  : {money(total_profit)}\n"
        f"Profit margin : {margin:.1f}%\n"
        f"Units sold    : {total_units:,.0f}\n\n"
        f"Top category by revenue : {best_cat}\n"
        f"Top region by revenue   : {region_revenue.idxmax()}\n"
        f"Top region by profit    : {region_profit.idxmax()}\n"
    )
    if not cat_margin.empty:
        summary_text += (
            f"Best margin category    : {cat_margin.idxmax()} "
            f"({cat_margin.max():.1f}%)\n"
            f"Lowest margin category  : {cat_margin.idxmin()} "
            f"({cat_margin.min():.1f}%)\n"
        )
    summary_text += ("\nNote: the dataset is synthetic and does not "
                     "represent official Apple business results.\n")

    st.download_button(
        "⬇️ Download summary (TXT)", data=summary_text,
        file_name="business_summary.txt", mime="text/plain"
    )

# ==================================================
# PAGE: MONTHLY TREND
# ==================================================
elif page == P_TREND:
    st.subheader("Monthly trend")

    if not has_dates:
        st.info("No valid sales dates are available.")
    else:
        dated = filtered.dropna(subset=["Sales_Date"]).copy()
        dated["Month"] = dated["Sales_Date"].dt.to_period("M").dt.to_timestamp()

        choice = st.radio("Show", ["Revenue", "Profit", "Units_Sold"],
                          horizontal=True)

        monthly = dated.groupby("Month")[choice].sum().reset_index()

        howto("Each point is one month. If the line goes up, "
              "the business is growing; if it goes down, it is slowing.")

        fig = px.line(monthly, x="Month", y=choice, markers=True,
                      color_discrete_sequence=[BLUE])
        st.plotly_chart(clean(fig, 420), use_container_width=True)

        best_month = monthly.loc[monthly[choice].idxmax(), "Month"]
        worst_month = monthly.loc[monthly[choice].idxmin(), "Month"]
        insight(f"Best month: **{best_month:%B %Y}**. "
                f"Lowest month: **{worst_month:%B %Y}**.")

# ==================================================
# PAGE: DATA TABLE
# ==================================================
elif page == P_TABLE:
    st.subheader("Dataset table")
    howto("This is the raw data after your sidebar filters. "
          "Type in the box to search.")

    search_text = st.text_input("🔍 Search category or region")
    table = filtered
    if search_text:
        hit = (
            table["Product_Category"].str.contains(search_text, case=False)
            | table["Region"].str.contains(search_text, case=False)
        )
        table = table[hit]

    rows = st.select_slider("Rows to show", [20, 50, 100, 500], value=50)
    st.dataframe(table.head(rows), use_container_width=True)

    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_business_analytics.csv",
        mime="text/csv"
    )

# ==================================================
# PAGE: REVENUE PREDICTION
# ==================================================
elif page == P_REVENUE:
    st.subheader("Revenue prediction")
    howto("Fill in the business details and click Run Prediction. "
          "The Random Forest model estimates the revenue.")

    if (os.path.exists(REVENUE_MODEL_PATH)
            and os.path.exists(REVENUE_FEATURES_PATH)):
        try:
            revenue_model = load_model(REVENUE_MODEL_PATH)
            revenue_features = load_model(REVENUE_FEATURES_PATH)

            submitted, rev_input = create_prediction_form(
                "revenue", categories_for_prediction)

            if submitted:
                model_input = encode_for_model(rev_input, revenue_features)
                prediction = float(revenue_model.predict(model_input)[0])

                row = rev_input.iloc[0]
                simple = (row["Units_Sold"] * row["Selling_Price"]
                          * (1 - row["Discount_Percent"] / 100))

                c1, c2 = st.columns([2, 1])
                c1.markdown(
                    f'<div class="result"><div>Predicted Revenue</div>'
                    f'<div class="big">₹{prediction:,.2f}</div></div>',
                    unsafe_allow_html=True)
                c2.metric("Simple check (units × price − discount)",
                          money(simple))

                # Which inputs the model relies on most overall
                importance = grouped_importance(revenue_model, revenue_features)
                if importance is not None:
                    st.subheader("Why? What the model pays attention to")
                    howto("A longer bar means the model relies more on that "
                          "input when predicting revenue.")
                    top = importance.head(8).sort_values(ascending=True)
                    fig = px.bar(
                        top.rename("Importance (%)").reset_index(),
                        x="Importance (%)", y="index",
                        orientation="h", color_discrete_sequence=[GREEN])
                    fig.update_yaxes(title="")
                    st.plotly_chart(clean(fig, 340), use_container_width=True)

                st.caption(
                    "This is an estimate based on synthetic training "
                    "data, not a guaranteed business result.")

        except Exception as error:
            st.error(f"Revenue prediction error: {error}")
    else:
        st.warning("Revenue model or feature file is missing. "
                   "Check the models folder.")

# ==================================================
# PAGE: PROFIT / LOSS PREDICTION
# ==================================================
elif page == P_CLASS:
    st.subheader("Profit / Loss prediction")
    howto("Fill in the business details and click Run Prediction. "
          "The classifier says whether the record is a Profit or a Loss, "
          "and the bars show how sure it is.")

    if (os.path.exists(CLASSIFIER_PATH)
            and os.path.exists(CLASSIFICATION_FEATURES_PATH)):
        try:
            profit_model = load_model(CLASSIFIER_PATH)
            classification_features = load_model(CLASSIFICATION_FEATURES_PATH)

            submitted, class_input = create_prediction_form(
                "classification", categories_for_prediction)

            if submitted:
                class_model_input = encode_for_model(
                    class_input, classification_features)
                predicted_class = profit_model.predict(class_model_input)[0]

                label = str(predicted_class).strip().lower()
                if label == "profit":
                    st.success("Predicted Class: PROFIT")
                elif label == "loss":
                    st.error("Predicted Class: LOSS")
                else:
                    st.info(f"Predicted Class: {predicted_class}")

                if hasattr(profit_model, "predict_proba"):
                    probs = profit_model.predict_proba(class_model_input)[0]
                    prob_df = pd.DataFrame({
                        "Class": [str(c) for c in profit_model.classes_],
                        "Probability (%)": [round(float(p) * 100, 2)
                                            for p in probs]
                    })
                    fig = px.bar(
                        prob_df, x="Probability (%)", y="Class",
                        orientation="h", range_x=[0, 100],
                        text="Probability (%)",
                        color_discrete_sequence=[BLUE])
                    st.plotly_chart(clean(fig, 250), use_container_width=True)

                importance = grouped_importance(
                    profit_model, classification_features)
                if importance is not None:
                    st.subheader("Why? What the model pays attention to")
                    howto("A longer bar means the model relies more on that "
                          "input when deciding between profit and loss.")
                    top = importance.head(8).sort_values(ascending=True)
                    fig = px.bar(
                        top.rename("Importance (%)").reset_index(),
                        x="Importance (%)", y="index",
                        orientation="h", color_discrete_sequence=[AMBER])
                    fig.update_yaxes(title="")
                    st.plotly_chart(clean(fig, 340), use_container_width=True)

                st.caption(
                    "The classifier predicts a learned label, not a "
                    "guaranteed financial outcome. Results depend on "
                    "the synthetic training dataset.")

        except Exception as error:
            st.error(f"Profit/loss prediction error: {error}")
    else:
        st.warning("Classification model or feature file is missing. "
                   "Check the models folder.")

# ==================================================
# PAGE: MODEL INSIGHTS
# ==================================================
elif page == P_MODEL:
    st.subheader("How good are the models?")
    st.caption(
        "Scores are calculated on a sample of up to 5,000 rows from the "
        "cleaned dataset. The models may have seen some of these rows during "
        "training, so treat the numbers as an approximate check, not a final "
        "test result."
    )

    # ---------- Revenue model ----------
    st.markdown("### 💰 Revenue model (Random Forest Regression)")

    if (os.path.exists(REVENUE_MODEL_PATH)
            and os.path.exists(REVENUE_FEATURES_PATH)):
        try:
            result = check_revenue_model(
                REVENUE_MODEL_PATH, REVENUE_FEATURES_PATH, DATA_PATH)

            if result is None:
                st.info("The dataset does not contain all the columns needed "
                        "to check this model.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.markdown(kpi("R² score", f"{result['r2']:.3f}",
                                "1.0 = perfect, 0 = no better than average"),
                            unsafe_allow_html=True)
                m2.markdown(kpi("Average error", money(result["mae"]),
                                "Typical gap between real and predicted"),
                            unsafe_allow_html=True)
                m3.markdown(kpi("Rows checked", f"{result['rows']:,}",
                                "Sample from the dataset"),
                            unsafe_allow_html=True)
                st.write("")
                insight(f"The model explains about {result['r2'] * 100:.0f}% "
                        f"of the ups and downs in revenue, and its "
                        f"predictions are off by around "
                        f"{money(result['mae'])} on average.")

            model = load_model(REVENUE_MODEL_PATH)
            feats = load_model(REVENUE_FEATURES_PATH)
            importance = grouped_importance(model, feats)
            if importance is not None:
                st.markdown("**Which inputs matter most?**")
                howto("A longer bar means the model relies more on that input.")
                top = importance.head(10).sort_values(ascending=True)
                fig = px.bar(
                    top.rename("Importance (%)").reset_index(),
                    x="Importance (%)", y="index",
                    orientation="h", color_discrete_sequence=[GREEN])
                fig.update_yaxes(title="")
                st.plotly_chart(clean(fig, 400), use_container_width=True)
                insight(f"**{importance.index[0]}** is the most important "
                        f"input ({importance.iloc[0]:.1f}%).")

        except Exception as error:
            st.error(f"Could not check the revenue model: {error}")
    else:
        st.warning("Revenue model or feature file is missing.")

    st.divider()

    # ---------- Classifier ----------
    st.markdown("### 🎯 Profit / Loss model (Random Forest Classifier)")

    if (os.path.exists(CLASSIFIER_PATH)
            and os.path.exists(CLASSIFICATION_FEATURES_PATH)):
        try:
            result = check_classifier(
                CLASSIFIER_PATH, CLASSIFICATION_FEATURES_PATH, DATA_PATH)

            if result is None:
                st.info("Could not check this model. The dataset needs a "
                        "Profit column and the model needs Profit / Loss "
                        "classes.")
            else:
                st.caption("A record counts as 'Profit' when its Profit "
                           "value is above zero.")
                a1, a2 = st.columns(2)
                a1.markdown(kpi("Accuracy", f"{result['accuracy'] * 100:.1f}%",
                                "Share of correct predictions"),
                            unsafe_allow_html=True)
                a2.markdown(kpi("Rows checked", f"{result['rows']:,}",
                                "Sample from the dataset"),
                            unsafe_allow_html=True)
                st.write("")

                st.markdown("**Confusion matrix**")
                howto("Each box counts records. The diagonal boxes "
                      "(top-left and bottom-right) are correct predictions. "
                      "The other two boxes are mistakes.")
                fig = px.imshow(
                    result["matrix"], text_auto=True,
                    color_continuous_scale=["#1A2231", "#4C8DFF"], aspect="auto")
                fig.update_coloraxes(showscale=False)
                fig = clean(fig, 340)
                st.plotly_chart(fig, use_container_width=True)

                matrix = result["matrix"].values
                correct = int(matrix[0][0] + matrix[1][1])
                insight(f"The model predicted {correct:,} of "
                        f"{result['rows']:,} records correctly "
                        f"({correct / result['rows'] * 100:.1f}%).")

            model = load_model(CLASSIFIER_PATH)
            feats = load_model(CLASSIFICATION_FEATURES_PATH)
            importance = grouped_importance(model, feats)
            if importance is not None:
                st.markdown("**Which inputs matter most?**")
                howto("A longer bar means the model relies more on that input.")
                top = importance.head(10).sort_values(ascending=True)
                fig = px.bar(
                    top.rename("Importance (%)").reset_index(),
                    x="Importance (%)", y="index",
                    orientation="h", color_discrete_sequence=[AMBER])
                fig.update_yaxes(title="")
                st.plotly_chart(clean(fig, 400), use_container_width=True)
                insight(f"**{importance.index[0]}** is the most important "
                        f"input ({importance.iloc[0]:.1f}%).")

        except Exception as error:
            st.error(f"Could not check the classifier: {error}")
    else:
        st.warning("Classification model or feature file is missing.")

# ==================================================
# PAGE: ABOUT PROJECT
# ==================================================
else:
    st.subheader("About this project")

    st.markdown(
        """
        <div class="card">
        <h4>🎯 Objective</h4>
        Analyse business sales data, find which products and regions perform
        best, and use machine learning to predict revenue and whether a
        sales record ends in profit or loss.
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)
    a.markdown(kpi("Records", f"{len(df):,}", "Rows in the cleaned dataset"),
               unsafe_allow_html=True)
    b.markdown(kpi("Columns", f"{df.shape[1]}", "Fields per record"),
               unsafe_allow_html=True)
    c.markdown(kpi("Categories", f"{df['Product_Category'].nunique()}",
                   f"Across {df['Region'].nunique()} regions"),
               unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
        <div class="card">
        <h4>🧪 Data preparation in this app</h4>
        • Revenue, Profit and Units Sold are converted to numbers; invalid
          values become 0.<br>
        • Missing or empty category and region names become "Unknown".<br>
        • Sales dates are converted to real dates for the monthly trend.<br>
        • The required columns are checked before anything is shown.
        </div>

        <div class="card">
        <h4>🤖 Machine learning models</h4>
        • <b>Revenue prediction:</b> Random Forest Regression.<br>
        • <b>Profit / Loss prediction:</b> Random Forest Classification.<br>
        • Both models use the same 17 business inputs (units, prices,
          costs, discount, segment, region, channel, demand, category,
          year and month).<br>
        • The <i>Model Insights</i> page shows accuracy and which inputs
          matter most.
        </div>

        <div class="card">
        <h4>🧭 How to use the dashboard</h4>
        1. Use the sidebar filters to choose categories, regions and dates.<br>
        2. Read the Overview charts; each one has a "How to read" note.<br>
        3. Check the Monthly Trend to see growth over time.<br>
        4. Try the two prediction pages with your own numbers.<br>
        5. Download the summary report or the filtered data.
        </div>

        <div class="card">
        <h4>🛠️ Tools used</h4>
        Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("See all dataset columns"):
        st.write(list(df.columns))

    st.caption("The Apple-product dataset is synthetic and does not "
               "represent official Apple business results.")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("Business Data Analytics, Visualization and Machine Learning "
           "Prediction System  |  Dashboard version 5")
st.caption("The Apple-product dataset is synthetic and does not represent "
           "official Apple business results.")