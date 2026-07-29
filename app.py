"""
NITI Aayog Anemia Screening Portal
----------------------------------
A Streamlit front-end for the XGBoost anemia-risk model trained in
CDAC_Project_Trial3_for_aspirational_districts.ipynb.

Run with:
    streamlit run app.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import importlib.metadata as _md
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ==========================================================
# 0. Paths (resolved relative to THIS file, not the shell's
#    current working directory -- fixes the most common
#    "model file not found" crash when the app is launched
#    from a different folder than app.py lives in)
# ==========================================================
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent if APP_DIR.name == "src" else APP_DIR # Adjust if your structure differs


def _find_latest(pattern: str) -> "Optional[Path]":
    """Return the most recently-dated file in APP_DIR matching `pattern`.

    The training notebook stamps model/feature exports with the date they
    were generated (e.g. `xgboost_anemia_optimized_20260729.pkl`), so the
    exact filename changes on every retrain. YYYYMMDD sorts correctly as
    plain text, so picking the lexicographically-last match reliably picks
    the newest export without app.py needing to hardcode a date.
    """
    matches = sorted(APP_DIR.glob(pattern))
    return matches[-1] if matches else None


JSON_MODEL_PATH = APP_DIR / "xgboost_anemia_model.json"  # not date-stamped by the notebook
PKL_MODEL_PATH = _find_latest("xgboost_anemia_optimized_*.pkl")

# ==========================================================
# 1. Page configuration & visual theme
# ==========================================================
st.set_page_config(
    page_title="NITI Aayog Anemia Screening Portal",
    page_icon="🩸",
    layout="wide",
)

PRIMARY = "#0F5257"      # deep teal - headers, primary UI accents
PRIMARY_DARK = "#0B3D40"
RISK_HIGH = "#C8483C"    # clinical coral-red - positive screen
RISK_LOW = "#2F8F6E"     # grounded green - clear screen
BG_SOFT = "#F5F8F8"
TEXT_MUTED = "#5C6B6E"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    h1, h2, h3 {{
        font-family: 'Lora', serif !important;
    }}
    .block-container {{
        padding-top: 2rem;
    }}
    .risk-banner {{
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }}
    .risk-banner.high {{
        background-color: #FBEAEA;
        color: {RISK_HIGH};
        border: 1px solid {RISK_HIGH}55;
    }}
    .risk-banner.low {{
        background-color: #E9F5F0;
        color: {RISK_LOW};
        border: 1px solid {RISK_LOW}55;
    }}
    .card {{
        background: white;
        border: 1px solid #E4EAEA;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }}
    .muted {{
        color: {TEXT_MUTED};
        font-size: 0.88rem;
    }}
    .disclaimer {{
        background-color: #FFF8E8;
        border: 1px solid #F0DFA6;
        border-radius: 8px;
        padding: 0.8rem 1.1rem;
        font-size: 0.86rem;
        color: #6B5A17;
        margin-top: 1.4rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩸 Anemia Screening & Risk Analytics Portal")
st.markdown(
    "Statistical risk screening built on **NFHS-5 data**, for frontline health "
    "workers supporting India's Aspirational Districts."
)

# ==========================================================
# 2. Model & Feature Checklist Loading (ALIGNED WITH PRODUCTION)
# ==========================================================

# FIXED: Load the associated feature checklist to guarantee column shape
FEATURE_CHECKLIST_PATH = REPO_ROOT / "models" / "xgboost_features_v1_20260729.pkl"

if FEATURE_CHECKLIST_PATH.exists():
    FALLBACK_FEATURE_ORDER = joblib.load(FEATURE_CHECKLIST_PATH)
else:
    # Safe fallback index if file hasn't moved yet (syntax fixed)
    FALLBACK_FEATURE_ORDER = [
        "state_code", "district_code", "age", "wealth_index_national", "wealth_index_state",
        "education_level", "total_children_born", "is_pregnant", "freq_milk_curd",
        "freq_pulses_beans", "freq_green_leafy_veg", "freq_fruits", "freq_eggs", "freq_fish",
        "freq_chicken_meat", "freq_fried_food", "freq_aerated_drinks", "is_aspirational",
        "residence_type_1", "residence_type_2"
    ] + [f"religion_{i}" for i in range(1, 10)] + ["religion_96"] \
      + [f"caste_ethnicity_{i}" for i in [1, 2, 3, 4, 8]] \
      + [f"marital_status_standard_{i}" for i in [0, 1, 3, 4, 5]] \
      + [f"marital_status_state_{i}" for i in range(0, 7)]


@st.cache_resource(show_spinner="Loading production model...")
def load_production_model():
    """Returns (model, feature_order, error_message)."""
    if JSON_MODEL_PATH.exists():
        try:
            m = xgb.XGBClassifier()
            m.load_model(str(JSON_MODEL_PATH))
            booster_features = m.get_booster().feature_names
            feat_order = list(booster_features) if booster_features else FALLBACK_FEATURE_ORDER
            return m, feat_order, None
        except Exception as e:
            json_error = f"Found {JSON_MODEL_PATH.name} but failed to load it: {e}"
    else:
        json_error = None

    if PKL_MODEL_PATH is not None and PKL_MODEL_PATH.exists():
        try:
            m = joblib.load(PKL_MODEL_PATH)
            try:
                # Safely extract booster names if embedded, else default to our saved checklist file
                feat_order = list(m.get_booster().feature_names) if m.get_booster().feature_names else FALLBACK_FEATURE_ORDER
            except Exception:
                feat_order = FALLBACK_FEATURE_ORDER
            return m, feat_order, None
        except Exception as e:
            pkl_error = str(e)
            combined = (json_error + "\n\n" if json_error else "") + f"Found {PKL_MODEL_PATH.name} but failed to load it: {pkl_error}"
            return None, FALLBACK_FEATURE_ORDER, combined

    return None, FALLBACK_FEATURE_ORDER, "No model file found."

model, FEATURE_ORDER, load_error = load_production_model()

if load_error:
    st.error("⚠️ Could not load the production model.")
    with st.expander("Show technical details"):
        st.code(load_error)

if not load_error and not FEATURE_CHECKLIST_PATH.exists():
    st.info(
        "ℹ️ Using the model's embedded/fallback feature order. Ensure "
        "'xgboost_features_v1_20260729.pkl' is placed in your models directory "
        "to guarantee column shape alignment."
    )

# ==========================================================
# 2b. State / district name lookup
# ==========================================================
LOCATION_LOOKUP_PATH = APP_DIR / "state_district_lookup.csv"


@st.cache_resource(show_spinner=False)
def load_location_lookup():
    if not LOCATION_LOOKUP_PATH.exists():
        return {}, {}, {}, f"'{LOCATION_LOOKUP_PATH.name}' not found in:\n{APP_DIR}"
    try:
        loc = pd.read_csv(LOCATION_LOOKUP_PATH)
        state_to_districts = (
            loc.sort_values("district_name")
               .groupby("state_name")["district_name"]
               .apply(list)
               .to_dict()
        )
        code_lookup = {
            (row.state_name, row.district_name): (int(row.state_code), int(row.district_code))
            for row in loc.itertuples()
        }
        aspirational_lookup = {
            (row.state_name, row.district_name): bool(row.is_aspirational)
            for row in loc.itertuples()
        }
        return state_to_districts, code_lookup, aspirational_lookup, None
    except Exception as e:
        return {}, {}, {}, f"Failed to read '{LOCATION_LOOKUP_PATH.name}': {e}"


STATE_TO_DISTRICTS, LOCATION_CODE_LOOKUP, ASPIRATIONAL_LOOKUP, location_error = load_location_lookup()

# ==========================================================
# 3. Sidebar inputs (layman-friendly labels)
# ==========================================================
st.sidebar.header("📋 Patient Profile")
age = st.sidebar.slider("Patient Age (Years)", min_value=15, max_value=49, value=28)
is_pregnant = st.sidebar.selectbox("Is the patient currently pregnant?", ["No", "Yes"])
total_children = st.sidebar.number_input("Total Children Born", min_value=0, max_value=15, value=1, step=1)

st.sidebar.header("🏠 Socioeconomic & Location")

if location_error:
    st.sidebar.warning("⚠️ State/district list unavailable -- using a default location.")
    with st.sidebar.expander("Show technical details"):
        st.caption(location_error)
    state_code, district_code = 10, 203
    location_label = "Bihar (default -- district list unavailable)"
    is_aspirational_flag = False
else:
    state_names = sorted(STATE_TO_DISTRICTS.keys())
    default_state = "Bihar" if "Bihar" in state_names else state_names[0]
    state_selected = st.sidebar.selectbox(
        "State", state_names, index=state_names.index(default_state)
    )
    district_names = sorted(STATE_TO_DISTRICTS[state_selected])
    default_district = "Pashchim Champaran" if "Pashchim Champaran" in district_names else district_names[0]
    district_selected = st.sidebar.selectbox(
        "District", district_names, index=district_names.index(default_district)
    )
    state_code, district_code = LOCATION_CODE_LOOKUP[(state_selected, district_selected)]
    is_aspirational_flag = ASPIRATIONAL_LOOKUP.get((state_selected, district_selected), False)
    location_label = f"{district_selected}, {state_selected}"

is_aspirational = "Yes" if is_aspirational_flag else "No"
st.sidebar.caption(
    f"📍 Aspirational District status: **{is_aspirational}** "
    "(auto-detected from the state/district selected above, per NITI Aayog's "
    "112 Aspirational Districts list)."
)

residence = st.sidebar.selectbox("Type of Residence Area", ["Urban", "Rural"])
wealth = st.sidebar.select_slider(
    "Household Wealth Index Level",
    options=["Poorest", "Poorer", "Middle", "Richer", "Richest"],
    value="Middle",
)
education = st.sidebar.selectbox(
    "Highest Education Completed",
    ["No Education", "Primary School", "Secondary School", "Higher Secondary/College"],
)

st.sidebar.header("🥦 Dietary Consumption Frequencies")
# FIXED: Maps directly to your optimized training scale 0 < 1 < 2 < 3
diet_mapping = {
    "Never": 0, 
    "Occasional / Few times a week": 1, 
    "Weekly": 2, 
    "Daily": 3
}
diet_options = list(diet_mapping.keys())

with st.sidebar.expander("Show all 9 dietary items", expanded=True):
    freq_milk = st.selectbox("Milk / Curd frequency", diet_options, index=1)
    freq_pulses = st.selectbox("Pulses / Beans frequency", diet_options, index=2)
    freq_veg = st.selectbox("Green Leafy Vegetables frequency", diet_options, index=2)
    freq_fruits = st.selectbox("Fruits frequency", diet_options, index=1)
    freq_eggs = st.selectbox("Eggs frequency", diet_options, index=1)
    freq_fish = st.selectbox("Fish frequency", diet_options, index=0)
    freq_chicken = st.selectbox("Chicken / Meat frequency", diet_options, index=0)
    freq_fried = st.selectbox("Fried Foods frequency", diet_options, index=1)
    freq_drinks = st.selectbox("Aerated Carbonated Drinks frequency", diet_options, index=0)

with st.sidebar.expander("🩺 Environment diagnostics"):
    for pkg in ["streamlit", "xgboost", "scikit-learn", "pandas", "numpy", "shap", "plotly"]:
        try:
            st.caption(f"{pkg}: {_md.version(pkg)}")
        except _md.PackageNotFoundError:
            st.caption(f"{pkg}: not installed")

# ==========================================================
# 4. Feature engineering
# ==========================================================
wealth_map = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
edu_map = {"No Education": 0, "Primary School": 1, "Secondary School": 2, "Higher Secondary/College": 3}

input_data = {
    "state_code": state_code,
    "district_code": district_code,
    "age": age,
    "wealth_index_national": wealth_map[wealth],
    "wealth_index_state": wealth_map[wealth],
    "education_level": edu_map[education],
    "total_children_born": total_children,
    "is_pregnant": 1 if is_pregnant == "Yes" else 0,
    "freq_milk_curd": diet_mapping[freq_milk],
    "freq_pulses_beans": diet_mapping[freq_pulses],
    "freq_green_leafy_veg": diet_mapping[freq_veg],
    "freq_fruits": diet_mapping[freq_fruits],
    "freq_eggs": diet_mapping[freq_eggs],
    "freq_fish": diet_mapping[freq_fish],
    "freq_chicken_meat": diet_mapping[freq_chicken],
    "freq_fried_food": diet_mapping[freq_fried],
    "freq_aerated_drinks": diet_mapping[freq_drinks],
    "is_aspirational": 1 if is_aspirational == "Yes" else 0,
    "residence_type_1": 1 if residence == "Urban" else 0,
    "residence_type_2": 1 if residence == "Rural" else 0,
}
for i in range(1, 10):
    input_data[f"religion_{i}"] = 0
input_data["religion_96"] = 0
for i in [1, 2, 3, 4, 8]:
    input_data[f"caste_ethnicity_{i}"] = 0
for i in [0, 1, 3, 4, 5]:
    input_data[f"marital_status_standard_{i}"] = 0
for i in range(0, 7):
    input_data[f"marital_status_state_{i}"] = 0

df_features = pd.DataFrame([input_data]).reindex(columns=FEATURE_ORDER, fill_value=0)


def group_of(col: str) -> str:
    """Maps a raw model column name to a human-readable, grouped label."""
    friendly = {
        "age": "Age",
        "total_children_born": "Number of Children Born",
        "is_pregnant": "Currently Pregnant",
        "wealth_index_national": "Household Wealth Index",
        "wealth_index_state": "Household Wealth Index",
        "education_level": "Education Level",
        "is_aspirational": "Aspirational District Status",
        "freq_milk_curd": "Diet: Milk / Curd",
        "freq_pulses_beans": "Diet: Pulses / Beans",
        "freq_green_leafy_veg": "Diet: Green Leafy Vegetables",
        "freq_fruits": "Diet: Fruits",
        "freq_eggs": "Diet: Eggs",
        "freq_fish": "Diet: Fish",
        "freq_chicken_meat": "Diet: Chicken / Meat",
        "freq_fried_food": "Diet: Fried Foods",
        "freq_aerated_drinks": "Diet: Aerated Drinks",
        "state_code": "Geographic Code (State)",
        "district_code": "Geographic Code (District)",
    }
    if col in friendly:
        return friendly[col]
    if col.startswith("residence_type_"):
        return "Residence Area (Urban/Rural)"
    if col.startswith("religion_"):
        return "Religion"
    if col.startswith("caste_ethnicity_"):
        return "Caste / Ethnicity"
    if col.startswith("marital_status_"):
        return "Marital Status"
    return col

# ==========================================================
# 5. Main interface
# ==========================================================
tab_predict, tab_insights, tab_about = st.tabs(
    ["🎯 Risk Assessment", "📊 Model Insights", "ℹ️ About This Tool"]
)

# ---- Prediction (shared across tabs) ----
prob_anemic, predict_error = None, None
if model is not None:
    try:
        prob_anemic = float(model.predict_proba(df_features)[0, 1])
    except Exception as e:
        predict_error = str(e)

THRESHOLD = 0.49

with tab_predict:
    if model is None:
        st.warning("Fix the model loading error above before a prediction can be shown.")
    elif predict_error or prob_anemic is None:
        st.error("⚠️ The model loaded, but the prediction call failed.")
        with st.expander("Show technical details"):
            st.code(predict_error)
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Diagnostic Evaluation Output")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_anemic * 100,
                    number={"suffix": "%", "font": {"size": 42}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": RISK_HIGH if prob_anemic >= THRESHOLD else RISK_LOW},
                        "steps": [
                            {"range": [0, THRESHOLD * 100], "color": "#E9F5F0"},
                            {"range": [THRESHOLD * 100, 100], "color": "#FBEAEA"},
                        ],
                        "threshold": {
                            "line": {"color": "#333333", "width": 3},
                            "thickness": 0.85,
                            "value": THRESHOLD * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, width='stretch')

            if prob_anemic >= THRESHOLD:
                st.markdown(
                    '<div class="risk-banner high">🚨 POSITIVE SCREEN FOR ANEMIA RISK</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    **Clinical suggestion framework**
                    - Score sits above the optimized intervention risk floor (**{THRESHOLD}**).
                    - Recommended action: prioritize for hemoglobin verification (clinical blood test).
                    """
                )
            else:
                st.markdown(
                    '<div class="risk-banner low">✅ LOW RISK PROFILE DETECTED</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    **Clinical suggestion framework**
                    - Profile tracks under the active intervention threshold.
                    - Recommended action: maintain routine preventive supplementation via local Anganwadi supply networks.
                    """
                )

        with col2:
            st.subheader("Patient Input Recap")
            recap = pd.DataFrame(
                {
                    "Field": [
                        "Location", "Age", "Pregnant", "Residence", "Wealth Index", "Education",
                        "Children Born", "Aspirational District",
                    ],
                    "Value": [
                        location_label, f"{age} yrs", is_pregnant, residence, wealth, education,
                        str(total_children), is_aspirational,
                    ],
                }
            )
            st.dataframe(recap, hide_index=True, width='stretch')
            with st.expander("Show full dietary inputs"):
                diet_recap = pd.DataFrame(
                    {
                        "Item": ["Milk/Curd", "Pulses/Beans", "Green Leafy Veg", "Fruits",
                                 "Eggs", "Fish", "Chicken/Meat", "Fried Food", "Aerated Drinks"],
                        "Frequency": [freq_milk, freq_pulses, freq_veg, freq_fruits, freq_eggs,
                                      freq_fish, freq_chicken, freq_fried, freq_drinks],
                    }
                )
                st.dataframe(diet_recap, hide_index=True, width='stretch')

        st.markdown(
            '<div class="disclaimer">⚕️ This tool produces a statistical screening '
            'estimate for triage/prioritization only. It is not a diagnosis. Confirm any '
            'positive screen with a laboratory hemoglobin test before clinical action.</div>',
            unsafe_allow_html=True,
        )

# ---- Tab 2: Model Insights ----
with tab_insights:
    st.subheader("What's driving this assessment?")

    if model is None:
        st.warning("Load the model successfully to see driver analysis.")
    else:
        @st.cache_resource(show_spinner=False)
        def get_shap_explainer(_m):
            import shap  # local import: optional dependency
            return shap.TreeExplainer(_m)

        grouped, kind = None, None
        try:
            explainer = get_shap_explainer(model)
            raw_shap = explainer.shap_values(df_features)
            if isinstance(raw_shap, list):
                raw_shap = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            raw_shap = np.asarray(raw_shap).reshape(-1)
            contrib = pd.DataFrame({"raw_col": df_features.columns, "value": raw_shap})
            contrib["group"] = contrib["raw_col"].apply(group_of)
            grouped = contrib.groupby("group")["value"].sum().reset_index()
            grouped["abs_value"] = grouped["value"].abs()
            grouped = grouped.sort_values("abs_value", ascending=False).head(8).sort_values("value")
            kind = "local"
        except ImportError:
            kind = None
        except Exception:
            kind = None

        if kind is None:
            try:
                raw_scores = model.get_booster().get_score(importance_type="total_gain")
                contrib = pd.DataFrame({"raw_col": list(raw_scores.keys()), "value": list(raw_scores.values())})
                contrib["group"] = contrib["raw_col"].apply(group_of)
                grouped = contrib.groupby("group")["value"].sum().reset_index()
                grouped["value"] = grouped["value"] / grouped["value"].sum() * 100
                grouped["abs_value"] = grouped["value"].abs()
                grouped = grouped.sort_values("abs_value", ascending=False).head(8).sort_values("value")
                kind = "global"
            except Exception:
                grouped, kind = None, None

        if grouped is None or grouped.empty:
            st.info("Feature-importance data isn't available for this model file.")
        else:
            if kind == "local":
                st.caption(
                    "Shows how each factor pushed **this patient's** predicted risk up "
                    "(red) or down (green), compared with an average patient in the "
                    "training data — computed with SHAP."
                )
                colors = [RISK_HIGH if v > 0 else RISK_LOW for v in grouped["value"]]
                x_title = "Impact on this patient's predicted risk"
            else:
                st.caption(
                    "The `shap` package isn't installed, so this shows overall feature "
                    "importance across *all* patients in the training data — not specific "
                    "to this patient. Install `shap` (see requirements.txt) for a true "
                    "per-patient explanation."
                )
                colors = [PRIMARY] * len(grouped)
                x_title = "Share of total model importance (%)"

            fig_bar = go.Figure(
                go.Bar(x=grouped["value"], y=grouped["group"], orientation="h", marker_color=colors)
            )
            fig_bar.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title=x_title,
                yaxis_title=None,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_bar, width='stretch')

        st.markdown(
            "> ⚠️ **Modeling caveat:** the state/district codes are numeric NFHS "
            "administrative IDs, but the tree model treats them as ordinary numeric "
            "values rather than pure categories. In testing, `district_code` alone can "
            "carry as much weight as age or district status — worth revisiting (e.g. "
            "target-encoding or one-hot-encoding geography) in a future model iteration."
        )

# ---- Tab 3: About ----
with tab_about:
    st.subheader("About this screening tool")
    st.markdown(
        """
        This portal serves a **XGBoost classifier** trained on harmonized **NFHS-5**
        (National Family Health Survey) individual-level records, predicting the
        probability that a woman aged 15–49 is anemic (hemoglobin < 11.0 g/dL).

        **Reported model performance** (survey-weighted, held-out test set, from the
        training notebook):
        - Cross-validated ROC-AUC: **0.632**
        - Decision threshold: **0.49** (chosen to maximize weighted macro F1)
        - Weighted macro F1 at that threshold: **0.59** (precision/recall around 0.54–0.64
          depending on class)

        These numbers indicate **modest, better-than-chance discrimination** — appropriate
        for flagging patients for follow-up testing, but not strong enough to stand in for
        a laboratory diagnosis.
        """
    )
    st.markdown(
        """
        **Data & methodology**
        - Source: NFHS-5 (2019–21) individual recode, restricted to women aged 15–49.
        - Target: binary anemia flag derived from measured hemoglobin level.
        - `is_aspirational` flags NITI Aayog's 112 Aspirational Districts by district code.
        - Categorical fields (religion, caste/ethnicity, marital status, residence) are
          one-hot encoded to match the 47 columns the model was trained on.
        """
    )
    st.caption(
        "This is an academic/demo project and screening aid — not an officially "
        "deployed NITI Aayog system."
    )