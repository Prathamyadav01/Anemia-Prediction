import os
for env_var in ["OPENBLAS", "OMP", "MKL", "VECLIB_MAXIMUM", "NUMEXPR_NUM"]:
    os.environ[f"{env_var}_THREADS"] = "1"

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================
# 0. Paths & Theme Config
# ==========================================================
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR, DATA_DIR = APP_DIR / "models", APP_DIR / "data"

st.set_page_config(page_title="Anemia Risk Predictor", page_icon="🩸", layout="wide")
PRIMARY, RISK_HIGH, RISK_LOW, TEXT_MUTED = "#0F5257", "#C8483C", "#2F8F6E", "#5C6B6E"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;700&family=Inter:wght@400;600&display=swap');
    h1, h2, h3 {{ font-family: 'Lora', serif !important; }}
    .block-container {{ padding-top: 2rem; }}
    .risk-banner {{ padding: 1.1rem; border-radius: 10px; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.6rem; text-align: center; }}
    .high {{ background: #FBEAEA; color: {RISK_HIGH}; border: 1px solid {RISK_HIGH}55; }}
    .low {{ background: #E9F5F0; color: {RISK_LOW}; border: 1px solid {RISK_LOW}55; }}
    .disclaimer {{ background: #FFF8E8; border: 1px solid #F0DFA6; border-radius: 8px; padding: 0.8rem; font-size: 0.86rem; color: #6B5A17; margin-top: 1.4rem; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 1. Load Data & Models (Cached)
# ==========================================================
@st.cache_resource(show_spinner="Loading model & assets...")
def load_assets():
    json_path = MODELS_DIR / "xgboost_anemia_model.json"
    pkl_matches = sorted(MODELS_DIR.glob("xgboost_anemia_optimized_*.pkl"))
    feat_path = MODELS_DIR / "xgboost_features_v1_20260729.pkl"
    
    fallback_feats = ["state_code", "district_code", "age", "wealth_index_national", "wealth_index_state", 
                      "education_level", "total_children_born", "is_pregnant", "freq_milk_curd", "freq_pulses_beans", 
                      "freq_green_leafy_veg", "freq_fruits", "freq_eggs", "freq_fish", "freq_chicken_meat", 
                      "freq_fried_food", "freq_aerated_drinks", "is_aspirational", "residence_type_1", "residence_type_2"] + \
                     [f"religion_{i}" for i in range(1, 10)] + ["religion_96"] + [f"caste_ethnicity_{i}" for i in [1, 2, 3, 4, 8]] + \
                     [f"marital_status_standard_{i}" for i in [0, 1, 3, 4, 5]] + [f"marital_status_state_{i}" for i in range(7)]
    
    feat_order = joblib.load(feat_path) if feat_path.exists() else fallback_feats
    model, err = None, None

    try:
        if json_path.exists():
            model = xgb.XGBClassifier()
            model.load_model(str(json_path))
        elif pkl_matches:
            model = joblib.load(pkl_matches[-1])
        else: err = "No model found in the models/ directory."
    except Exception as e: err = str(e)
    
    # Load Locations
    loc_path = DATA_DIR / "state_district_lookup.csv"
    st_dist, codes, asp = {}, {}, {}
    if loc_path.exists():
        loc = pd.read_csv(loc_path)
        st_dist = loc.groupby("state_name")["district_name"].apply(list).to_dict()
        codes = {(r.state_name, r.district_name): (r.state_code, r.district_code) for r in loc.itertuples()}
        asp = {(r.state_name, r.district_name): bool(r.is_aspirational) for r in loc.itertuples()}
    
    return model, feat_order, err, st_dist, codes, asp, not loc_path.exists()

model, FEATURE_ORDER, load_error, STATE_TO_DISTRICTS, CODES, ASPIRATIONAL, loc_err = load_assets()

def format_col(c):
    """Shortened feature namer for cleaner SHAP charts"""
    if c.startswith("freq_"): return c.replace("freq_", "").replace("_", " ").title()
    for p, n in zip(["residence_", "religion_", "caste_", "marital_"], ["Residence", "Religion", "Caste", "Marital Status"]):
        if c.startswith(p): return n
    return {"age":"Age", "total_children_born":"Children", "is_pregnant":"Pregnant", "is_aspirational":"Aspirational Dist"}.get(c, c.replace("_", " ").title())

# ==========================================================
# 2. Sidebar Navigation & Links
# ==========================================================
st.sidebar.title("🩸 Navigation")
page = st.sidebar.radio("Go to", ["📊 Project Overview & Insights", "🩺 Interactive Screening Tool", "🤖 AI Clinical Assistant (RAG)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Project Links")
st.sidebar.markdown("[GitHub Repository](https://github.com/Prathamyadav01/Anemia-Prediction)")
st.sidebar.markdown("[Jupyter Notebook](https://github.com/Prathamyadav01/Anemia-Prediction/blob/main/notebook/Project_Best_Model_.ipynb)")
st.sidebar.markdown("**Authors:** Pratham Yadav, Kushagra Sharma, Mayur Saini")

if load_error: st.sidebar.error(f"⚠️ Model load error: {load_error}")

# ==========================================================
# 3. Page 1: Project Overview & Insights
# ==========================================================
if page == "📊 Project Overview & Insights":
    st.title("🩸 Predicting Anemia Risk in India")
    
    st.markdown("""
    ### Executive Summary
    Anemia is a silent epidemic impacting maternal and child health. Clinical blood tests are often expensive and logistically difficult to deploy in highly underdeveloped areas. 
    
    This project leverages **XGBoost** trained on harmonized **NFHS-5 data** to provide a purely socioeconomic and dietary screening tool for community health workers. By predicting the probability that a woman aged 15–49 is anemic (hemoglobin < 11.0 g/dL), we can triage and prioritize patients effectively without requiring a single drop of blood.
    """)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Performance")
        st.markdown("""
        **Survey-weighted held-out test metrics:**
        *   **Cross-validated ROC-AUC:** 0.632
        *   **Optimized Decision Threshold:** 0.49
        *   **Methodology:** Categorical fields are explicitly one-hot encoded. The `is_aspirational` flag maps directly to NITI Aayog's 112 Aspirational Districts framework.
        
        *These numbers indicate modest, better-than-chance discrimination—ideal for flagging patients for clinical follow-ups.*
        """)
        
    with col2:
        st.subheader("🌍 Global Feature Importance")
        st.markdown("What factors drive the model's decision-making across *all* patients?")
        if model:
            try:
                raw = model.get_booster().get_score(importance_type="total_gain")
                df_glob = pd.DataFrame({"col": list(raw.keys()), "val": list(raw.values())})
                df_glob["grp"] = df_glob["col"].apply(format_col)
                grp_glob = df_glob.groupby("grp")["val"].sum().reset_index()
                grp_glob["val"] = grp_glob["val"] / grp_glob["val"].sum() * 100
                grp_glob = grp_glob.sort_values("val", ascending=True).tail(10)
                
                fig_g = go.Figure(go.Bar(x=grp_glob["val"], y=grp_glob["grp"], orientation="h", marker_color=PRIMARY))
                fig_g.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Global Importance (%)")
                st.plotly_chart(fig_g, use_container_width=True)
            except Exception:
                st.info("Global feature importance unavailable.")

# ==========================================================
# 4. Page 2: Interactive Screening Tool
# ==========================================================
elif page == "🩺 Interactive Screening Tool":
    st.title("📋 Patient Risk Screening Form")
    st.markdown("Enter the patient's socio-demographic and dietary survey responses below for an instant risk assessment.")
    
    # 4a. Location Lookup (Kept outside the form so it updates dynamically)
    st.subheader("📍 Geographic Location")
    if loc_err or not STATE_TO_DISTRICTS:
        st.warning("Location data missing; using defaults.")
        st_code, dist_code, is_asp = 10, 203, False
        loc_label = "Default Location"
    else:
        l_col1, l_col2, l_col3 = st.columns(3)
        with l_col1: st_sel = st.selectbox("State", sorted(STATE_TO_DISTRICTS.keys()))
        with l_col2: dist_sel = st.selectbox("District", sorted(STATE_TO_DISTRICTS[st_sel]))
        
        st_code, dist_code = CODES.get((st_sel, dist_sel), (0,0))
        is_asp = ASPIRATIONAL.get((st_sel, dist_sel), False)
        loc_label = f"{dist_sel}, {st_sel}"
        
        with l_col3:
            st.markdown("<br>", unsafe_allow_html=True) # alignment spacer
            st.caption(f"Aspirational District Status: **{'✅ Yes' if is_asp else '❌ No'}**")

    # 4b. The Batched Form
    with st.form("patient_form"):
        st.subheader("👤 Socio-Demographic Data")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.slider("Patient Age (Years)", 15, 49, 20)
            residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        with c2:
            is_preg = st.selectbox("Currently Pregnant?", ["No", "Yes"])
            wealth = st.selectbox("Household Wealth Index", ["Poorest", "Poorer", "Middle", "Richer", "Richest"], index=2)
        with c3:
            children = st.number_input("Total Children Born", 0, 15, 0)
            edu = st.selectbox("Highest Education", ["No Education", "Primary School", "Secondary School", "Higher Secondary/College"])

        st.markdown("---")
        st.subheader("🥦 Dietary Consumption Frequencies")
        diet_opts = {"Never": 0, "Occasional": 1, "Weekly": 2, "Daily": 3}
        
        d1, d2, d3 = st.columns(3)
        with d1:
            f_milk = st.selectbox("Milk or Curd", list(diet_opts.keys()), index=1)
            f_pulses = st.selectbox("Pulses or Beans", list(diet_opts.keys()), index=2)
            f_veg = st.selectbox("Dark Green Veg", list(diet_opts.keys()), index=2)
        with d2:
            f_fruits = st.selectbox("Fruits", list(diet_opts.keys()), index=1)
            f_eggs = st.selectbox("Eggs", list(diet_opts.keys()), index=0)
            f_fish = st.selectbox("Fish", list(diet_opts.keys()), index=0)
        with d3:
            f_meat = st.selectbox("Chicken or Meat", list(diet_opts.keys()), index=0)
            f_fried = st.selectbox("Fried Food", list(diet_opts.keys()), index=1)
            f_aerated = st.selectbox("Aerated Drinks", list(diet_opts.keys()), index=1)
            
        submit_button = st.form_submit_button("Analyze Patient Risk", use_container_width=True)

    # 4c. Prediction Logic & Results
    if submit_button:
        if not model:
            st.error("Cannot predict: Model failed to load.")
        else:
            # Assemble Features
            base_data = {
                "state_code": st_code, "district_code": dist_code, "age": age, 
                "wealth_index_national": ["Poorest", "Poorer", "Middle", "Richer", "Richest"].index(wealth)+1,
                "wealth_index_state": ["Poorest", "Poorer", "Middle", "Richer", "Richest"].index(wealth)+1,
                "education_level": ["No Education", "Primary School", "Secondary School", "Higher Secondary/College"].index(edu),
                "total_children_born": children, "is_pregnant": 1 if is_preg=="Yes" else 0,
                "is_aspirational": int(is_asp), "residence_type_1": int(residence=="Urban"), "residence_type_2": int(residence=="Rural"),
                "freq_milk_curd": diet_opts[f_milk], "freq_pulses_beans": diet_opts[f_pulses], "freq_green_leafy_veg": diet_opts[f_veg],
                "freq_fruits": diet_opts[f_fruits], "freq_eggs": diet_opts[f_eggs], "freq_fish": diet_opts[f_fish],
                "freq_chicken_meat": diet_opts[f_meat], "freq_fried_food": diet_opts[f_fried], "freq_aerated_drinks": diet_opts[f_aerated]
            }
            df_features = pd.DataFrame([base_data]).reindex(columns=FEATURE_ORDER, fill_value=0)
            
            # Explicit Type Casting (Robustness Fix)
            for col in df_features.columns:
                df_features[col] = df_features[col].astype(float)

            # Inference
            THRESHOLD = 0.49
            prob = float(model.predict_proba(df_features)[0,1])
            
            st.markdown("---")
            st.subheader("🎯 Diagnostic Evaluation Output")
            
            r_col1, r_col2 = st.columns([1, 1.2])
            
            with r_col1:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, number={"suffix":"%"}, gauge={
                    "axis": {"range": [0,100]}, "bar": {"color": RISK_HIGH if prob>=THRESHOLD else RISK_LOW},
                    "steps": [{"range": [0, THRESHOLD*100], "color": "#E9F5F0"}, {"range": [THRESHOLD*100, 100], "color": "#FBEAEA"}],
                    "threshold": {"line": {"color": "#333", "width": 3}, "value": THRESHOLD*100}}))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                if prob >= THRESHOLD:
                    st.markdown(f'<div class="risk-banner high">🚨 POSITIVE SCREEN DETECTED</div>', unsafe_allow_html=True)
                    st.caption(f"Score sits above the intervention risk floor ({THRESHOLD}). Prioritize for clinical blood test verification.")
                else:
                    st.markdown('<div class="risk-banner low">✅ LOW RISK DETECTED</div>', unsafe_allow_html=True)
                    st.caption("Tracks under intervention threshold. Maintain routine preventive supplementation.")
                    
            with r_col2:
                st.markdown("##### 🧬 What drove this patient's score?")
                try:
                    import shap
                    sv = shap.TreeExplainer(model).shap_values(df_features)
                    sv = sv[1] if isinstance(sv, list) and len(sv)>1 else (sv[0] if isinstance(sv, list) else sv)
                    df_shap = pd.DataFrame({"col": df_features.columns, "val": np.asarray(sv).reshape(-1)})
                    df_shap["grp"] = df_shap["col"].apply(format_col)
                    grp_shap = df_shap.groupby("grp")["val"].sum().reset_index()
                    grp_shap = grp_shap.reindex(grp_shap["val"].abs().sort_values(ascending=False).head(7).index).sort_values("val")
                    
                    fig_s = go.Figure(go.Bar(x=grp_shap["val"], y=grp_shap["grp"], orientation="h", 
                                             marker_color=[RISK_HIGH if v>0 else RISK_LOW for v in grp_shap["val"]]))
                    fig_s.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="SHAP Values (Red = Pushed Risk Up, Green = Pushed Down)")
                    st.plotly_chart(fig_s, use_container_width=True)
                except Exception: 
                    st.info("Install `shap` library to see patient-specific driver analysis.")
            
            st.markdown('<div class="disclaimer">⚕️ This tool produces a statistical screening estimate for triage/prioritization only. It is not a diagnosis. Confirm any positive screen with a laboratory hemoglobin test before clinical action.</div>', unsafe_allow_html=True)

# ==========================================================
# 5. Page 3: AI Clinical Assistant (RAG)
# ==========================================================
elif page == "🤖 AI Clinical Assistant (RAG)":
    st.title("🤖 Anemia Mukt Bharat Clinical Guidelines Assistant")
    st.caption("Ask questions about IFA dosages, referral protocols, or age-specific treatments.")

    @st.cache_resource(show_spinner="Loading knowledge base...")
    def init_rag_chain():
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.2,
            api_key=st.secrets["GOOGLE_API_KEY"]
        )

        system_prompt = (
            "You are an expert clinical assistant for community health workers (ASHAs) in India.\n"
            "Use the provided context from official health guidelines to answer the user's question.\n"
            "If you do not know the answer based on the text, state clearly that the guidelines do not mention it.\n"
            "Keep your response concise, clear, and actionable.\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    rag_chain = init_rag_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("e.g., What is the protocol for a teenager with severe anemia?"):
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching official guidelines..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})