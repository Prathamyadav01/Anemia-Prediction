# Anemia Mukt Bharat — Predictive Screening & Clinical Decision Support

A machine learning project using **NFHS-4** and **NFHS-5** (National Family Health Survey) data to analyze and predict district-level "Aspirational District" status, as defined under the **NITI Aayog Aspirational Districts Programme**. Includes a trained model, a Streamlit web app with an AI clinical assistant (RAG chatbot), and a Tableau dashboard.

## 🔗 Links
- **App:** [Anemia_Mukt_Bharat_Website Link](https://anemia-prediction-yj52tfhnzhca6q8gshgg6x.streamlit.app)
- **Tableau Public Dashboard:** [NFHS-5 Predictive Anemia Vulnerability Dashboard](https://public.tableau.com/views/NFHS-5PredictiveAnemiaVulnerabilityDashboard/NFHS-5PredictiveAnemiaVulnerabilityDashboard?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 📁 Repository Structure
```
├── app.py                          # Main Streamlit application
├── ingest.py                       # One-time script: builds chroma_db from guideline PDFs
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── secrets.toml                # GOOGLE_API_KEY (not committed — see Setup)
├── chroma_db/                      # Vector store built by ingest.py (not committed)
├── guidelines/                     # Source clinical guideline PDFs for the RAG assistant
├── notebook/
│   └── model_training.ipynb        # Colab notebook — EDA & model training
├── models/
│   ├── xgboost_features_v1_20260729.pkl
│   ├── xgboost_anemia_optimized_20260729.pkl
│   └── xgboost_anemia_model.json         # Model config / metrics
├── data/
│   ├── nfhs4_clean.csv
│   ├── nfhs5_clean.csv
│   └── state_district_aspirational_status.csv
├── tableau/
│   └── dashboard.twbx              # optional — see tableau/README.md
├── .gitignore
└── README.md
```

## 📊 Data Sources
- **NFHS-4 & NFHS-5** — district-level health/demographic indicators (cleaned)
- **NITI Aayog Aspirational Districts list** — state/district aspirational-status labels
- **Tableau workbook data** — sourced from NITI Aayog (merged NFHS-5 + NITI Aayog Aspirational Districts extract, with geo codes)
- **Clinical guideline PDFs** (`guidelines/`) — source documents for the RAG-based AI Clinical Assistant, e.g. Anemia Mukt Bharat protocol, Control of Iron Deficiency Anaemia guidelines

## 🧠 Model
- Two-stage pipeline: **`xgboost_features_v1_20260729.pkl`** handles feature extraction/engineering, and **`xgboost_anemia_optimized_20260729.pkl`** performs the final prediction on the extracted features
- Model Selection: XGBoost (benchmarked against LightGBM, CatBoost, and Balanced Random Forest)
- Best Model Chosen: Tuned XGBoost (`RandomizedSearchCV`: max_depth=9, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9), selected for the most balanced recall across classes
- Key metrics (survey-weighted, optimal threshold=0.49):
  - ROC-AUC: **0.6327**
  - Macro F1: **0.5921**
  - Recall — Anemic (1): **0.64** | Non-Anemic (0): **0.55**
  - Aspirational Districts sub-population: Macro F1 **0.5644**, Anemic Recall **0.80**
- Config/metrics for both stages in `xgboost_anemia_model.json`

## 🤖 AI Clinical Assistant (RAG)
A chat tab in the app that answers questions about IFA dosages, referral protocols, and age-specific treatment using retrieval-augmented generation over the PDFs in `guidelines/`.
- Embeddings: HuggingFace `all-MiniLM-L6-v2`
- Vector store: Chroma (persisted to `chroma_db/`)
- LLM: Google Gemini via `langchain-google-genai`

## 🚀 Setup & Run
```bash
git clone https://github.com/Prathamyadav01/Anemia-Prediction.git
cd Anemia-Prediction

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Add your Gemini API key in `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your-api-key-here"
```

Build the vector store from the PDFs in `guidelines/` (run once, and again whenever those PDFs change):
```bash
python ingest.py
```

Run the app:
```bash
streamlit run app.py
```
