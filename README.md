# Aspirational Districts Prediction — NFHS Data Analysis

A machine learning project using **NFHS-4** and **NFHS-5** (National Family Health Survey) data to analyze and predict district-level "Aspirational District" status, as defined under the **NITI Aayog Aspirational Districts Programme**. Includes a trained model, a web app, and a Tableau dashboard.

## 🔗 Links
- **App (if deployed):** [add your deployed app link here]
- **Tableau Public Dashboard:** [add your Tableau Public link here]

## 📁 Repository Structure
```
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── model_training.ipynb        # Colab notebook — EDA & model training
├── models/
│   ├── model_1.pkl
│   ├── model_2.pkl
│   └── model_metadata.json         # Model config / metrics
├── data/
│   ├── nfhs4_clean.csv
│   ├── nfhs5_clean.csv
│   └── state_district_aspirational_status.csv
├── tableau/
│   └── dashboard.twbx              # optional — see note below
├── .gitignore
└── README.md
```

## 📊 Data Sources
- **NFHS-4 & NFHS-5** — district-level health/demographic indicators (cleaned)
- **NITI Aayog Aspirational Districts list** — state/district aspirational-status labels
- **Tableau workbook data** — sourced from NITI Aayog

## 🧠 Model
- Algorithm: [e.g. Random Forest / Logistic Regression / XGBoost]
- Key metrics: [accuracy / F1 / etc.]
- Two trained models saved as `.pkl`; config/metrics in `model_metadata.json`

## 🚀 Setup & Run
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

# Run whichever applies to your app:
python app.py
# or, if it's a Streamlit app:
streamlit run app.py
```

## 📌 Note on large files
The Tableau workbook, model files, or CSVs may be too large for a normal Git push (GitHub blocks files over 100MB). Check sizes before pushing:
```bash
ls -lh models/*.pkl data/*.csv tableau/*.twbx
```
If anything is close to/over 100MB, use Git LFS (see setup steps in the chat) — or, for the Tableau workbook, just link to your Tableau Public dashboard instead of committing the `.twbx`.

## 📄 License
[Add a license if you want, e.g. MIT]
