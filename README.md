# 🍔 Big Mac Index — Advanced Analytics Dashboard

An interactive Streamlit dashboard performing **Descriptive, Diagnostic, Predictive & Prescriptive** analytics on The Economist's Big Mac Index data (2001–2025) across 55 countries.

## 📊 Seven Research Objectives

1. **Currency Misvaluation Assessment** — PPP-based over/undervaluation across countries
2. **Inflation & Big Mac Price Dynamics** — CPI-driven price change analysis
3. **Real Purchasing Power Comparison** — Inflation-adjusted cross-country analysis
4. **Price Convergence Testing** — Sigma & beta convergence in global markets
5. **GDP Impact on Adjusted Index** — GDP per capita vs. currency valuation
6. **Exchange Rate Influence** — FX movements and USD Big Mac price correlation
7. **Raw vs. GDP-Adjusted Misvaluation** — Structural economic difference identification

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/<your-username>/bigmac-dashboard.git
cd bigmac-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

## 📁 Project Structure

```
bigmac-dashboard/
├── app.py                 # Main Streamlit application
├── big_mac_v2.csv         # Dataset
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 📦 Data

Source: The Economist's Big Mac Index (2001–2025), enriched with World Bank CPI and GDP per capita data.

---

*Built as part of the SP Jain School of Global Management — MBA Applied Research Project*
