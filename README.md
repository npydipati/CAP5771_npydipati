**1.What are you trying to do? Articulate your objectives using absolutely no jargon.** <br>
We want to predict which countries and regions will struggle with electricity access in the next few years. <br>
We'll use information about their economy, population, and current infrastructure to identify patterns. <br>
The goal is to create a system that tells aid organizations and governments: "These are the places that need help most urgently" and "Here's what factors make the biggest difference." <br>

**2.How is it done today, and what are the limits of current practice?** <br>
**Current Approach:** <br>
Organizations track energy access through annual surveys <br>
Decisions are often reactive - helping after problems become severe <br>
Analysis is mostly descriptive (showing what happened) not predictive (forecasting what will happen) <br>
Different organizations use different data, making it hard to compare <br>
**Limits:** <br>
By the time data is collected and reported, it's already 1-2 years old <br>
No systematic way to predict which countries need help next <br>
Limited understanding of which factors matter most <br>
Resource allocation is often based on political factors rather than data-driven need <br>

## Data Sources

This project utilizes openly accessible global development data from the World Bank.

### Primary Source

All variables are obtained from the World Bank’s World Development Indicators (WDI) database.

- Organization: World Bank  
- Dataset: World Development Indicators (WDI)  
- Access Method: CSV download via World Bank API  
- Time Span: 1990–2023  
- Geographical Scope: Global (country-level)

The following indicators are extracted:

- Access to electricity (% of population)
- GDP per capita (current USD)
- Urban population (% of total)
- Rural population (% of total)
- Population density (people per sq. km)
- Total population
- Renewable energy consumption (% of total final energy consumption)
- Government Effectiveness Index

### Data Reproducibility

No proprietary data is used.

Raw CSV files are:

- Stored directly in the repository under `data/raw/`

This ensures full reproducibility of the data acquisition process.

### Derived Variables

The following variables are created during preprocessing:

- `risk_category` — Energy poverty classification (Severe / Moderate / Minimal)
- `prediction_year` — Forecast year
- `predicted_electricity_access` — Model-generated electricity access estimates

These fields are not part of the original dataset and are computed within the project workflow.


## Repository Structure

```
CAP5771_npydipati/
├── README.md                          
├── requirements.txt                   # Python dependencies with versions
├── data_dictionary.pdf                # Variable definitions and sources
├── database_schema.png                # SQLite database schema diagram
├── energy_poverty.db                  # SQLite database (raw data)
├── energy_poverty_prediction.ipynb    
│
├── data_wrangling.ipynb               # Data cleaning & feature engineering
├── data_modeling.ipynb                # Model training & evaluation
├── data_visualization_static.ipynb    # Static dashboard with widgets
│── .gitignore
├── data/
│   ├── raw/                           # Raw World Bank CSVs (8 indicators)
│   │   ├── 1.AccessToElectricityAPI_.../
│   │   ├── 2.GDPAPI_.../
│   │   ├── 3.UrbanAPI_.../
│   │   ├── 4.RuralAPI_.../
│   │   ├── 5.PopulationDensityAPI_.../
│   │   ├── 6.TotalPopulationAPI_.../
│   │   ├── 7.RenewableEnergyAPI_.../
│   │   ├── 8.GovernmentEffectivenessAPI_.../
│   │   └── integrated_dataset.csv
│   └── processed/
│       ├── analysis_ready.csv         # Final clean dataset (7,412 rows, 218 countries)
│       ├── train_data.csv             # Training set (1990–2017)
│       ├── validation_data.csv        # Validation set (2018–2020)
│       ├── test_data.csv              # Test set (2021–2023)
│       ├── test_predictions.csv       # Model predictions on test set
│       ├── feature_importance.csv     # Feature importance scores (best models)
│       ├── forecast_2024_2027.csv     # 2024–2027 baseline forecast
│       ├── merged_dataset.csv         # Merged raw dataset 
│       └── *_long.csv                 # Per-indicator reshaped datasets
│
└── diary/                             # Project diary 
    ├── problem_formulation.txt
    ├── data_acquisition (sources and relevance).txt
    ├── data_acquisition II(database storage).txt
    ├── data_exploration.txt
    ├── reflection_and_next_steps.txt
    ├── data_wrangling.txt
    ├── data_wrangling II.txt
    ├── data_modeling.txt
    ├── data_modeling II.txt
    └── data_visualization_static.txt
```

### Folders
- `data/raw/` — Raw datasets downloaded from World Bank WDI
- `data/processed/` — Processed datasets after preprocessing
- `diary/` — Log of project workflow, decisions, and observations

### Files
- `README.md` — This overview and instructions
- `requirements.txt` — Python packages and versions required
- `data_dictionary.pdf` — Detailed description of variables
- `energy_poverty_prediction.ipynb` — Jupyter notebook containing code and analysis
- `energy_poverty.db` — SQLite database containing all data sources
- `database_schema.png` — Visual diagram of the database schema showing tables, fields, and primary keys
- `data_wrangling.ipynb`
    Reproducible pipeline from raw CSVs to clean, analysis-ready dataset.
    - Loads and reshapes all 8 World Bank indicators from wide to long format
    - Removes non-country aggregates (income groups, regional totals)
    - Applies per-country linear interpolation for missing values
    - Engineers features: `risk_category`, `elec_access_change`, `gdp_growth`, `urban_change`
    - Runs data quality checks: impossible values, duplicate detection, data leakage check
    - **Output**: `data/processed/analysis_ready.csv` (7,412 rows, 15 columns, zero nulls)
 
- `data_modeling.ipynb`
    Trains and evaluates classification and regression models.
    - Loads `analysis_ready.csv` and validates schema
    - Applies 80/10/10 temporal split (Train: 1990–2017, Val: 2018–2020, Test: 2021–2023)
    - **Classification** (predicts `risk_category`: Severe / Moderate / Minimal):
    - RF Baseline: Val Accuracy **0.9235** ← best
    - RF Improved, XGBoost Baseline, XGBoost Improved compared
    - **Regression** (predicts `electricity_access` %):
    - RF Improved: Val R² **0.9185** ← best
    - RF Baseline, XGBoost Baseline, XGBoost Improved compared
    - Uses StratifiedKFold (classification) and TimeSeriesSplit (regression) for CV
    - **Outputs**: `test_predictions.csv`, `feature_importance.csv`, `forecast_2024_2027.csv`
 
- `data_visualization_static.ipynb`
    Static dashboard with interactive widgets built using `ipywidgets`.
    - **9 static views**: global trend, risk distribution, progress ranking, feature importance, correlation, model comparison, test evaluation, GDP scatter, 2024–2027 forecast
    - **3 interactive widgets**:
    - Widget A — Country Explorer (dropdown): select any country to see full trajectory + model predictions + forecast
    - Widget B — Global Year Snapshot (slider): drag to explore GDP vs electricity access for any year 1990–2023
    - Widget C — Risk Transition Explorer (range slider): compare risk category transitions between any two years
 

Instructions to Reproduce the Work
1. Clone the repository
- git clone https://github.com/npydipati/CAP5771_npydipati.git
- cd CAP5771_npydipati
2. Set up Python environment
```bash
    python3 -m venv venv

    # Activate virtual environment
    source venv/bin/activate        # Mac/Linux
    # OR
    venv\Scripts\activate           # Windows

    pip install --upgrade pip
    pip install -r requirements.txt
```
3. Open the Jupyter notebook
```bash
    jupyter notebook energy_poverty_prediction.ipynb
```
4. Run the wrangling notebook
```bash
jupyter notebook data_wrangling.ipynb
```
Run all cells top to bottom. This regenerates `data/processed/analysis_ready.csv` and all intermediate CSVs.
 
5. Run the modeling notebook
```bash
jupyter notebook data_modeling.ipynb
```
Run all cells top to bottom. This trains all models and saves `test_predictions.csv`, `feature_importance.csv`, and `forecast_2024_2027.csv` to `data/processed/`.
 
6. Run the dashboard notebook
```bash
jupyter notebook data_visualization_static.ipynb
```
Run all cells top to bottom. Interactive widgets appear inline in the notebook. Make sure `ipywidgets` is installed (included in `requirements.txt`).
 
> **Note**: The processed data files are already included in `data/processed/` so you can run the dashboard directly without re-running wrangling and modeling.
 

## Key Findings
 
1. **Global access improved dramatically** — average electricity access rose from ~72% (1990) to ~90% (2023), but ~700M people remain without access, concentrated in Sub-Saharan Africa.
 
2. **Renewable energy and GDP are the strongest predictors** — `renewable_energy_percent` and `gdp_per_capita` dominate feature importance in both models. Countries that invested in both simultaneously showed the fastest improvements.
 
3. **The model forecasts slow but continued progress to 2027** — 27 countries are projected to remain in Severe risk category by 2027 without major structural changes. The Moderate category is the hardest to predict (transition zone between improving and stagnating countries).

## Model Performance Summary
 
| Task | Model | Val Score | Test Score |
|------|-------|-----------|------------|
| Classification | RF Baseline | Accuracy 0.9235 | Accuracy 0.8746 |
| Regression | RF Improved | R² 0.9185 | R² 0.8636 |
