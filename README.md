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
├── requirements.txt           # Project dependencies
├── data_dictionary.pdf        # Variable definitions and data dictionary
├── energy_poverty_prediction.ipynb  # Main notebook
├── energy_poverty.db          # SQLite database
├── database_schema.png        # Database schema visualization
├── data/
│   ├── raw/                   # Raw CSV files from World Bank WDI
│   └── processed/             # Processed datasets
├── diary/                     # Project diary/log
│   ├── problem_formulation.txt
│   ├── data_acquisition__sources_and_relevance_.txt
│   ├── data_acquisition_II_database_storage_.txt
│   ├── data_exploration.txt
│   └── reflection_and_next_steps.txt
└── .gitignore
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
4. Run notebook cells sequentially
- Load raw CSVs from data/raw/
- Inspect dataset structure and missing values (.info(), .describe(), .isnull().sum())
- Preprocess data and create derived variables (risk_category, prediction_year, predicted_electricity_access)
5. Outputs
- Tables, figures appear directly in the notebook
