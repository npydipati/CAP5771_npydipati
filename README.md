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
