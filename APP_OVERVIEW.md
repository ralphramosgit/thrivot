# Affordability App - Project Overview

An interactive web app that helps users determine which U.S. cities they can afford to live in based on their income and personal budget.

---

## Project Description

This application allows users to input their financial profile and explore an interactive map of U.S. cities. When a user clicks on any city, the app instantly calculates whether they can afford to live there, displays a full monthly cost breakdown, and surfaces key considerations such as weather risk, crime, and job market conditions. A secondary recommendation dashboard uses machine learning to suggest the top 5 cities best matched to the user's profile.

The project is built on real datasets covering cost of living, housing rental prices, crime rates, and natural disaster risk across U.S. cities. Machine learning powers the affordability scoring and city recommendation engine.

---

## Tech Stack

### Frontend

- Next.js (React, TypeScript)
- Tailwind CSS
- Interactive map using react-simple-maps or Mapbox

### Backend

- Python, FastAPI
- Scikit-learn, XGBoost, Pandas, NumPy
- Model serialized as a .pkl file

### Infrastructure

- AWS Lambda (serverless compute for ML inference)
- AWS ECS Fargate (serverless dockerized FastAPI App (ML Model))
- AWS S3 (static city dataset + trained model storage)
- Vercel (frontend deployment)

---

## App Usage Flow

### Page 1 - Map Search

#### Step 1: User Inputs Background Profile (one-time form before interacting with the map)

- Salary or monthly income
- Max budget for housing (rent)
- Any other fixed monthly payments (loans, subscriptions, etc.)
- Transportation type: by car or by public commute
- Family size: just me / me and a partner / family with children

#### Step 2: User Clicks a City on the Map

A result panel slides open displaying two sections.

**Cost Info**

- Affordable or not (yes/no verdict)
- Affordability score (0 to 100)
- Predicted monthly surplus or deficit in dollars
- Average monthly housing cost
- Average monthly food spending
- Transportation cost (adjusted for car vs. commute preference)
- Utilities
- Miscellaneous and general cost of living (includes healthcare)
- Possible savings or spare money remaining
- Summary section:
  - Total monthly spending
  - Estimated state and local income tax

**Considerations**

- Weather conditions (flood risk, hurricane risk, wildfire risk, tornado risk - sourced from FEMA National Risk Index)
- Safety (violent crime rate and property crime rate - sourced from FBI UCR data)
- Unemployment rate (city or metro level)

---

### Page 2 - Recommendation Dashboard

#### Step 1: User Inputs Background Profile (same fields as above)

#### Step 2: Additional Preferences

- Filter out specific states
- Additional filters (to be defined - e.g. minimum safety score, maximum disaster risk)

#### Step 3: Top 5 Recommended Cities Output

The recommendation engine uses ML clustering to group cities into profiles and rank the top 5 cities most financially compatible with the user's input.

---

## Datasets

| Dataset                             | Source                 | Granularity  | Purpose                                         |
| ----------------------------------- | ---------------------- | ------------ | ----------------------------------------------- |
| US Cost of Living (MIT Living Wage) | Kaggle                 | County level | Food, transport, utilities, healthcare costs    |
| Zillow Observed Rent Index (ZORI)   | Zillow Research        | City level   | Rent prices by city                             |
| FBI Uniform Crime Reports           | Kaggle                 | City level   | Violent and property crime rates                |
| FEMA National Risk Index            | FEMA (direct download) | County level | Flood, wildfire, hurricane, tornado risk scores |
| US Cities Lat/Lng                   | Kaggle                 | City level   | City coordinates for map rendering              |

### Data Consolidation Strategy

All datasets are merged into one master CSV using a shared `county_state_key` (format: `cook_il`) as the join key. Cities are matched to their parent county so county-level cost data can be joined with city-level rent and crime data. The final master CSV has one row per city with all features as columns. This CSV is stored in AWS S3 and loaded by Lambda at inference time.

---

## Machine Learning Overview

### Problem Type

Regression - predicting a continuous output (monthly surplus or deficit in dollars) given a user's financial profile and a city's cost features.

### Target Variable

```
predicted_monthly_surplus = monthly_take_home_income - total_estimated_monthly_costs
```

A positive value means the user has money left over. A negative value means they cannot afford the city at their current income.

### Features (Model Inputs)

User-side inputs:

- Monthly take-home income (after estimated tax)
- Max rent budget
- Family size multiplier (1, 2, 3, or 4 people)
- Transportation type (car vs. commute - affects transport cost column used)

City-side features from the master dataset:

- Median rent for appropriate bedroom count (from Zillow, based on family size)
- Food cost (from MIT COL dataset, scaled by family size)
- Transportation cost (car or transit depending on user input)
- Utilities cost
- Healthcare and miscellaneous cost
- Violent crime rate
- FEMA composite risk score

### Model

Primary model: XGBoost Regressor

- Learns non-linear relationships between city cost features and affordability
- Captures interactions between variables (e.g. cheap rent but high utilities in certain regions)
- Trained once locally in Python, saved as a .pkl file, loaded by FastAPI on Lambda

Why not a simple if/else:
A basic comparison of salary vs. rent only answers one dimension. The ML model predicts the full financial picture across all cost factors simultaneously and outputs one number - the monthly surplus or deficit - which powers the affordability score, the yes/no verdict, and the recommendation ranking.

### Affordability Score (0 to 100)

Derived from the predicted surplus using a normalization function across all cities. A score of 50 means the user is breaking even. A score of 85 means strong financial breathing room. A score of 20 means the city is technically possible but financially risky.

### Recommendation Engine

The recommendation dashboard uses KMeans clustering to group cities into profiles (e.g. affordable with strong job market, cheap but limited opportunity, expensive but high wages). The top 5 recommended cities are those with the highest affordability scores within the cluster most compatible with the user's financial profile.

### Model Training Summary

1. Load master CSV
2. Engineer features (family size multipliers, tax estimates, transport column selection)
3. Define target: calculated monthly surplus based on a baseline profile
4. Train XGBoost regressor with cross-validation
5. Evaluate using RMSE and R-squared
6. Save model as model.pkl to be loaded by FastAPI

---

## Data Flow (End to End)

```
User fills profile form (Next.js frontend)
        |
        v
POST request to FastAPI endpoint (AWS Lambda)
        |
        v
Lambda loads model.pkl and master_cities.json from S3
        |
        v
Model runs prediction for user profile + selected city
        |
        v
Returns: affordability score, surplus/deficit, cost breakdown, considerations
        |
        v
Next.js displays result panel on the map
```

---

## Known Limitations (for research paper)

- Salary data uses median estimates - actual job offers may be 20 to 40 percent below median for entry-level roles
- Cost of living data is at the county level for food, utilities, and transport - intra-county variation is not captured
- The model is a snapshot in time - it does not forecast future rent increases or cost changes
- Tax estimates are simplified to state income tax rates - city-level taxes (e.g. New York City local tax) are not fully modeled
- Healthcare costs assume employer-sponsored coverage - self-employed or uninsured users will face higher actual costs
- Affordability does not account for quality of life factors such as proximity to family, cultural amenities, or school quality

---

## Research Context

This project is developed as part of a data science research program. The goal is to demonstrate applied machine learning on real-world socioeconomic data to produce a tool that is useful to everyday people making major life decisions. The application addresses the global cost of living crisis - where wages are not keeping pace with housing and living costs - by giving individuals data-driven visibility into where their income can realistically support their lifestyle.
