# Data Merge Guide

This document outlines the strategy and step-by-step plan for merging all five cleaned datasets into a single master city-level dataset used by the Thrivot ML model and API.

---

## Cleaned Datasets Summary

| File | Granularity | Rows | Join Key(s) |
|------|------------|------|-------------|
| `cost_of_living_us_cleaned.csv` | County × Family Size | ~22,001 | `county_state_key` |
| `crime_cleaned.csv` | City | 972 | `city_state_key` |
| `nri_weather_risk_cleaned.csv` | County | 3,144 | `county_state_key` |
| `uscities_latlon_cleaned.csv` | City | 28,338 | `city_state_key`, `county_state_key` |
| `zillow_zori_rent_cleaned.csv` | County × Month | ~206,746 | `county_state_key` |

---

## Key Decisions Before Merging

### 1. Zillow Rent — Temporal Aggregation
The Zillow ZORI data spans **2015-01-31 to 2026-04-30** (136 monthly snapshots per county).

**Decision:** Compute the **trailing 12-month average** (most recent 12 months: May 2025 – Apr 2026) to produce one `avg_rent` value per county. This smooths seasonal volatility and reflects current market conditions.

```python
latest = zillow[zillow['date'] >= '2025-05-01']
zillow_agg = latest.groupby('county_state_key')['zori'].mean().reset_index()
zillow_agg.rename(columns={'zori': 'avg_monthly_rent'}, inplace=True)
```

### 2. Cost of Living — Family Size Dimension
The cost of living dataset has **7 rows per county**, one per family profile:

| Code | Meaning |
|------|---------|
| `1p0c` | 1 adult, 0 children |
| `1p1c` | 1 adult, 1 child |
| `1p2c` | 1 adult, 2 children |
| `1p3c` | 1 adult, 3 children |
| `2p0c` | 2 adults, 0 children |
| `2p1c` | 2 adults, 1 child |
| `2p2c` | 2 adults, 2 children |

**Decision:** **Pivot wide** — one row per county with columns prefixed by family code (e.g., `1p0c_food_cost`, `2p2c_total_cost`). This keeps the master dataset flat and lets the API filter at runtime based on the user's family size input.

```python
col_cols = ['food_cost', 'transportation_cost', 'healthcare_cost',
            'other_necessities_cost', 'childcare_cost', 'taxes', 'total_cost']
pivoted = col_pivot(cost_of_living, index='county_state_key',
                    columns='family_member_count', values=col_cols)
# Flatten multi-index columns: 1p0c_food_cost, 2p2c_total_cost, etc.
```

### 3. Crime Data — Sparse Coverage
Crime data covers only **972 cities** out of 28k+ in the latlon file. All other cities will have `NaN` for crime columns — this is expected and should be left as-is (not filled). The model/API should handle missing crime gracefully (e.g., show "data unavailable").

---

## Merge Strategy

**Spine:** `uscities_latlon_cleaned.csv` — the city-level backbone with both `city_state_key` and `county_state_key`, enabling joins to both city-level and county-level datasets.

```
uscities_latlon
    └── LEFT JOIN crime          ON city_state_key
    └── LEFT JOIN nri_weather    ON county_state_key
    └── LEFT JOIN zillow_agg     ON county_state_key
    └── LEFT JOIN col_pivoted    ON county_state_key
```

All joins are **LEFT JOINs** to preserve every city in the latlon file even when county or crime data is missing.

---

## Step-by-Step Merge Plan (`data_merge.ipynb`)

### Step 1 — Imports & Load All Datasets
```python
import pandas as pd

cities    = pd.read_csv('../cleaned_data/uscities_latlon_cleaned.csv')
crime     = pd.read_csv('../cleaned_data/crime_cleaned.csv')
weather   = pd.read_csv('../cleaned_data/nri_weather_risk_cleaned.csv')
col       = pd.read_csv('../cleaned_data/cost_of_living_us_cleaned.csv')
zillow    = pd.read_csv('../cleaned_data/zillow_zori_rent_cleaned.csv')
```

### Step 2 — Aggregate Zillow to Trailing 12-Month Average
```python
zillow['date'] = pd.to_datetime(zillow['date'])
cutoff = zillow['date'].max() - pd.DateOffset(months=12)
zillow_agg = (
    zillow[zillow['date'] > cutoff]
    .groupby('county_state_key')['zori']
    .mean()
    .reset_index()
    .rename(columns={'zori': 'avg_monthly_rent'})
)
```

### Step 3 — Pivot Cost of Living Wide
```python
cost_cols = ['food_cost', 'transportation_cost', 'healthcare_cost',
             'other_necessities_cost', 'childcare_cost', 'taxes', 'total_cost']

col_pivot = col.pivot_table(
    index='county_state_key',
    columns='family_member_count',
    values=cost_cols
)
col_pivot.columns = [f'{fam}_{metric}' for metric, fam in col_pivot.columns]
col_pivot = col_pivot.reset_index()
```

### Step 4 — Build the Master Merge
```python
master = cities.copy()
master = master.merge(crime[['city_state_key', 'violent_crime_rate', 'property_crime_rate', 'population']],
                      on='city_state_key', how='left')
master = master.merge(weather.drop(columns=['state', 'county']),
                      on='county_state_key', how='left')
master = master.merge(zillow_agg, on='county_state_key', how='left')
master = master.merge(col_pivot, on='county_state_key', how='left')
```

### Step 5 — Validate & Save
```python
print(f"Master shape: {master.shape}")
print(f"Cities with rent data: {master['avg_monthly_rent'].notna().sum()}")
print(f"Cities with crime data: {master['violent_crime_rate'].notna().sum()}")
print(f"Cities with weather data: {master['hurricane_risk_score'].notna().sum()}")
print(f"Cities with CoL data: {master['1p0c_total_cost'].notna().sum()}")

master.to_csv('../cleaned_data/master_cities.csv', index=False)
print("Saved to cleaned_data/master_cities.csv")
```

---

## Expected Output Columns

| Column Group | Columns |
|---|---|
| **City Identity** | `city`, `state`, `county`, `lat`, `lng`, `city_state_key`, `county_state_key` |
| **Crime** | `population`, `violent_crime_rate`, `property_crime_rate` |
| **Weather Risk** | `hurricane_risk_score`, `wildfire_risk_score`, `tornado_risk_score`, `flood_risk_score`, `county_fips` |
| **Rent** | `avg_monthly_rent` |
| **Cost of Living (×7 family profiles)** | `{profile}_food_cost`, `{profile}_transportation_cost`, `{profile}_healthcare_cost`, `{profile}_other_necessities_cost`, `{profile}_childcare_cost`, `{profile}_taxes`, `{profile}_total_cost` |

Total cost of living columns: 7 metrics × 7 family profiles = **49 columns**

---

## Post-Merge Next Steps

1. **Exploratory Analysis** — check null rates per column, flag counties with no rent or CoL data
2. **Feature Engineering** — compute `affordability_score` from user income vs. `total_cost + avg_monthly_rent`
3. **ML Model Training** — use master dataset to train the recommendation clustering model (KMeans or similar)
4. **API Integration** — serialize master CSV to S3; FastAPI loads it on cold start and filters by user profile at runtime
5. **Housing Cost Override** — the app uses Zillow `avg_monthly_rent` for housing, not the `housing_cost` from the CoL dataset (MIT Living Wage data uses a different methodology); keep both but document which the API uses
