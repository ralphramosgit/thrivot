import pandas as pd
import numpy as np

# -- Load all cleaned datasets -------------------------------------------------
cities  = pd.read_csv('../cleaned_data/uscities_latlon_cleaned.csv')
crime   = pd.read_csv('../cleaned_data/crime_cleaned.csv')
weather = pd.read_csv('../cleaned_data/nri_weather_risk_cleaned.csv')
col     = pd.read_csv('../cleaned_data/cost_of_living_us_cleaned.csv')
zillow  = pd.read_csv('../cleaned_data/zillow_zori_rent_cleaned.csv')

print("Loaded datasets:")
print(f"  cities   : {cities.shape}")
print(f"  crime    : {crime.shape}")
print(f"  weather  : {weather.shape}")
print(f"  col      : {col.shape}  ← 7 rows per county (one per family profile)")
print(f"  zillow   : {zillow.shape}  ← 136 monthly rows per county")

# -- Step 1: Aggregate Zillow → 1 row per county (trailing 12-month avg rent) --
# Zillow has 136 monthly rows per county (Jan 2015 – Apr 2026).
# Average the most recent 12 months to reflect current market and smooth seasonality.
zillow['date'] = pd.to_datetime(zillow['date'])
cutoff = zillow['date'].max() - pd.DateOffset(months=12)
zillow_agg = (
    zillow[zillow['date'] > cutoff]
    .groupby('county_state_key')['zori']
    .mean()
    .reset_index()
    .rename(columns={'zori': 'avg_monthly_rent'})
)
zillow_agg['avg_monthly_rent'] = zillow_agg['avg_monthly_rent'].round(2)

print(f"\nZillow aggregated: {zillow_agg.shape[0]} counties")
print(f"Date window used : {cutoff.date()} → {zillow['date'].max().date()}")

# -- Step 2: Pivot Cost of Living wide → 1 row per county ---------------------
# Each county has 7 rows (one per family profile). Pivot so each profile becomes
# a column prefix e.g. 1p0c_food_cost, 2p2c_total_cost → 56 cost cols total.
#
#   1p0c = 1 adult 0 children  |  2p0c = 2 adults 0 children
#   1p1c = 1 adult 1 child     |  2p1c = 2 adults 1 child
#   1p2c = 1 adult 2 children  |  2p2c = 2 adults 2 children
#   1p3c = 1 adult 3 children

cost_metrics = [
    'housing_cost',           # MIT living wage estimate (reference only; API uses Zillow avg_monthly_rent)
    'food_cost',
    'transportation_cost',
    'healthcare_cost',
    'other_necessities_cost',
    'childcare_cost',
    'taxes',
    'total_cost',
]
# PivtO; rows = county, columns =
col_pivot = col.pivot_table(
    index='county_state_key',
    columns='family_member_count',
    values=cost_metrics,
    aggfunc='first'
)

# Flatten multiindex columns: e.g. (housing_cost, 1p0c) → 1p0c_housing_cost
col_pivot.columns = [f'{profile}_{metric}' for metric, profile in col_pivot.columns]
col_pivot = col_pivot.reset_index()

# Attach median_family_income as a single county-level column
col_income = col.groupby('county_state_key')['median_family_income'].first().reset_index()
col_pivot = col_pivot.merge(col_income, on='county_state_key', how='left')

print(f"\nCoL pivoted: {col_pivot.shape[0]} counties, {col_pivot.shape[1]} columns")

# -- Step 3: Build master city-level dataset -----------------------------------
# Spine: uscities_latlon — every city with both city_state_key and county_state_key.
# Multiple cities in the same county inherit the same county-level data.
# All joins are LEFT so no cities are dropped for missing data.

master = cities.copy()

# 1) Crime — city level (sparse: ~972 of 28k cities; rest NaN)
master = master.merge(
    crime[['city_state_key', 'population', 'violent_crime_rate', 'property_crime_rate']],
    on='city_state_key',
    how='left'
)

# 2) Weather risk — county level (all cities in same county share scores)
master = master.merge(
    weather[['county_state_key', 'county_fips',
             'hurricane_risk_score', 'wildfire_risk_score',
             'tornado_risk_score', 'flood_risk_score']],
    on='county_state_key',
    how='left'
)

# 3) Rent — county level (avg_monthly_rent from Zillow, trailing 12 months)
master = master.merge(
    zillow_agg[['county_state_key', 'avg_monthly_rent']],
    on='county_state_key',
    how='left'
)

# 4) Cost of living — county level, pivoted wide (56 cost cols + median_family_income)
master = master.merge(
    col_pivot,
    on='county_state_key',
    how='left'
)

print(f"\nMaster shape: {master.shape}")

# -- Step 4: Validate coverage & save -----------------------------------------
total = len(master)

print("=" * 55)
print(f"  Master dataset: {total:,} cities")
print("=" * 55)

coverage = {
    'avg_monthly_rent (Zillow)'   : 'avg_monthly_rent',
    'violent_crime_rate (FBI)'    : 'violent_crime_rate',
    'hurricane_risk_score (FEMA)' : 'hurricane_risk_score',
    '1p0c_total_cost (MIT CoL)'   : '1p0c_total_cost',
    'lat/lng'                     : 'lat',
}
for label, col_name in coverage.items():
    count = master[col_name].notna().sum()
    print(f"  {label:<35} {count:>6,}  ({count / total * 100:.1f}%)")

dupes = master.duplicated(subset='city_state_key').sum()
print(f"\n  Duplicate city_state_key rows : {dupes}")

# Sanity check: show cities sharing a county have the same rent
sample_county = master[master['county_state_key'].notna()].groupby('county_state_key').filter(lambda g: len(g) > 1)
if not sample_county.empty:
    example = sample_county['county_state_key'].iloc[0]
    print(f"\n  Example — cities in county '{example}':")
    print(sample_county[sample_county['county_state_key'] == example][
        ['city', 'county', 'avg_monthly_rent', 'violent_crime_rate']
    ].to_string(index=False))

# Save
out_path = '../cleaned_data/master_data.csv'
master.to_csv(out_path, index=False)
print(f"\n  Saved → {out_path}")
print(f"  Rows    : {master.shape[0]:,}")
print(f"  Columns : {master.shape[1]}")
