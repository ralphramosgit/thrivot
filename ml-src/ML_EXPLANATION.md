# Thrivot Machine Learning Writeup

This document explains the two machine learning notebooks in plain language: what they do, what
every step produces, what each graph means, what files come out, and what to do next. No shorthand
left unexplained.

There are **two notebooks**, and they run in order:

1. `income_regression_model.ipynb` - predicts **how much a city costs per month**.
2. `income_classification_model.ipynb` - predicts **whether a user's income can afford a city** (a 0-100 score).

The first one must run before the second, because the second loads a file the first one saves.

---

## 1. Short answer: did anything go wrong

No. Every cell in both notebooks runs top to bottom and produces output. Together they save **four
files** that the FastAPI backend will load (listed in Section 6).

The results come out very strong (regression R² ≈ 0.99, classifier accuracy ≈ 0.98). That is not a
bug - it happens because cost of living is strongly determined by the inputs (geography and area
income), and the affordability label is built directly from income vs. cost. The numbers are real;
they just describe an easier problem than they first appear. Two honest caveats: crime and
population are only present for about 3% of rows, so those features contribute little.

---

## 2. What the two notebooks are trying to do

Thrivot tells a user which U.S. cities they can realistically afford. The ML core does two jobs.

**Job one - regression (`income_regression_model.ipynb`).** Predict how much it costs per month to
live in a given city for a given household type. This is a dollar number (e.g. $5,956/month) and is
a property of the **city**, not the user.

**Job two - classification (`income_classification_model.ipynb`).** Given a **user's income** and a
city, decide whether the household can afford it. This outputs a probability that becomes the 0-100
affordability score and the Affordable / Stretched / Unaffordable verdict shown on the map.

Two models exist because they answer two different questions: "what does this place cost" vs. "given
your wallet, can you live here."

---

## 3. The data

Both notebooks load one merged file: `../data/cleaned_data/master_data_clean.csv`. That master file
was built earlier by joining five public datasets on a shared county-and-state key:

- **MIT Living Wage** - food, transport, utilities, healthcare, etc. (county level, annual).
- **Zillow Observed Rent Index (ZORI)** - the actual observed rent by city.
- **FBI crime reports** - violent and property crime rate by city.
- **FEMA National Risk Index** - flood, wildfire, hurricane, tornado risk scores by county.
- **US cities latitude/longitude** - for the map and as location features.

After loading, the master file holds one row per city with all of these columns side by side.

---

## 4. The regression notebook, step by step

### Step 1. Imports

Loads pandas, numpy, matplotlib, scikit-learn, and xgboost. Setup only.

### Step 2. Build one unified cost-of-living target

The most important design decision. MIT's total cost is annual and contains a **modelled** housing
figure. We don't want a guessed housing number when Zillow gives us the real one. So for each
household profile:

```
cost_of_living_monthly = (MIT total cost − MIT housing cost) / 12  +  real Zillow monthly rent
```

In words: take everything MIT measures except its fake housing number, convert yearly to monthly,
and bolt on the **actual observed rent**. **Rent is therefore the single biggest component of the
target** - it is not missing, it is the core of the number this model predicts.

MIT has seven household profiles (combinations of adults + children): `1p0c`, `1p1c`, `1p2c`,
`1p3c`, `2p0c`, `2p1c`, `2p2c`. Each city has all seven, so every city becomes seven rows (long
format) that share the same crime/weather/location but differ in cost by household size.

### Step 3. Exploratory analysis & correlation

A two-panel figure: a correlation heatmap (which variables move together) and a histogram of the
cost target (clipped at the 99th percentile so a few ultra-expensive cities don't stretch the plot).
Key takeaway: `avg_monthly_rent` correlates almost perfectly with the target because it **is** part
of the target - which is exactly why rent is excluded from the model inputs later.

### Step 3b. Weather risk deep-dive (three graphs)

A focused look at the four FEMA risk scores:

1. **Weather-only correlation heatmap** - how the four risks relate to each other and to cost.
   Flood (≈0.23) and wildfire (≈0.14) have the strongest tie to cost; tornado is basically flat.
2. **Cost of living across weather-risk bands** - each risk split low→high into five bands, showing
   mean cost in each. Flood shows the clearest "higher risk → higher cost" climb.
3. **Distribution of each risk score** - one histogram per risk across all unique cities.

### Step 4. Features & target (leakage-free)

Inputs are **independent city characteristics only**: lat, lng, population, the two crime rates, the
four weather risks, area median family income, and the household profile (encoded 0-6). No cost
component (food, utilities, rent) is used as an input, because they are baked into the target -
using them would be circular reasoning. Crime/population exist for only ~3% of rows; XGBoost handles
the missing values natively, so the rows are kept.

### Step 5. City-grouped train/test split

Each city appears seven times (once per profile). A random split could leak a city into both train
and test. We split by **city group** so each city is entirely in one side. The output confirms zero
shared cities - the honest way to measure performance on genuinely new places.

### Step 6. Cross-validation

Uses **GroupKFold** so every fold is scored on cities it never trained on. The four bars sit at
≈0.988 with essentially zero variation - the model is stable, not lucky.

### Step 7. Train final model + training curve

Fits the production XGBoost regressor while recording RMSE at every boosting round for both train
and validation. The curve falls and flattens - the classic "model learning" graph.

### Step 8. Evaluate on unseen cities

- **R² ≈ 0.988** (explains ~99% of cost variation on cities it never saw)
- **RMSE ≈ $230/month** (typical error)
- **MAE ≈ $158/month** (average error, ~3% of the ~$5,938 median)

Two panels: predicted-vs-actual (dots hug the diagonal) and residuals (centered on zero, no bias).
Both axes are clipped at the 99th percentile so rare luxury cities don't distort the view.

### Step 10. Save the model

Saves **`model_cost.pkl`** (the regressor) and **`features_regression.json`** (the schema + metrics).
`model_cost.pkl` is loaded by the classification notebook and by the backend.

---

## 5. The classification notebook, step by step

### Step 1-2. Imports & rebuild the same cost target

Same data preparation as the regression notebook, producing the same monthly cost figure per
city-profile.

### Step 3. Load the cost model

Loads **`model_cost.pkl`** from the regression notebook (used only to display a predicted cost in
the demo). If the file is missing it trains a quick fallback, so the notebook still runs standalone -
but the proper flow is to run the regression notebook first.

### Step 4. Build a training set where income VARIES

This is the key idea. Each city-profile is replicated at **five sampled incomes** ($1,800-$16,000/mo,
log-uniform). The label is whether that specific income covers that city's real cost:
`affordable = (user_income_monthly >= cost_of_living_monthly)`. Because `user_income_monthly` is now
an explicit input, the model learns how affordability changes **with income** instead of memorising
one fixed answer per city.

### Step 5. Class balance

A bar chart confirming the labels are roughly evenly split (≈50/50). Balanced labels mean the
classifier can't cheat by always guessing one answer.

### Step 6. City-grouped split

Same city-grouped split as before, so whole cities are held out for testing.

### Step 7. Train the classifier + training curve

Fits an XGBoost classifier, recording **log-loss** at every boosting round for train and validation.
The falling curve is the "model learning" graph.

### Step 8. Evaluate - confusion matrix & ROC

- **Accuracy ≈ 0.98** (gets the yes/no right ~98% of the time)
- **ROC-AUC ≈ 0.998** (near-perfect ranking of affordable above unaffordable)

Two panels: the confusion matrix (huge correct diagonal, tiny errors) and the ROC curve (pinned to
the top-left corner).

### Step 9. Precision-Recall & score separation

The PR curve plus two overlaid histograms of the predicted score for truly-affordable vs.
truly-not-affordable rows. Cleanly separated humps = the model distinguishes the classes well.

### Step 10. What drives affordability ASIDE from income

User income obviously dominates affordability (it holds ~69% of the model's total importance), so a
plain importance chart is just one giant income bar that flattens everything else. This graph
**removes `user_income_monthly`** and re-plots the rest (weather risks highlighted in red), so the
secondary drivers become visible. The ranking: **family profile** (household size), then **area
median family income**, then geography (`lng`) - followed immediately by all four **weather risk**
scores. Combined, the weather risks hold ~6.5% of total importance and ~21% of the non-income
importance, which confirms it is meaningful to surface weather in the web app's considerations panel.

### Step 11. Proof it learned income - the income sweep (headline graph)

Fix one city and sweep income from low to high. The affordability score climbs from 0 to 100 and
crosses 50% **right at the city's true cost**. In the latest run the crossover landed within a few
dollars of the true cost - proof the model learned the income relationship rather than memorising.

### Step 12. Assess function & save

Defines `assess_city_income(feature_row, user_monthly_income)` - what the backend calls. It predicts
the city's cost (via `model_cost.pkl`), scores affordability using the user's income, and returns
the predicted cost, monthly surplus, the 0-100 score, and a verdict. Then it saves
**`model_afford_income.pkl`** (the classifier) and **`features_income.json`** (its schema + metrics).

---

## 6. What gets produced (the four files)

| File                       | Produced by             | Used by                           | Purpose                                |
| -------------------------- | ----------------------- | --------------------------------- | -------------------------------------- |
| `model_cost.pkl`           | regression notebook     | classification notebook + backend | predicts a city's monthly cost         |
| `features_regression.json` | regression notebook     | backend                           | schema + metrics for the cost model    |
| `model_afford_income.pkl`  | classification notebook | backend                           | predicts the 0-100 affordability score |
| `features_income.json`     | classification notebook | backend                           | schema + metrics for the classifier    |

These four files are the ML deliverables your FastAPI backend loads. Nothing else is needed at
inference time except the master CSV (for each city's features).

---

## 7. How to run it (what to actually do)

You only need to re-run when the data or code changes. To regenerate everything from scratch:

1. Open `income_regression_model.ipynb` → **Run All**. This saves `model_cost.pkl` and
   `features_regression.json`.
2. Open `income_classification_model.ipynb` → **Run All**. This loads `model_cost.pkl` and saves
   `model_afford_income.pkl` and `features_income.json`.

**Order matters**: regression first (it produces `model_cost.pkl`), classification second (it
consumes it). If you only changed graphs or markdown, you don't need to re-run anything - the saved
`.pkl`/`.json` files are already current.

---

## 8. How the app uses the models

- The user enters their monthly income. The app already knows each city's features.
- `model_cost.pkl` predicts what the selected city costs per month.
- `model_afford_income.pkl` turns the user's income + city features into a 0-100 affordability score.
- Verdict thresholds: **≥70 Affordable**, **40-69 Stretched**, **<40 Unaffordable**.
- The predicted cost also lets the app show the monthly surplus/deficit and rank cities on the map.

---

## 9. Next steps (toward the frontend)

The ML side is done and exporting the four files. To connect it to the frontend you described
(Next.js map + FastAPI backend), the next pieces are:

1. **FastAPI backend** - load the four files once at startup; expose an endpoint like
   `POST /assess` that accepts `{ city_state_key, user_monthly_income, family_profile }` and returns
   `{ predicted_cost, monthly_surplus, affordability_score, verdict }` by calling the same
   `assess_city_income` logic from the classification notebook.
2. **City features endpoint** - serve each city's row (cost, crime, weather risks, lat/lng) so the
   map can render markers and the result panel can show the considerations section.
3. **Frontend (Next.js)** - the profile form, the interactive map, and the result panel that calls
   `/assess` when a city is clicked.
4. **(Later) Recommendation engine** - the Page 2 "top 5 cities" feature (KMeans clustering) is not
   built yet; it can be a third notebook/model when you get to it.

For now, you can start building the frontend against a mocked `/assess` response shaped like the
dictionary `assess_city_income` returns, then swap in the real FastAPI endpoint.

---

## 10. Graphs quick reference

**Regression notebook**

- Correlation heatmap - rent is part of the target (so it's excluded); which variables move together.
- Cost distribution histogram - most cities near the ~$5,956 median, clipped tail.
- Weather deep-dive (3 graphs) - risk correlation, cost-by-risk-band, risk distributions.
- Cross-validation bars - four folds near 0.988, proving stability.
- Training curve - RMSE falls and flattens.
- Predicted-vs-actual + residuals - dots hug the diagonal, errors centered on zero.

**Classification notebook**

- Class balance - labels roughly 50/50.
- Training curve - log-loss falls.
- Confusion matrix + ROC - tiny errors, near-perfect ranking.
- Precision-Recall + score separation - cleanly separated classes.
- Drivers aside from income - importance with income removed; family profile, area income, geography, then the four weather risks (~21% of non-income importance).
- Income sweep - score crosses 50% at the city's true cost (the headline graph).
