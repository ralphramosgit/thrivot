import pandas as pd
import os

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(r"c:\Users\ralph\projects\thrivot\raw datasets\cost_of_living_us.csv")

print(f"Shape before cleaning: {df.shape}")

# ── 1. Drop duplicate rows ────────────────────────────────────────────────────
df = df.drop_duplicates()

# ── 2. Drop rows with null county, state, or areaname ────────────────────────
df = df.dropna(subset=["county", "state", "areaname"])

# ── 3. Standardize state to uppercase 2-letter abbreviations ─────────────────
df["state"] = df["state"].astype(str).str.strip().str.upper()

# ── 4. Standardize county column ──────────────────────────────────────────────
#    lowercase → strip whitespace → remove trailing " county"
df["county"] = (
    df["county"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s*county\s*$", "", regex=True)
    .str.strip()
)

# ── 5. Create county_state_key ────────────────────────────────────────────────
df["county_state_key"] = (
    df["county"].str.replace(r"\s+", "_", regex=True)
    + "_"
    + df["state"].str.lower()
)

# ── 6. Keep only rows where total family size (parents + children) is 1–4 ─────
#    family_member_count is formatted as e.g. "1p0c", "2p3c"
#    Parse parents and children from the string and sum them.
def parse_family_size(val):
    val = str(val).strip().lower()
    try:
        parents = int(val.split("p")[0])
        children = int(val.split("p")[1].replace("c", ""))
        return parents + children
    except (IndexError, ValueError):
        return None

df["family_size"] = df["family_member_count"].apply(parse_family_size)
df = df[df["family_size"].isin([1, 2, 3, 4])].drop(columns=["family_size"])

# ── 7. Convert cost columns to numeric ───────────────────────────────────────
cost_cols = [
    "housing_cost",
    "food_cost",
    "transportation_cost",
    "healthcare_cost",
    "other_necessities_cost",
    "childcare_cost",
    "taxes",
    "total_cost",
    "median_family_income",
]
for col in cost_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 8. Drop rows where total_cost is null or zero ────────────────────────────
df = df[df["total_cost"].notna() & (df["total_cost"] != 0)]

print(f"Shape after cleaning:  {df.shape}")

# ── 9. Save ───────────────────────────────────────────────────────────────────
output_path = r"c:\Users\ralph\projects\thrivot\data\cleaned_data\cost_of_living_us_cleaned.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to: {output_path}")
