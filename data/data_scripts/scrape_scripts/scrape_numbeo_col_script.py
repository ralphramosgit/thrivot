"""
Scrape actual Cost of Living PRICES for all US cities from Numbeo.

Strategy:
  1. Visit each of the 50 US state pages to collect every city URL with data.
  2. Visit each city detail page and scrape ~42 real dollar values
     (restaurants, groceries, rent, utilities, transport, etc.)

Output: raw_data/numbeo_us_cities_col_raw.csv
"""

import os
import csv
import time
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "https://www.numbeo.com"
STATE_URL_TPL = BASE + "/cost-of-living/admin1Unit?country=United+States&unit={state}"

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Seconds to wait between city-page requests (be respectful)
REQUEST_DELAY = 1.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "raw_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "numbeo_us_cities_col_raw.csv")

# ---------------------------------------------------------------------------
# Label → CSV column name mapping  (all ~42 line items)
# ---------------------------------------------------------------------------
LABEL_TO_COL = {
    # Restaurants
    "Meal at an Inexpensive Restaurant":                                  "meal_inexpensive_restaurant",
    "Meal for Two at a Mid-Range Restaurant (Three Courses, Without Drinks)": "meal_midrange_2people",
    "Combo Meal at McDonald's (or Equivalent Fast-Food Meal)":            "combo_meal_mcdonalds",
    "Domestic Draft Beer (1 Pint)":                                       "domestic_beer_draft_pint",
    "Imported Beer (12 oz Small Bottle)":                                 "imported_beer_bottle_restaurant",
    "Cappuccino (Regular Size)":                                          "cappuccino",
    "Soft Drink (Coca-Cola or Pepsi, 12 oz Small Bottle)":               "soft_drink",
    "Bottled Water (12 oz)":                                              "water_bottle_small",
    # Markets
    "Milk (Regular, 1 Liter)":                                           "milk_1l",
    "Fresh White Bread (1 lb Loaf)":                                     "bread_1lb",
    "White Rice (1 lb)":                                                  "rice_1lb",
    "Eggs (12, Large Size)":                                              "eggs_12",
    "Local Cheese (1 lb)":                                               "cheese_1lb",
    "Chicken Fillets (1 lb)":                                            "chicken_fillets_1lb",
    "Beef Round or Equivalent Back Leg Red Meat (1 lb)":                 "beef_1lb",
    "Apples (1 lb)":                                                      "apples_1lb",
    "Bananas (1 lb)":                                                     "bananas_1lb",
    "Oranges (1 lb)":                                                     "oranges_1lb",
    "Tomatoes (1 lb)":                                                    "tomatoes_1lb",
    "Potatoes (1 lb)":                                                    "potatoes_1lb",
    "Onions (1 lb)":                                                      "onions_1lb",
    "Lettuce (1 Head)":                                                   "lettuce_head",
    "Bottled Water (50 oz)":                                              "water_bottle_large",
    "Bottle of Wine (Mid-Range)":                                         "wine_bottle_midrange",
    "Domestic Beer (16.9 oz Bottle)":                                    "domestic_beer_bottle_market",
    "Imported Beer (12 oz Small Bottle)":                                 "imported_beer_bottle_market",
    "Cigarettes (Pack of 20, Marlboro)":                                  "cigarettes_marlboro_20",
    # Transportation
    "One-Way Ticket (Local Transport)":                                   "transport_one_way_ticket",
    "Monthly Public Transport Pass (Regular Price)":                      "transport_monthly_pass",
    "Taxi Start (Standard Tariff)":                                       "taxi_start",
    "Taxi 1 mile (Standard Tariff)":                                     "taxi_per_mile",
    "Taxi 1 Hour Waiting (Standard Tariff)":                             "taxi_1hr_wait",
    "Gasoline (1 Liter)":                                                 "gasoline_1l",
    "Volkswagen Golf 1.5 (or Equivalent New Compact Car)":               "car_vw_golf_new",
    "Toyota Corolla Sedan 1.6 (or Equivalent New Mid-Size Car)":         "car_toyota_corolla_new",
    # Utilities
    "Basic Utilities for 915 Square Feet Apartment (Electricity, Heating, Cooling, Water, Garbage)": "utilities_basic_monthly",
    "Mobile Phone Plan (Monthly, with Calls and 10GB+ Data)":            "mobile_phone_plan_monthly",
    "Broadband Internet (Unlimited Data, 60 Mbps or Higher)":            "internet_60mbps_monthly",
    # Sports & Leisure
    "Monthly Fitness Club Membership":                                    "gym_membership_monthly",
    "Tennis Court Rental (1 Hour, Weekend)":                              "tennis_court_1hr",
    "Cinema Ticket (International Release)":                              "cinema_ticket",
    # Childcare
    "Private Full-Day Preschool or Kindergarten, Monthly Fee per Child": "preschool_monthly",
    "International Primary School, Annual Tuition per Child":            "international_school_annual",
    # Clothing
    "Jeans (Levi's 501 or Similar)":                                     "jeans_levis",
    "Summer Dress in a Chain Store (e.g. Zara or H&M)":                 "summer_dress",
    "Nike Running Shoes (Mid-Range)":                                     "nike_running_shoes",
    "Men's Leather Business Shoes":                                       "leather_business_shoes",
    # Rent
    "1 Bedroom Apartment in City Centre":                                 "rent_1br_city_centre",
    "1 Bedroom Apartment Outside of City Centre":                         "rent_1br_outside_centre",
    "3 Bedroom Apartment in City Centre":                                 "rent_3br_city_centre",
    "3 Bedroom Apartment Outside of City Centre":                         "rent_3br_outside_centre",
    # Buy
    "Price per Square Feet to Buy Apartment in City Centre":             "buy_sqft_city_centre",
    "Price per Square Feet to Buy Apartment Outside of Centre":          "buy_sqft_outside_centre",
    # Salary
    "Average Monthly Net Salary (After Tax)":                            "avg_monthly_net_salary",
    "Annual Mortgage Interest Rate (20-Year Fixed, in %)":               "mortgage_rate_annual_pct",
}

CSV_COLUMNS = (
    ["city", "state", "city_url"]
    + list(LABEL_TO_COL.values())
    + ["scraped_at"]
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_soup(url: str, retries: int = 3, backoff: float = 4.0) -> BeautifulSoup:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts.")


# ---------------------------------------------------------------------------
# Step 1 – collect all city URLs from state pages
# ---------------------------------------------------------------------------

def get_city_urls_for_state(state: str) -> dict[str, str]:
    """
    Returns {city_url: state_name} for every city listed on a state page.
    """
    url = STATE_URL_TPL.format(state=state.replace(" ", "+"))
    try:
        soup = fetch_soup(url)
    except RuntimeError as exc:
        log.error("Skipping state %s: %s", state, exc)
        return {}

    cities = {}
    # "All Cities in X with Available Data" section contains plain <a> links.
    # Numbeo emits absolute hrefs (https://www.numbeo.com/cost-of-living/in/...).
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "numbeo.com/cost-of-living/in/" in href:
            # Normalise to absolute URL (strip any query string / fragment)
            full_url = href.split("?")[0].split("#")[0]
            if full_url not in cities:
                cities[full_url] = state
        elif href.startswith("/cost-of-living/in/"):
            full_url = BASE + href.split("?")[0].split("#")[0]
            if full_url not in cities:
                cities[full_url] = state
    return cities


def collect_all_city_urls() -> dict[str, str]:
    """Scrape all 50 state pages and return {city_url: state}."""
    all_cities: dict[str, str] = {}
    for i, state in enumerate(US_STATES, 1):
        log.info("[%d/%d] Collecting cities for %s", i, len(US_STATES), state)
        state_cities = get_city_urls_for_state(state)
        log.info("  → %d cities found", len(state_cities))
        all_cities.update(state_cities)
        time.sleep(1.0)
    log.info("Total unique city URLs collected: %d", len(all_cities))
    return all_cities


# ---------------------------------------------------------------------------
# Step 2 – scrape actual prices from each city page
# ---------------------------------------------------------------------------

def clean_price(raw: str) -> str:
    """Strip currency symbol, commas, whitespace. Return empty string if N/A."""
    val = raw.strip().lstrip("$").replace(",", "").strip()
    return val if val not in ("", "?", "N/A", "-") else ""


def parse_city_prices(soup: BeautifulSoup) -> dict[str, str]:
    """
    Parse all price tables on a city detail page.
    Returns a dict keyed by CSV column name.
    """
    prices: dict[str, str] = {col: "" for col in LABEL_TO_COL.values()}

    # Prices live in <table class="data_wide_table"> elements
    for table in soup.find_all("table", class_="data_wide_table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            label = tds[0].get_text(" ", strip=True)
            # Remove trailing "Edit" text that Numbeo injects
            label = label.replace(" Edit", "").strip()
            col_name = LABEL_TO_COL.get(label)
            if col_name:
                prices[col_name] = clean_price(tds[1].get_text(strip=True))

    return prices


def extract_city_state_from_page(soup: BeautifulSoup, fallback_state: str) -> tuple[str, str]:
    """Extract city and state from the page <h1> or <title>."""
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        # "Cost of Living in San Francisco, CA"  →  "San Francisco", "CA"
        prefix = "Cost of Living in "
        if text.startswith(prefix):
            location = text[len(prefix):]
            if "," in location:
                parts = location.rsplit(",", 1)
                return parts[0].strip(), parts[1].strip()
            return location.strip(), fallback_state
    return "", fallback_state


# ---------------------------------------------------------------------------
# Step 3 – persist results
# ---------------------------------------------------------------------------

def save_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info("Saved %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── 1. Collect city URLs ──────────────────────────────────────────────
    log.info("=== Phase 1: collecting city URLs from all state pages ===")
    city_map = collect_all_city_urls()   # {url: state}

    if not city_map:
        log.error("No cities found. Aborting.")
        return

    scraped_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    records: list[dict] = []
    errors: list[str] = []

    # ── 2. Scrape each city detail page ───────────────────────────────────
    log.info("=== Phase 2: scraping prices for %d cities ===", len(city_map))
    for idx, (city_url, state) in enumerate(city_map.items(), 1):
        log.info("[%d/%d] %s", idx, len(city_map), city_url)
        try:
            soup = fetch_soup(city_url)
        except RuntimeError as exc:
            log.error("  SKIP %s: %s", city_url, exc)
            errors.append(city_url)
            continue

        city_name, city_state = extract_city_state_from_page(soup, state)
        prices = parse_city_prices(soup)

        record = {
            "city": city_name,
            "state": city_state,
            "city_url": city_url,
            "scraped_at": scraped_at,
            **prices,
        }
        records.append(record)

        # Save incrementally every 25 cities so progress is never lost
        if idx % 25 == 0:
            save_csv(records, OUTPUT_FILE)
            log.info("  (checkpoint saved — %d records so far)", len(records))

        time.sleep(REQUEST_DELAY)

    # ── 3. Final save ─────────────────────────────────────────────────────
    save_csv(records, OUTPUT_FILE)

    log.info("=== Done ===")
    log.info("Cities scraped : %d", len(records))
    log.info("Errors / skipped: %d", len(errors))
    print(f"\nDone. {len(records)} cities saved to:\n  {OUTPUT_FILE}")
    if errors:
        print(f"  ({len(errors)} URLs failed — see logs above)")


if __name__ == "__main__":
    main()

