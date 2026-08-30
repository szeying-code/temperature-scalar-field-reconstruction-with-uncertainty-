import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "GlobalLandTemperaturesByCity.csv"
CHECK_DATE = "2003-07-01"

CITIES = [
    "Tacna",
    "Arica",
    "Puno",
    "Juliaca"
]

# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print("Loading original dataset...")

data = pd.read_csv(DATA_FILE)

print(f"Total rows in original dataset: {len(data)}")

# ============================================================
# FILTER THE DATE AND CITIES
# ============================================================

check = data[
    (data["dt"] == CHECK_DATE) &
    (data["City"].isin(CITIES))
].copy()

# ============================================================
# DISPLAY ORIGINAL VALUES
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL DATASET CHECK")
print("=" * 80)

if check.empty:
    print("No matching rows found.")
else:
    print(
        check[
            [
                "City",
                "Country",
                "Latitude",
                "Longitude",
                "AverageTemperature"
            ]
        ].to_string(index=False)
    )

# ============================================================
# CHECK EACH CITY INDIVIDUALLY
# ============================================================

print("\n" + "=" * 80)
print("CITY-BY-CITY CHECK")
print("=" * 80)

for city in CITIES:

    city_data = check[check["City"] == city]

    if city_data.empty:
        print(f"\n{city}: NOT FOUND")
    else:
        print(f"\n{city}:")
        print(
            city_data[
                [
                    "City",
                    "Country",
                    "Latitude",
                    "Longitude",
                    "AverageTemperature"
                ]
            ].to_string(index=False)
        )

# ============================================================
# CHECK WHETHER COORDINATES ARE IDENTICAL
# ============================================================

print("\n" + "=" * 80)
print("COORDINATE COMPARISON")
print("=" * 80)

pairs = [
    ("Tacna", "Arica"),
    ("Puno", "Juliaca")
]

for city1, city2 in pairs:

    row1 = check[check["City"] == city1]
    row2 = check[check["City"] == city2]

    print(f"\n{city1} vs {city2}")

    if row1.empty or row2.empty:
        print("Cannot compare — one or both cities were not found.")
        continue

    lat1 = row1.iloc[0]["Latitude"]
    lon1 = row1.iloc[0]["Longitude"]

    lat2 = row2.iloc[0]["Latitude"]
    lon2 = row2.iloc[0]["Longitude"]

    print(f"{city1}: Latitude = {lat1}, Longitude = {lon1}")
    print(f"{city2}: Latitude = {lat2}, Longitude = {lon2}")

    if lat1 == lat2 and lon1 == lon2:
        print(">>> IDENTICAL COORDINATES")
    else:
        print(">>> DIFFERENT COORDINATES")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for city1, city2 in pairs:

    row1 = check[check["City"] == city1]
    row2 = check[check["City"] == city2]

    if row1.empty or row2.empty:
        continue

    same_coordinates = (
        row1.iloc[0]["Latitude"] == row2.iloc[0]["Latitude"]
        and
        row1.iloc[0]["Longitude"] == row2.iloc[0]["Longitude"]
    )

    if same_coordinates:
        print(
            f"{city1} / {city2}: "
            "SAME coordinates in original dataset"
        )
    else:
        print(
            f"{city1} / {city2}: "
            "DIFFERENT coordinates in original dataset"
        )