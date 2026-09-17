import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import Counter
import json

# STEP 1: Load data
with open("merged_labeled_speeches.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# STEP 2: Extract year and decade
df["year"] = pd.to_datetime(df["date"]).dt.year
df["decade"] = (df["year"] // 10) * 10

# STEP 3: Explode multilabels
df = df.explode("labels")

# STEP 4: Compute most common label by decade, party, and state
grouped = df.groupby(["decade", "speaker_party", "speaker_state"])["labels"] \
            .agg(lambda x: Counter(x).most_common(1)[0][0]).reset_index()

# STEP 5: Get U.S. map (you'll need to map state abbreviations to full names)
state_abbrev_to_name = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia"
}
grouped["state_name"] = grouped["speaker_state"].map(state_abbrev_to_name)

us = us = gpd.read_file("/Users/abilopez/Downloads/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp")
us = us[us['admin'] == 'United States of America'] 

# STEP 6: Plot for each party and decade
for decade in grouped["decade"].unique():
    for party in grouped["speaker_party"].unique():
        subset = grouped[(grouped["decade"] == decade) & (grouped["speaker_party"] == party)]
        print("All states in shapefile:", us['name'].nunique())
        print("States with data in this subset:", subset['state_name'].nunique())
        print("States missing data:")
        print(set(us['name']) - set(subset['state_name']))
        merged = us.merge(subset, left_on="name", right_on="state_name", how="left")

        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
        merged.plot(column="labels", cmap="tab20", ax=ax, legend=True, edgecolor='black')
        ax.set_title(f"Top Speech Category per State ({party}) - {decade}s")
        ax.axis("off")
        plt.tight_layout()
        plt.show()