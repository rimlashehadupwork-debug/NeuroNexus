import pandas as pd

# 1. Load your current data (assuming it has no header row)
# If your dataset already has other column names, remove `header=None`
df = pd.read_csv("parkinsons.csv", header=None)

# 2. Add the required column headers
# The platform requires at least 'name' and 'status' columns.
# We will create a mock 'name' column using the first ID column.
df.rename(columns={df.columns[0]: "name", df.columns[-1]: "status"}, inplace=True)

# Ensure 'name' is formatted as a string identifier if it's currently a number
df["name"] = "subject_" + df["name"].astype(str)

# 3. Save the correctly formatted CSV
df.to_csv("parkinsons_fixed.csv", index=False)
print("Fixed dataset saved as 'parkinsons_fixed.csv'")
