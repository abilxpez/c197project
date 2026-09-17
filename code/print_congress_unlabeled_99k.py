import json

# Replace with your actual file name
file_path = "newly_labeled_speeches.json"

# Load the data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Commented out to avoid full iteration
# print(f"Total number of speeches: {len(data)}")

# Print the fields from the first entry
if data:
    print("Fields in the first speech entry:")
    for key in data[0].keys():
        print(f"- {key}")
else:
    print("The file is empty.")
