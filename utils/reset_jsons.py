import json
import glob
json_paths = glob.glob("projects/*.json")

for path in json_paths:
    with open(path, "r") as f:
        data = json.load(f)
    # Reset the JSON data to an empty dictionary
    data = {}
    with open(path, "w") as f:
        json.dump(data, f, indent=4)