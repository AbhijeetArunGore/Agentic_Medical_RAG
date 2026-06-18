import json

# Change mock data to trigger a safety failure
with open("C:\\Users\\abhis\\Desktop\\Agentic_Medical_RAG\\data\\patient_data.json", "r") as f:
    data = json.load(f)

data['notes'][0]['text'] += " Patient has a history of bilateral renal artery stenosis."

with open("C:\\Users\\abhis\\Desktop\\Agentic_Medical_RAG\\data\\patient_data.json", "w") as f:
    json.dump(data, f, indent=4)

print("Updated data with contraindication.")
