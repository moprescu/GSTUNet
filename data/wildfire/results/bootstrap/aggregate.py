import json
import os
import numpy as np
import pandas as pd

# Set directory containing the files (optional: set to '.' if in current directory)
data_dir = '.'

# Initialize dictionary to collect data
county_data = {}

# Read files and collect values
for i in range(1, 41):
    file_path = os.path.join(data_dir, f'gstunet_{i}.txt')
    with open(file_path, 'r') as f:
        data = json.load(f)
        for county, values in data.items():
            if county in ['additional_resp', 'train_epochs']:  # Skip the global total
                continue
            if county not in county_data:
                county_data[county] = {
                    'additional_resp': [],
                    'pop': values['pop']
                }
            county_data[county]['additional_resp'].append(values['additional_resp'])

# Compute statistics
summary = []
for county, info in county_data.items():
    resp_values = np.array(info['additional_resp'])
    mean = np.mean(resp_values)
    lower = np.percentile(resp_values, 2.5)
    upper = np.percentile(resp_values, 97.5)
    pop = info['pop']
    summary.append({
        'County': county,
        'Mean': round(mean, 2),
        '2.5%': round(lower, 2),
        '97.5%': round(upper, 2),
        'Population': pop
    })

# Save to CSV
df = pd.DataFrame(summary)
df = df.sort_values('County')
df.to_csv('gstunet_summary.csv', index=False)

print("Summary CSV saved as 'gstunet_summary.csv'.")

