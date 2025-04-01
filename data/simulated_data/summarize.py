import os
import json
import pandas as pd

# Define ranges for beta_1 and dim_horizon
beta_1_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Example values, adjust as needed
dim_horizon_values = [5, 10]#[2, 5, 10]  # Example values, adjust as needed

# Initialize a list to store the table rows
results = []

# Iterate over beta_1 and dim_horizon values
for beta_1 in beta_1_values:
    for dim_horizon in dim_horizon_values:
        # Define the base path
        base_path = f"./linear/dgp_linear_beta_1_{beta_1:.1f}/models"

        # Define file names
        file_names = {
            "UNet": f"unetlstm_best_loss_dim_horizon_{dim_horizon}.txt",
            "GSTUNet": f"gstunet_best_loss_dim_horizon_{dim_horizon}.txt",
            "STCINet": f"stcinet_best_loss_dim_horizon_{dim_horizon}.txt",
            "IPWUNet": f"ipwunet_best_loss_dim_horizon_{dim_horizon}.txt",
        }

        # Iterate over the models
        for model_name, file_name in file_names.items():
            file_path = os.path.join(base_path, file_name)

            # Check if the file exists
            if os.path.exists(file_path):
                # Read the JSON file and extract "Denormalized Test MSE"
                with open(file_path, "r") as file:
                    try:
                        data = json.load(file)
                        test_mse = data.get("Denormalized Test RMSE", "")
                        test_se = data.get("Denormalized Test SD", "")
                    except json.JSONDecodeError:
                        test_mse = ""
            else:
                test_mse = None
                test_se = None

            # Format the test MSE and SE for display
            if test_mse is not None and test_se is not None:
                formatted_result = f"{test_mse:.3f} ± {test_se:.3f}"
            else:
                formatted_result = ""

            # Append the result to the table
            results.append({
                "dim_horizon": dim_horizon,
                "model_name": model_name,
                "beta_1": beta_1,
                "Denormalized Test RMSE": formatted_result
            })

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Pivot the table to have beta_1 as columns
pivot_df = df.pivot_table(
    index=["dim_horizon", "model_name"],
    columns="beta_1",
    values="Denormalized Test RMSE",
    aggfunc="first"  # Use the first value if duplicates exist
).reset_index()

# Sort by dim_horizon
model_order = ["UNet", "GSTUNet", "STCINet", "IPWUNet"] 
pivot_df["model_name"] = pd.Categorical(pivot_df["model_name"], categories=model_order, ordered=True)
pivot_df = pivot_df.sort_values(by=["dim_horizon", "model_name"])

pivot_df = pivot_df.sort_values(by="dim_horizon")

# Rename columns for clarity
pivot_df.columns.name = None  # Remove the name of the columns index
pivot_df = pivot_df.rename_axis(None, axis=1)  # Clean up axis labels

# Format float values to 3 decimal places
pivot_df = pivot_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else x)

# Save the pivoted DataFrame to a CSV file
output_path = "./linear/summary.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pivot_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Pivoted summary saved to {output_path}")