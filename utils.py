import json


def balance_conditions(df, condition_col):
    """
    Subsets the DataFrame so that each category in the specified condition column has an equal number of rows.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    condition_col (str): The name of the column containing the condition categories.

    Returns:
    pd.DataFrame: The balanced DataFrame.
    """
    # Get the minimum count of rows across all unique values in the condition column
    min_count = df[condition_col].value_counts().min()

    # Sample the minimum count of rows for each unique value in the condition column
    balanced_df = df.groupby(condition_col).apply(lambda x: x.sample(min_count)).reset_index(drop=True)

    return balanced_df


def generate_eeg_metrics_json(file_path, values):
    # Check if the values array has exactly two elements
    if len(values) != 2:
        raise ValueError("The input array must contain exactly two values.")

    # Data to be written to JSON
    data = {
        "eeg_metrics": {
            "engagement": {
                "focus": values[1],
                "mind_wandering": values[0]
            }
        }
    }

    # Writing JSON data to a file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file created at: {file_path}")