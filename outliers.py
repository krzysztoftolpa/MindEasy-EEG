def clean_outliers_with_iqr(df, columns):
    """
    Cleans the specified columns in the DataFrame by replacing outliers
    with the median of the column using the interquartile range (IQR) method.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to clean.

    Returns:
    pd.DataFrame: The DataFrame with cleaned columns.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        median = df[col].median()

        df[col] = df[col].apply(lambda x: median if x < lower_bound or x > upper_bound else x)

    return df
