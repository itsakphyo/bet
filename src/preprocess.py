def preprocess_data(combined_df):
    """Preprocess data by selecting columns, adding totals, and filtering rows."""
    columns_to_keep = ["FTHG", "FTAG", "PSH", "PSD", "PSA"]
    keep_df = combined_df[columns_to_keep].copy()
    keep_df["Total"] = keep_df["FTAG"] + keep_df["FTHG"]
    drop_df = keep_df.dropna()
    df = drop_df[["PSH", "PSD", "PSA", "Total"]]
    df = df[df["Total"] <= 8]
    return df
