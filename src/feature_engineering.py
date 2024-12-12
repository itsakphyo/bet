def add_new_features(df):
    """Add new features based on existing columns."""
    required_columns = ["PSH", "PSD", "PSA"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The DataFrame must contain the columns: {required_columns}")

    df['feature_1'] = df['PSH'] + df['PSD'] + df['PSA']
    df['feature_2'] = (df['PSH'] + df['PSA']) - df['PSD']
    df['feature_3'] = df['PSH'] * df['PSD'] / (df['PSA'] + 1e-6) 
    df['feature_4'] = df['PSH'] - df['PSD'] * df['PSA']
    df['feature_5'] = df['PSH'] ** 2 + df['PSD'] ** 2 - df['PSA'] ** 2
    df['feature_6'] = df['PSD'] / (df['PSH'] + 1e-6) + df['PSA']
    df['feature_7'] = df['PSA'] / (df['PSD'] + 1e-6) - df['PSH']
    df['feature_8'] = (df['PSH'] * df['PSD']) ** 0.5 + (df['PSA'] ** 0.5)
    df['feature_9'] = df['PSH'] * df['PSA'] + df['PSD'] ** 2
    df['feature_10'] = df['PSD'] ** 3 / (df['PSH'] + df['PSA'] + 1e-6)
    df['feature_11'] = (df['PSH'] - df['PSD']) / (df['PSA'] + 1e-6)
    df['feature_12'] = df['PSA'] ** 0.5 * df['PSH'] / (df['PSD'] + 1e-6)
    df['feature_13'] = (df['PSH'] * df['PSD'] * df['PSA']) ** (1/3)
    df['feature_14'] = df['PSH'] / (df['PSD'] + df['PSA'] + 1e-6) + df['PSD']
    df['feature_15'] = df['PSH'] ** 2 - df['PSD'] * df['PSA']
    df['feature_16'] = (df['PSH'] + df['PSD']) ** 2 - df['PSA'] ** 2
    
    return df