import pandas as pd
import joblib
import yaml
import os
from preprocess import preprocess_data
from feature_engineering import add_new_features
from train import train_model
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor


def load_config():
    """Load configuration from YAML file."""
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    # Load data
    folder_path = config["data_folder"]
    dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
            dfs.append(df)

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Preprocess and add features
    processed_df = preprocess_data(combined_df)
    df = add_new_features(processed_df)

    print("Data Shape:", df.shape)

    # Training
    model, mae = train_model(df, config) #where should i save this mae like saving model, since i want to use it in the next step
    print(f"Training complete. Mean Absolute Error: {mae:.4f}")

    # Save model
    model_path = os.path.join(config["model_folder"], "catboost_model.pkl")
    model_data = {
        "model": model,
        "mae": mae
    }
    joblib.dump(model_data, model_path)
    print(f"Model and MAE saved to {model_path}")


if __name__ == "__main__":
    main()
