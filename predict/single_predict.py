import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.feature_engineering import add_new_features


def predict(input_odds, model_path="models/catboost_model.pkl"):
    """
    Predict total goals based on input odds using the pre-trained model.

    Args:
        input_odds (dict): Dictionary with keys 'PSH', 'PSD', 'PSA'.
        model_path (str): Path to the saved model file.

    Returns:
        tuple: Predicted total goals and Mean Absolute Error (MAE).
    """
    loaded_data = joblib.load(model_path)
    loaded_model = loaded_data["model"]
    loaded_mae = loaded_data["mae"]

    test_data = pd.DataFrame([input_odds])

    test_data = add_new_features(test_data)

    prediction = loaded_model.predict(test_data)
    return prediction[0], loaded_mae


def visualize_distribution(input_odds, predicted_value, mae):
    """
    Visualize the probability distribution of predicted goals with MAE and input details.

    Args:
        input_odds (dict): Dictionary with keys 'PSH', 'PSD', 'PSA'.
        predicted_value (float): The predicted total goals.
        mae (float): Mean Absolute Error (MAE).
    """
    x = np.linspace(0, predicted_value + 4 * mae, 1000)
    y = norm.pdf(x, predicted_value, mae)

    plt.figure(figsize=(10, 7))
    plt.plot(x, y, label="Prediction Distribution", color='blue')
    plt.fill_between(x, y, alpha=0.1, color='blue')
    plt.axvline(predicted_value, color='red', linestyle='dashed', label=f"Predicted Value {predicted_value:.3f}")

    text_str = (
        f"Input Odds:\n"
        f"Home Win (PSH): {input_odds['PSH']}\n"
        f"Draw (PSD): {input_odds['PSD']}\n"
        f"Away Win (PSA): {input_odds['PSA']}\n\n"
        f"Predicted Total Goals: {predicted_value:.3f}\n"
        f"Mean Absolute Error (MAE): {mae:.4f}"
    )
    plt.text(
        0.02, 0.95, text_str, fontsize=8, transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.title('Total Goal Distribution')
    plt.xlabel('Total Goals')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(axis="x")
    plt.show()


if __name__ == "__main__":
    input_odds = {
        "PSH": float(input("Enter Home win odds (PSH): ")),
        "PSD": float(input("Enter Draw odds (PSD): ")),
        "PSA": float(input("Enter Away win odds (PSA): "))
    }

    predicted_goals, mae = predict(input_odds)

    visualize_distribution(input_odds, predicted_goals, mae)