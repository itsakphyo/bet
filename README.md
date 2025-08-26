# Football Total Goals Predictor

A machine learning project that predicts the total number of goals in football matches using CatBoost regression and betting odds data. The model leverages historical match data from various European leagues to make accurate predictions for total goals scored.

## Features

- **CatBoost Regression Model**: Advanced gradient boosting for accurate goal predictions
- **Multi-League Support**: Trained on data from Premier League (E0), Championship (E1), League One (E2), League Two (E3), Bundesliga (D1), La Liga (SP1), Serie A (I1), Ligue 1, and Scottish Premier League (SC0)
- **Feature Engineering**: 16 sophisticated features derived from betting odds
- **Interactive Prediction**: Command-line interface for single match predictions
- **Probability Visualization**: Normal distribution plots showing prediction confidence
- **Configurable Parameters**: Easy model tuning through YAML configuration

## Model Performance

The model uses Mean Absolute Error (MAE) as the primary evaluation metric, providing reliable predictions with confidence intervals based on historical performance.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)

### Installation

1. **Clone the repository:**

2. **Install dependencies using Poetry:**

3. **Verify data files are present:**
   Ensure your `data/` folder contains CSV files with historical match data including betting odds.

### Training the Model

1. **Configure model parameters:**
   Edit `config.yaml` to adjust CatBoost hyperparameters:
   ```yaml
   catboost:
     iterations: 500
     learning_rate: 0.1
     depth: 4
   ```

2. **Train the model:**
   ```bash
   python src/main.py
   ```

3. **Check the output:**
   - Model will be saved to `models/catboost_model.pkl`
   - Training MAE will be displayed in the console

### Making Predictions

1. **Single match prediction:**
   ```bash
   python predict/single_predict.py
   ```

2. **Enter betting odds when prompted:**
   - Home win odds (PSH)
   - Draw odds (PSD)
   - Away win odds (PSA)

3. **View results:**
   - Predicted total goals
   - Probability distribution plot
   - Model confidence (MAE)

## Project Structure

```
bet/
├── config.yaml              # Model configuration
├── poetry.lock             # Dependency lock file
├── pyproject.toml          # Project dependencies and metadata
├── README.md               # Project documentation
├── data/                   # Historical match data
│   ├── E0.csv             # Premier League
│   ├── E1.csv             # Championship
│   ├── D1.csv             # Bundesliga
│   ├── SP1.csv            # La Liga
│   ├── I1.csv             # Serie A
│   └── ...                # Other league data
├── models/                 # Trained model storage
│   └── catboost_model.pkl # Pre-trained CatBoost model
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── main.py            # Main training script
│   ├── preprocess.py      # Data preprocessing utilities
│   ├── feature_engineering.py # Feature creation functions
│   └── train.py           # Model training logic
└── predict/                # Prediction utilities
    └── single_predict.py  # Interactive prediction script
```

## Data Requirements

Your CSV files should contain the following columns:
- `FTHG`: Full Time Home Goals
- `FTAG`: Full Time Away Goals  
- `PSH`: Pinnacle Sports Home win odds
- `PSD`: Pinnacle Sports Draw odds
- `PSA`: Pinnacle Sports Away win odds


## Feature Engineering

The model creates 16 engineered features from betting odds:

- **Basic combinations**: Sum, differences, and ratios of odds
- **Polynomial features**: Squared and cubed transformations
- **Geometric features**: Square roots and geometric means
- **Complex interactions**: Multi-way feature combinations

These features capture non-linear relationships between betting odds and goal outcomes.

## Model Details

- **Algorithm**: CatBoost Regressor
- **Target Variable**: Total goals per match (FTHG + FTAG)
- **Data Filtering**: Matches with ≤8 total goals for realistic predictions
- **Validation**: Train/Validation/Test split (60/20/20)
- **Evaluation**: Mean Absolute Error (MAE)

## Usage Examples

### Command Line Prediction
```bash
python predict/single_predict.py
# Enter Home win odds: 2.1
# Enter Draw odds: 3.4
# Enter Away win odds: 3.2
# Output: Predicted total goals with visualization
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

**Aung Khant Phyo** - [itsakphyo@gmail.com](mailto:itsakphyo@gmail.com)

Project Link: [https://github.com/itsakphyo/bet](https://github.com/itsakphyo/bet)

## Disclaimer

This project is for educational and research purposes only. Betting involves financial risk and should be done responsibly. The predictions are not guarantees and past performance does not indicate future results. Please gamble responsibly and within your means.
