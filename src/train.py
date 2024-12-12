from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

def train_model(df, config):
    """Train a CatBoost model."""
    y = df["Total"]
    X = df.drop(columns=["Total"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=config["catboost"]["iterations"],
        learning_rate=config["catboost"]["learning_rate"],
        depth=config["catboost"]["depth"],
        loss_function="RMSE",
        verbose=50
    )

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae
