import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from scripts.config import Config


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.feature_sets = {}

    def _make_model(self):
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )

    def _clean_training_data(self, df):
        df = df.copy()

        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

        df = df.dropna(subset=["game_date", "market", "player_name", "actual_value", "line"])

        # Only remove truly identical rows — line movement and multiple snapshots are valid
        df = df.drop_duplicates()

        df = df.sort_values("game_date").reset_index(drop=True)

        return df

    # Columns that must never be model features (leakage or non-numeric metadata)
    _EXCLUDE_COLS = {
        "actual_value",   # target — leakage
        "hit",            # derived from actual_value — leakage
        "edge",           # derived from actual_value — leakage
        "market",
        "player_name",
        "game_date",
        "bet_type",
        "bookmaker",
        "defense_source",
        "timestamp",
        "last_update",
        "commence_time",
        "normalized_player_name",
        "merge_key",
        "american_odds",
        "odds",
        "price",
        "implied_prob",
    }

    def _get_feature_columns(self, market_df):
        # `line` is intentionally NOT excluded — it is a predictive feature
        candidate_cols = [c for c in market_df.columns if c not in self._EXCLUDE_COLS]

        numeric_features = []
        for col in candidate_cols:
            converted = pd.to_numeric(market_df[col], errors="coerce")
            if converted.notna().sum() > 0:
                numeric_features.append(col)

        return numeric_features

    def train_market_model(self, features_df, market):
        print("\n" + "=" * 60)
        print(f"Training model: {market}")
        print("=" * 60)

        market_df = features_df[
            features_df["market"] == market
        ].copy()

        market_df = self._clean_training_data(market_df)

        if len(market_df) < 100:
            print(f"Not enough data for {market}: {len(market_df)} samples")
            return None

        feature_cols = self._get_feature_columns(market_df)

        if not feature_cols:
            print(f"No usable numeric features for {market}")
            return None

        self.feature_sets[market] = feature_cols

        X = market_df[feature_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = pd.to_numeric(
            market_df["actual_value"],
            errors="coerce"
        )

        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]
        market_df = market_df.loc[valid_rows]

        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=feature_cols,
            index=X.index
        )

        print(f"Training samples: {len(X_imputed):,}")
        print(f"Features used: {len(feature_cols):,}")

        n_splits = 3

        if len(X_imputed) < 500:
            n_splits = 2

        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_imputed), start=1):
            print(f"\nFold {fold}/{n_splits}")

            X_train = X_imputed.iloc[train_idx]
            X_val = X_imputed.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = self._make_model()
            model.fit(X_train, y_train)

            preds = model.predict(X_val)

            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)

            lines = pd.to_numeric(
                market_df.iloc[val_idx]["line"],
                errors="coerce"
            )

            hit_rate = (
                ((preds > lines.values) == (y_val.values > lines.values)).mean()
                * 100
            )

            print(f"MAE: {mae:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"R2: {r2:.3f}")
            print(f"Line direction accuracy: {hit_rate:.1f}%")

            fold_metrics.append({
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "hit_rate": hit_rate,
            })

        avg_metrics = {
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "r2": float(np.mean([m["r2"] for m in fold_metrics])),
            "hit_rate": float(np.mean([m["hit_rate"] for m in fold_metrics])),
        }

        print("\nAverage performance")
        print(f"MAE: {avg_metrics['mae']:.3f}")
        print(f"RMSE: {avg_metrics['rmse']:.3f}")
        print(f"R2: {avg_metrics['r2']:.3f}")
        print(f"Line direction accuracy: {avg_metrics['hit_rate']:.1f}%")

        final_model = self._make_model()
        final_model.fit(X_imputed, y)

        # Use out-of-sample CV RMSE — in-sample residuals are too small and
        # produce artificially high confidence values in the predictor.
        residual_std = avg_metrics["rmse"]

        importance_df = self._get_feature_importance(final_model, feature_cols)

        if importance_df is not None:
            print(f"\nTop 15 features for {market}:")
            print(importance_df.head(15).to_string(index=False))
        else:
            print(f"\nFeature importances not available for {market}")

        model_info = {
            "model": final_model,
            "imputer": imputer,
            "features": feature_cols,
            "metrics": avg_metrics,
            "residual_std": residual_std,
            "feature_importance": importance_df.to_dict("records") if importance_df is not None else [],
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
            "model_type": "xgboost",
        }

        self.models[market] = model_info
        self.metrics[market] = avg_metrics

        return avg_metrics

    def _get_feature_importance(self, model, feature_cols):
        if not hasattr(model, "feature_importances_"):
            return None
        importances = model.feature_importances_
        if importances.sum() == 0:
            # HistGradientBoostingRegressor may return all-zeros; don't surface as real signal
            return None
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def train_all_models(self, training_data):
        print("\n" + "=" * 60)
        print("Training all models")
        print("=" * 60)

        if isinstance(training_data, (str, Path)):
            data = pd.read_csv(training_data)
            print(f"Loaded training data: {len(data):,} rows")
        elif isinstance(training_data, pd.DataFrame):
            data = training_data.copy()
            print(f"Using dataframe: {len(data):,} rows")
        else:
            raise ValueError("training_data must be a file path or DataFrame")

        data = self._clean_training_data(data)

        print("\nMarkets:")
        for market, count in data["market"].value_counts().items():
            print(f"  {market}: {count:,}")

        results = {}

        for market in sorted(data["market"].dropna().unique()):
            metrics = self.train_market_model(data, market)

            if metrics:
                results[market] = metrics

        self.save_models()

        print("\n" + "=" * 60)
        print("Training summary")
        print("=" * 60)

        if results:
            print(pd.DataFrame(results).T.to_string())

        print(f"\nModels trained: {len(self.models)}")

        return results

    def save_models(self):
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        print("\nSaving models...")

        for market, model_info in self.models.items():
            filename = Config.MODELS_DIR / f"{market}_model.pkl"

            with open(filename, "wb") as f:
                pickle.dump(model_info, f)

            print(
                f"Saved {market}: "
                f"{len(model_info['features'])} features -> {filename}"
            )

    @staticmethod
    def load_models():
        models = {}

        print("\nLoading models...")

        for file in Config.MODELS_DIR.glob("*_model.pkl"):
            market = file.stem.replace("_model", "")

            with open(file, "rb") as f:
                models[market] = pickle.load(f)

            feature_count = len(models[market].get("features", []))
            print(f"Loaded {market}: {feature_count} features")

        return models

    def compare_feature_importance_across_markets(self):
        if not self.models:
            print("No models trained yet")
            return

        for market, model_info in self.models.items():
            print("\n" + "=" * 60)
            print(market)
            print("=" * 60)

            importance_df = pd.DataFrame(
                model_info.get("feature_importance", [])
            )

            if not importance_df.empty:
                print(importance_df.head(15).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Train NBA player prop prediction models"
    )

    parser.add_argument(
        "--data-file",
        default="data/prepared_training_data.csv",
        help="Path to prepared training data"
    )

    parser.add_argument(
        "--compare-features",
        action="store_true",
        help="Print feature importance comparison"
    )

    args = parser.parse_args()

    data_file = Path(args.data_file)

    if not data_file.exists():
        print(f"Training data not found: {data_file}")
        print("Run data preparation first:")
        print("python3 -m scripts.data_preparation")
        return

    trainer = ModelTrainer()

    trainer.train_all_models(data_file)

    if args.compare_features:
        trainer.compare_feature_importance_across_markets()

    print("\nTraining complete")
    print(f"Models saved to: {Config.MODELS_DIR}")


if __name__ == "__main__":
    main()