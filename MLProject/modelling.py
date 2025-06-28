import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def main(n_estimators: int, max_depth: int, data_path: str):
    mlflow.sklearn.autolog()

    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    df = pd.read_csv(data_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print(f"Accuracy: {acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=257)
    parser.add_argument("--max_depth", type=int, default=25)
    parser.add_argument("--data_path", type=str, default="train_preprocessing.csv")
    args = parser.parse_args()

    main(args.n_estimators, args.max_depth, args.data_path)
