import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

def main(n_estimators: int, data_path: str):
    mlflow.sklearn.autolog()

    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    df = pd.read_csv(data_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Simpan model sebagai MLflow model (flavor sklearn)
        mlflow.sklearn.log_model(model, "model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="train_preprocessing.csv")
    args = parser.parse_args()

    main(args.n_estimators, args.data_path)
