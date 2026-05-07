import pandas as pd
from pathlib import Path
import mlflow.sklearn
import os
import pytest
import numpy as np

def load_model():

    model_root = './src/app/models'
    model_root = Path(model_root)
    folders = [f for f in model_root.iterdir() if f.is_dir()]

    try:
        if not folders:
            raise FileNotFoundError("No experiment folders found in directory.")

        elif len(folders) == 1:
            model_path = os.path.join(model_root, folders[0].name)

        elif len(folders) > 1:
            import glob
            local_model_paths = glob.glob(model_root)
            model_path = max(local_model_paths, key=os.path.getmtime)

        else:
            raise ValueError(
                f"Multiple experiment folders found: {[f.name for f in folders]}. "
                "Please specify the experiment name."
            )

    except Exception as e:
        print(f"Error: {e}")

    model_file = os.path.join(model_path, 'artifacts')
    print(f"✅ Try to load model")

    try:
        model = mlflow.sklearn.load_model(model_file)
        print(f"✅ Model loaded successfully from {model_file}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {model_file}: {e}")


# ✅ Fixture to load model once
@pytest.fixture(scope="module")
def model():
    return load_model()


# ✅ Fixture for sample input
@pytest.fixture
def sample_input():
    return np.array([[10.5, 1, 30, 5, 200.0, 0]])


# ✅ 1. Model loads correctly
def test_model_loaded(model):
    assert model is not None


# ✅ 2. Prediction shape
def test_prediction_shape(model, sample_input):
    preds = model.predict(sample_input)
    assert len(preds) == 1


# ✅ 3. Prediction values (0 or 1)
def test_prediction_values(model, sample_input):
    preds = model.predict(sample_input)
    assert preds[0] in [0, 1]


# ✅ 4. Probability range
def test_probability_range(model, sample_input):
    probs = model.predict_proba(sample_input)
    prob_class_1 = probs[0][1]

    assert 0.0 <= prob_class_1 <= 1.0


# ✅ 5. Deterministic output
def test_deterministic_predictions(model, sample_input):
    p1 = model.predict(sample_input)
    p2 = model.predict(sample_input)

    np.testing.assert_array_equal(p1, p2)


# ✅ 6. Extreme values
def test_extreme_values(model):
    extreme_input = np.array([[9999, 0, 0, 0, 0.0, 10]])

    preds = model.predict(extreme_input)
    assert len(preds) == 1


# ✅ 7. Invalid shape should fail
def test_invalid_input_shape(model):
    bad_input = np.array([10.5, 1, 30])  # wrong shape

    with pytest.raises(Exception):
        model.predict(bad_input)


# ✅ 8. Business logic sanity check
def test_high_risk_customer(model):
    high_risk = np.array([[100, 0, 100, 1, 0.0, 0]])

    prob = model.predict_proba(high_risk)[0][1]

    assert prob > 0.3  # adjust threshold if needed

# ✅ 8. Business logic sanity check
def test_low_risk_customer(model):
    high_risk = np.array([[2, 1, 2, 10, 10000.0, 0]])

    prob = model.predict_proba(high_risk)[0][1]

    assert prob < 0.3  # adjust threshold if needed


def test_model_recall_with_custom_threshold(model):

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score

    df = pd.read_csv("data/features.csv")

    X = df.drop(columns=['churn', 'Customer ID'])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    probs = model.predict_proba(X_test)[:, 1]

    THRESHOLD = 0.35
    preds = (probs >= THRESHOLD).astype(int)

    # 6. Evaluate recall
    recall = recall_score(y_test, preds)

    # 7. Assert business requirement
    assert recall >= 0.80, f"Recall too low at threshold {THRESHOLD}: {recall:.3f}"

def test_break_ci():
    assert 1 == 2
