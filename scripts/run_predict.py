import os
import pandas as pd
import mlflow
import argparse
from pathlib import Path
from pydantic import BaseModel, validator

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    """

    model_root = args.model_root
    model_root = Path(model_root)
    model_path = Path(model_root)

    if not model_root.exists():
        raise FileNotFoundError(f"The base path does not exist: {model_root}")

    # List all folders
    folders = [f for f in model_root.iterdir() if f.is_dir()]

    try:
        if not folders:
            raise FileNotFoundError("No experiment folders found in directory.")

        if args.experiment_name:
            # Try to find a folder matching the name
            matched = [f for f in folders if f.name == args.experiment_name]
            if matched:
                model_path = matched[0]
            else:
                raise ValueError(f"No folder matches the experiment name: {args.experiment_name}")

        elif len(folders) == 1:
            model_path = os.path.join(model_root, folders[0].name)

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
        model = mlflow.pyfunc.load_model(model_file)
        print(f"✅ Model loaded successfully from {model_file}")
    except Exception as e:
        print(f"❌ Failed to load model from {model_file}: {e}")

    print(f"✅ Ask for input data")

    keys = ["avg_days_between", "has_multiple_purchases", "Recency", "Frequency", "Monetary", "returns"]

    class InputData(BaseModel):
        avg_days_between: float
        has_multiple_purchases: int
        Recency: int
        Frequency: int
        Monetary: float
        returns: int

        # Validator for feature2
        @validator('has_multiple_purchases')
        def feature2_must_be_0_or_1(cls, v):
            if v not in (0, 1):
                raise ValueError("has_multiple_purchases must be either 0 or 1")
            return v

    while True:
        user_input = input(f"Enter 6 numbers separated by spaces for {keys}: ")
        try:
            values = [float(x) for x in user_input.strip().split()]
            if len(values) != len(keys):
                raise ValueError(f"Expected {len(keys)} numbers, got {len(values)}")

            data_dict = dict(zip(keys, values))

            # Validate using Pydantic
            validated_data = InputData(**data_dict)
            break
        except Exception as e:
            print(f"Invalid input: {e}. Please try again.")

    # Create dictionary
    df = pd.DataFrame([validated_data.dict()])
    print("\nValidated DataFrame:")
    print(df)

    preds = model.predict(df)
    result = preds.tolist()[0]

    from rich.console import Console

    console = Console()
    if result == 1:
        console.print("!!! Likely to churn !!!", style="bold red")  # High risk - needs intervention
    else:
        console.print("!!! Not likely to churn !!!", style="bold green")  # Low risk - maintain normal service

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test pipeline")
    p.add_argument("--model_root", type=str, default=None)
    p.add_argument("--experiment-name", type=str, default=None)

    args = p.parse_args()
    main(args)