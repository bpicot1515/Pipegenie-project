# build_metabase.py

""" This is the script to build the metabase for meta-learning, to have a stronger 
meta-learning module we should increase the number of tasks in CLASSIFICATION_TASKS
and REGRESSION_TASKS """

import openml
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from pathlib import Path
import traceback
import os

from pipegenie.classification import PipegenieClassifier
from pipegenie.regression import PipegenieRegressor

# Your list of tasks
CLASSIFICATION_TASKS = [232, 241, 245, 273, 275, 288, 336, 340, 2119, 2120, 2121, 2122, 2123, 2125, 2356]
REGRESSION_TASKS = [359997, 359998, 360000, 360001, 360003, 167146, 360004, 360005, 360006, 360007, 211696, 360009, 360010, 360011, 360012]

# --- Use absolute paths or paths relative to the script's location for robustness ---
SCRIPT_DIR = Path(__file__).parent.resolve()
CLASSIFICATION_GRAMMAR = SCRIPT_DIR / "../../tutorials/sample-grammar-classification.xml"
REGRESSION_GRAMMAR = SCRIPT_DIR / "../../tutorials/sample-grammar-regression.xml"
METABASE_PATH = SCRIPT_DIR / "openml_metabase"
# -------------------------------------------------------------------------------------

def get_preprocessor(categorical_features_indices, numerical_features_indices):
    """
    Creates a scikit-learn preprocessing pipeline for the data.
    This version includes imputation AND scaling for numerical features.
    """
    numeric_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features_indices),
            ('cat', categorical_transformer, categorical_features_indices)
        ],
        remainder='passthrough'
    )
    return preprocessor


def run_openml_task(task_id: int):
    """
    Loads, preprocesses, and runs a pipegenie model on a single OpenML task,
    saving the results to the metabase.
    """
    print(f"\n▶️ Processing OpenML Task ID: {task_id} ---")
    try:
        # 1. Load data from OpenML
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True)
        
        print(f"   Dataset: {dataset.name}, Shape: {dataset.qualities['NumberOfInstances']}x{dataset.qualities['NumberOfFeatures']}")

        X_df, y_series, categorical_indicator, _ = dataset.get_data(
            dataset_format="dataframe", target=task.target_name
        )
        
        train_indices, _ = task.get_train_test_split_indices()
        X_train_df, y_train_series = X_df.iloc[train_indices], y_series.iloc[train_indices]

        # 2. Determine task type and prepare target variable 'y'
        y_train_np = None
        if isinstance(task, openml.tasks.OpenMLClassificationTask):
            ModelClass, grammar_file = PipegenieClassifier, CLASSIFICATION_GRAMMAR
            if y_train_series.dtype == 'object' or pd.api.types.is_categorical_dtype(y_train_series.dtype):
                print("   Target variable is categorical. Applying LabelEncoder.")
                le = LabelEncoder()
                y_train_np = le.fit_transform(y_train_series)
            else:
                y_train_np = y_train_series.to_numpy()
        
        elif isinstance(task, openml.tasks.OpenMLRegressionTask):
            ModelClass, grammar_file = PipegenieRegressor, REGRESSION_GRAMMAR
            y_train_np = y_train_series.to_numpy()
        
        else:
            print(f"   Skipping task {task_id}: Unknown task type.")
            return

        # 3. Preprocess features 'X'
        categorical_features_indices = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
        numerical_features_indices = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]
        
        preprocessor = get_preprocessor(categorical_features_indices, numerical_features_indices)
        
        X_train_processed = preprocessor.fit_transform(X_train_df)
        
        # --- FIX: EXPLICITLY CONVERT TO FLOAT64 AND HANDLE NaNs ---
        # This is the critical fix that solves the dtype errors.
        X_train_processed = np.nan_to_num(X_train_processed.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # ---------------------------------------------------------
        
        print("   Preprocessing complete.")

        # 4. Configure and run the pipegenie model
        model = ModelClass(
            grammar=str(grammar_file.resolve()),
            generations=200,
            pop_size=200,
            elite_size=20,
            n_jobs=-1,
            seed=42,
            timeout=3600,
            outdir=f"temp_results/task_{task_id}",
            metabase_path=str(METABASE_PATH.resolve()),
            
            verbose=True, # Set to True to see generational logs in nohup.out
        )
        
        model.fit(X_train_processed, y_train_np)
        
        print(f"✅ Successfully finished and saved task {task_id}.")

    except Exception as e:
        print(f"❌ Failed to process task {task_id}: {e}")
        traceback.print_exc()


def main():
    # Ensure all directories exist
    METABASE_PATH.mkdir(exist_ok=True)
    Path("temp_results").mkdir(exist_ok=True)
    
    print("Starting Meta-Base Build Process... This will overwrite existing results.")
    
    #all_tasks = CLASSIFICATION_TASKS + REGRESSION_TASKS
    all_tasks = REGRESSION_TASKS
    for task_id in all_tasks:
        run_openml_task(task_id)
        
    print("\nMeta-Base Build Process Finished.")
    print(f"Knowledge base is located in: '{METABASE_PATH}'")


if __name__ == '__main__':
    # It's good practice to ensure the environment is set up for multiprocessing
    # especially if you might run this on different OSes.
    from multiprocessing import set_start_method
    try:
        # 'fork' is generally faster but can be less safe if not handled carefully.
        # 'spawn' is safer but slower as it starts a fresh process.
        # For long-running, independent tasks like this, 'fork' is usually fine.
        set_start_method("fork")
    except RuntimeError:
        # This will happen if the start method has already been set.
        pass
    
    main()