import pandas as pd
import time
from datetime import datetime
from pipegenie.classification import PipegenieClassifier
from pipegenie.model_selection import train_test_split
from pipegenie.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from pprint import pformat  # For clean model info formatting



# Define the grammar file
grammar_file = "../tutorials/sample-grammar-classification.xml"

OUTPUT_FILE = "chart_generic_no_output_33.txt"

def log(msg):
    timestamp = datetime.now().strftime("[%H:%M:%S] ")
    print(timestamp + msg)
    with open(OUTPUT_FILE, "a") as f:
        f.write(timestamp + msg + "\n")

def run_experiment(seed):
    log(f"\n🚀 Running AutoML with seed {seed}")
    raw_start_time = datetime.now()
    start_time = time.time()

    # Load dataset
    df = pd.read_csv('../../autosklearn/metalearn/datasets/chart.csv')
    df = df.drop(columns=['Version', 'TestID', 'TestName'])

    # Encode 'Status'
    df['Status'] = LabelEncoder().fit_transform(df['Status'])

    # Split data
    X = df.drop(columns=['Result'])
    y = df['Result']

    


    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)



    # Create an instance of the PipegenieClassifier
    model = PipegenieClassifier(
        grammar=grammar_file,
        generations=200,
        pop_size=200,
        elite_size=20,
        n_jobs=1,
        seed=seed,
        timeout=3600,
        outdir=f"sample-results-chart-{seed}",
    )

    # Fit the classifier
    model.fit(X_train, y_train)

    # Predict using the classifier
    y_pred = model.predict(X_test)

    # Evaluate the classifier using the default scoring method
    print(f"Model score: {model.score(X_test, y_test)}")

    # Evaluate the classifier using another metric
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}")

    raw_end_time = datetime.now()
    elapsed_time = time.time() - start_time
    elapsed_str = f"{elapsed_time:.2f} seconds"

    # Logging
    
    
    log(f"✅ Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    log(f"🕒 Start Time: {raw_start_time}")
    log(f"🕒 End Time: {raw_end_time}")
    log(f"⏱️ Elapsed Time: {elapsed_str}")

if __name__ == "__main__":
    # Clear old log
    open(OUTPUT_FILE, "w").close()
    for seed in range(33, 43):  
        run_experiment(seed)
