""" # All your imports at the top
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pipegenie.classification import PipegenieClassifier # Or however you import it
import multiprocessing # Add this import

# Any global constants or functions that DON'T start processes

def main(): # Optional: put your main logic in a function
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the grammar file path
    # Make sure this path is correct relative to where you run the script
    # If pipegenie_test.py is in evoflow-dev-main, and grammar is in a subdir like 'grammars'
    # grammar_file = "grammars/your_grammar.xml" 
    # OR if it's in the same directory:
    grammar_file = "tutorials/sample-grammar-classification.xml"

    # Initialize PipegenieClassifier
    # (Assuming the rest of your parameters from the traceback are correct)
    model = PipegenieClassifier(
        grammar=grammar_file,
        generations=5,
        pop_size=50,
        elite_size=5,
        n_jobs=5,
        seed=42,
        timeout=90,
        eval_timeout=36,
        outdir="sample-results",
    )

    # Fit the model
    print("Starting model fitting...") # Add a print statement for feedback
    model.fit(X_train, y_train)
    print("Model fitting finished.")

    # Predict (optional, for testing)
    # y_pred = model.predict(X_test)
    # print(f"Predictions: {y_pred}")

    # Evaluate (optional, for testing)
    # score = model.score(X_test, y_test)
    # print(f"Test score: {score}")

# THIS IS THE CRUCIAL PART
if __name__ == '__main__':
    multiprocessing.freeze_support() # Call this first in the main block on Windows
    main() # Call your main logic
 """
















from pipegenie.classification import PipegenieClassifier
from pipegenie.model_selection import train_test_split
from pipegenie.metrics import balanced_accuracy_score
from sklearn.datasets import load_iris


def main():
    # Load the dataset
    X, y = load_iris(return_X_y=True)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define the grammar file
    grammar_file = "../../tutorials/sample-grammar-classification.xml"

    # Create an instance of the PipegenieClassifier
    model = PipegenieClassifier(
        grammar=grammar_file,
        generations=5,
        pop_size=50,
        elite_size=5,
        n_jobs=5,
        seed=42,
        timeout=90,
        eval_timeout=36,
        outdir="sample-results",
        metalearning_seeding_strategy=1,
        ensemble_file_format = 'json',
        verbose=False,
    )

    # Fit the classifier
    model.fit(X_train, y_train)

    # Predict using the classifier
    y_pred = model.predict(X_test)

    # Evaluate the classifier using the default scoring method
    print(f"Model score: {model.score(X_test, y_test)}")

    # Evaluate the classifier using another metric
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}")

# THIS IS THE CRUCIAL PART
if __name__ == '__main__':
    
    main() # Call your main logic
 