from pipegenie.regression import PipegenieRegressor
from pipegenie.model_selection import train_test_split
from pipegenie.metrics import root_mean_squared_error
from sklearn.datasets import load_diabetes

# Load the dataset
X, y = load_diabetes(return_X_y=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)

# Define the grammar file
grammar_file = "../../tutorials/sample-grammar-regression.xml"

# Create an instance of the PipegenieRegressor
model = PipegenieRegressor(
    grammar=grammar_file,
    generations=5,
    pop_size=50,
    elite_size=5,
    maximization=False,
    n_jobs=5,
    seed=9,
    save_to_metabase=False,
    enable_metalearning=True,
    metabase_path="metabase_test",
    timeout=90,
    outdir="sample-results-regression",
    metalearning_seeding_strategy=0.10,
)

# Fit the regressor
model.fit(X_train, y_train)

# Predict using the regressor
y_pred = model.predict(X_test)

# Evaluate the regressor using the default scoring method
print(f"Model score: {model.score(X_test, y_test)}")

# Evaluate the regressor using another metric
print(f"Root mean squared error: {root_mean_squared_error(y_test, y_pred)}")