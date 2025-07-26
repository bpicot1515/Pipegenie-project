# pipegenie/metalearning/metabase.py

import json
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from pipegenie.evolutionary._individual import Individual, Fitness
from pipegenie.syntax_tree._encoding import NonTerminalNode, TerminalNode

if TYPE_CHECKING:
    from collections.abc import Sequence

class MetaBase:
    """Manages the meta-learning knowledge base."""

    def __init__(self, metabase_path: str):
        self.path = Path(metabase_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.runs: list[dict[str, Any]] = self._load_runs()
        self.feature_names: Optional[list[str]] = None
        self.feature_matrix: Optional[np.ndarray] = None
        
        if self.runs:
            self._build_feature_matrix()

    def _load_runs(self) -> list[dict[str, Any]]:
        """Loads all experiment runs from the metabase directory."""
        runs = []
        for file_path in self.path.glob("*.json"):
            try:
                with file_path.open("r") as f:
                    runs.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                # Ignore corrupted files
                continue
        return runs

    def _build_feature_matrix(self) -> None:
        """Builds a normalized numpy array of meta-features for distance calculation."""
        if not self.runs:
            return

        # Get a consistent order of feature names from the first run
        self.feature_names = sorted(self.runs[0]['metafeatures'].keys())
        
        matrix = []
        for run in self.runs:
            # Ensure all runs have the same features in the same order
            row = [run['metafeatures'].get(name, 0.0) for name in self.feature_names]
            matrix.append(row)
        
        self.feature_matrix = np.array(matrix, dtype=float)
        
        # Normalize the feature matrix for distance calculation
        self.min_vals = np.nanmin(self.feature_matrix, axis=0)
        self.max_vals = np.nanmax(self.feature_matrix, axis=0)
        self.range = self.max_vals - self.min_vals
        self.range[self.range == 0] = 1 # Avoid division by zero

    def find_similar_pipelines(self, query_metafeatures: dict[str, float], n: int = 5) -> list[Individual]:
        """
        Finds the most similar datasets and returns their best pipelines.

        Parameters
        ----------
        query_metafeatures : dict
            The meta-features of the new dataset.
        n : int
            The number of pipelines to return.

        Returns
        -------
        pipelines : list of Individual
            A list of Individual objects to seed the initial population.
        """
        if self.feature_matrix is None or self.feature_names is None:
            return []

        # Ensure query vector has the same order and normalize it
        query_vector = np.array([query_metafeatures.get(name, 0.0) for name in self.feature_names])
        normalized_query = (query_vector - self.min_vals) / self.range

        # Normalize the stored matrix on-the-fly
        normalized_matrix = (self.feature_matrix - self.min_vals) / self.range

        # Calculate Euclidean distances
        distances = np.linalg.norm(normalized_matrix - normalized_query, axis=1)
        
        # Get the indices of the most similar runs
        similar_indices = np.argsort(distances)
        
        # Collect pipelines from the most similar runs
        pipelines = []
        seen_pipelines = set()
        for idx in similar_indices:
            run = self.runs[idx]
            for ind_content in run['elite_pipelines']:
                pipeline_str = self._rehydrate_to_string(ind_content)
                if pipeline_str not in seen_pipelines:
                    try:
                        individual = self._rehydrate_individual(ind_content)
                        pipelines.append(individual)
                        seen_pipelines.add(pipeline_str)
                    except Exception:
                        continue # Skip if rehydration fails

                if len(pipelines) >= n:
                    return pipelines
        return pipelines

    def save_run(self, metafeatures: dict[str, float], elite: "Sequence[Individual]") -> None:
        """Saves the results of a new run to the meta-base."""
        # Use a hash of meta-features to create a unique filename
        # This prevents duplicate entries for the same dataset
        m_hash = hashlib.sha256(json.dumps(sorted(metafeatures.items())).encode()).hexdigest()
        filepath = self.path / f"{m_hash}.json"

        run_data = {
            "metafeatures": metafeatures,
            "elite_pipelines": [self._serialize_individual(ind) for ind in elite]
        }

        with filepath.open("w") as f:
            json.dump(run_data, f, indent=4)

    def _serialize_individual(self, ind: Individual) -> list[dict[str, str]]:
        """Converts an Individual's content to a JSON-serializable list."""
        serialized = []
        for node in ind.content:
            node_dict = {"symbol": node.symbol}
            if isinstance(node, NonTerminalNode):
                node_dict["type"] = "nonterminal"
                node_dict["production"] = node.production
            elif isinstance(node, TerminalNode):
                node_dict["type"] = "terminal"
            serialized.append(node_dict)
        return serialized

    def _rehydrate_individual(self, content: list[dict[str, str]]) -> Individual:
        """Reconstructs an Individual from a serialized list."""
        rehydrated_content = []
        for node_dict in content:
            if node_dict["type"] == "nonterminal":
                node = NonTerminalNode(node_dict["symbol"], node_dict["production"])
            else: # Terminal
                # The 'code' part of the TerminalNode isn't needed for re-creating the structure
                # The SyntaxTreePipeline will parse the string representation later.
                node = TerminalNode(node_dict["symbol"], code=None)
            rehydrated_content.append(node)
        
        # Fitness is invalid until evaluated
        return Individual(rehydrated_content, fitness=Fitness(weights=(-1.0,)))

    def _rehydrate_to_string(self, content: list[dict[str, str]]) -> str:
        """Helper to get a string representation for uniqueness checks."""
        temp_ind = self._rehydrate_individual(content)
        return str(temp_ind)