# pipegenie/metalearning/metabase.py

import json
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from pipegenie.evolutionary._individual import Individual, Fitness
from pipegenie.syntax_tree._encoding import NonTerminalNode, TerminalNode, SyntaxTree
from pipegenie.syntax_tree._primitive_set import Primitive

if TYPE_CHECKING:
    from collections.abc import Sequence

class RehydratedCode:
    """A dummy object to hold a value during rehydration, satisfying the .name or .arity contract."""
    def __init__(self, value: Any, arity: Optional[int] = None):
        self.name = str(value)
        self.arity = arity if arity is not None else 0

class MetaBase:
    """Manages the meta-learning knowledge base."""

    def __init__(self, metabase_path: str):
        self.path = Path(metabase_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.runs: list[dict[str, Any]] = self._load_runs()
        self.feature_names: Optional[list[str]] = None

    def _load_runs(self) -> list[dict[str, Any]]:
        runs = []
        for file_path in self.path.glob("*.json"):
            try:
                with file_path.open("r") as f:
                    runs.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        return runs

    def _serialize_individual(self, ind: Individual) -> list[dict[str, Any]]:
        """
        Converts an Individual's content to the specific JSON format you have.
        """
        serialized = []
        for node in ind.content:
            node_dict = {"symbol": node.symbol}
            if isinstance(node, NonTerminalNode):
                node_dict["type"] = "nonterminal"
                node_dict["production"] = node.production
            elif isinstance(node, TerminalNode):
                node_dict["type"] = "terminal"
                node_dict["arity"] = node.arity
                # For both algorithms and hyperparameters, the resolved value/name is stored in 'name'
                if node.code is not None:
                    node_dict["name"] = node.code.name if hasattr(node.code, 'name') else node.symbol
                else:
                    node_dict["name"] = node.symbol
            serialized.append(node_dict)
        return serialized

    def _rehydrate_individual(self, content: list[dict[str, Any]]) -> Individual:
        """
        Reconstructs an Individual from your specific JSON format.
        """
        rehydrated_content = []
        for node_dict in content:
            symbol = node_dict["symbol"]
            node_type = node_dict["type"]

            if node_type == "nonterminal":
                node = NonTerminalNode(symbol, node_dict["production"])
            else:  # Terminal
                # THE FIX: Create a dummy code object that has the needed attributes.
                # The 'name' key from the JSON holds the value for hyperparameters.
                value = node_dict.get("name") 
                # The 'arity' key is present for algorithms.
                arity = node_dict.get("arity")
                
                # Create a simple object that satisfies both .name and .arity checks
                code_obj = RehydratedCode(value, arity)
                
                node = TerminalNode(symbol, code=code_obj)
            rehydrated_content.append(node)
        
        return Individual(rehydrated_content, fitness=Fitness(weights=(-1.0,)))


    def _rehydrate_to_string(self, content: list[dict[str, Any]]) -> str:
        # This rehydration now works because _rehydrate_individual is correct.
        temp_ind = self._rehydrate_individual(content)
        return str(temp_ind)


    def _infer_task_type_from_run(self, run_data: dict[str, Any]) -> Optional[str]:
        # This logic is correct and remains the same.
        if "task_type" in run_data:
            return run_data["task_type"]
        if "elite_pipelines" not in run_data or not run_data["elite_pipelines"]:
            return None
        
        first_pipeline_content = run_data["elite_pipelines"][0]
        for node_data in first_pipeline_content:
            if node_data.get("type") == "nonterminal":
                production_rule = node_data.get("production", "")
                if "classifier" in production_rule:
                    return "classification"
                if "regressor" in production_rule:
                    return "regression"
        return None

    def _build_feature_matrix_for_task(self, task_type: str) -> tuple[Optional[np.ndarray], list[dict]]:
        # This logic is correct and remains the same.
        task_specific_runs = []
        for run in self.runs:
            inferred_type = self._infer_task_type_from_run(run)
            if inferred_type == task_type:
                task_specific_runs.append(run)

        if not task_specific_runs:
            return None, []

        self.feature_names = sorted(task_specific_runs[0]['metafeatures'].keys())
        matrix = [[run['metafeatures'].get(name, 0.0) for name in self.feature_names] for run in task_specific_runs]
        return np.array(matrix, dtype=float), task_specific_runs

    def find_similar_pipelines(
        self,
        query_metafeatures: dict[str, float],
        task_type: str,
        n: int = 5
    ) -> list[Individual]:
        # This logic is correct and remains the same.
        feature_matrix, relevant_runs = self._build_feature_matrix_for_task(task_type)
        if feature_matrix is None or not relevant_runs or self.feature_names is None:
            return []

        min_vals = np.nanmin(feature_matrix, axis=0)
        max_vals = np.nanmax(feature_matrix, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        query_vector = np.array([query_metafeatures.get(name, 0.0) for name in self.feature_names])
        normalized_query = (query_vector - min_vals) / range_vals
        normalized_matrix = (feature_matrix - min_vals) / range_vals
        distances = np.linalg.norm(normalized_matrix - normalized_query, axis=1)
        
        similar_indices = np.argsort(distances)
        
        pipelines = []
        seen_pipelines = set()
        for idx in similar_indices:
            run = relevant_runs[idx]
            for ind_data in run['elite_pipelines']:
                pipeline_str = self._rehydrate_to_string(ind_data)
                if pipeline_str not in seen_pipelines:
                    try:
                        individual = self._rehydrate_individual(ind_data)
                        pipelines.append(individual)
                        seen_pipelines.add(pipeline_str)
                    except Exception:
                        continue
                if len(pipelines) >= n:
                    return pipelines
        return pipelines

    def save_run(
        self,
        metafeatures: dict[str, float],
        elite: "Sequence[Individual]",
        task_type: str
    ) -> None:
        """Saves the results of a new run to the meta-base with a task type tag."""
        m_hash = hashlib.sha256(json.dumps(sorted(metafeatures.items())).encode()).hexdigest()
        filepath = self.path / f"{m_hash}.json"

        run_data = {
            "task_type": task_type,
            "metafeatures": metafeatures,
            "elite_pipelines": [self._serialize_individual(ind) for ind in elite]
        }

        with filepath.open("w") as f:
            json.dump(run_data, f, indent=4)