#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Utility functions.
"""
import numpy as np # Add numpy import
import json # Add json import


def is_number(string: str) -> bool:
    """
    Check if a string is a number.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_number : bool
        True if the string is a number, False otherwise.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_bool(string: str) -> bool:
    """
    Check if a string is a boolean.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_bool : bool
        True if the string is a boolean, False otherwise.
    """
    return string.lower() in ('true', 'false')

def is_tuple(string: str) -> bool:
    """
    Check if a string is a tuple.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_tuple : bool
        True if the string is a tuple, False otherwise.
    """
    return string.startswith('(') and string.endswith(')')

# --- ADD THIS NEW FUNCTION ---
def serialize_estimator(estimator):
    """
    Recursively serializes a scikit-learn estimator or pipeline to a JSON-compatible dict.
    """
    if estimator is None:
        return None
        
    # Handle Pipelines
    if hasattr(estimator, 'steps'):
        return {
            "class": f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
            "steps": [
                {"name": name, "estimator": serialize_estimator(step)}
                for name, step in estimator.steps
            ]
        }
    
    # Handle single estimators
    params = {}
    if hasattr(estimator, 'get_params'):
        # Use deep=True to get all parameters, but handle nested estimators carefully.
        for key, value in estimator.get_params(deep=True).items():
            # Avoid serializing the entire nested pipeline structure multiple times
            if '__' in key:
                continue

            # --- THE FIX: ROBUST VALUE SERIALIZATION ---
            if hasattr(value, 'get_params'): # It's a nested estimator
                params[key] = serialize_estimator(value)
            elif callable(value): # It's a function
                params[key] = f"<function {value.__module__}.{value.__name__}>"
            elif isinstance(value, np.ndarray): # It's a numpy array
                params[key] = value.tolist() # Convert to a standard list
            elif isinstance(value, (np.integer, np.floating)): # It's a numpy number
                params[key] = value.item() # Convert to a standard Python number
            else:
                # For all other simple types (str, int, float, bool, None)
                try:
                    # Quick check if it's serializable
                    json.dumps(value)
                    params[key] = value
                except TypeError:
                    # If it's still not serializable, represent it as a string
                    params[key] = str(value)
            # --- END OF FIX ---

    return {
        "class": f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
        "params": params,
    }
