# pipegenie/base.py

#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Base class for all pipegenie estimators.
"""
import json # Add json import
import datetime # Added
import logging
import logging.handlers
import pickle
import random
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from math import ceil
from multiprocessing import Manager, cpu_count
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Optional # <-- I missed Union here

import numpy as np
from loky import ProcessPoolExecutor

from pipegenie.elite._elite import DiverseElite
from pipegenie.evolutionary._individual import Fitness, Individual
from pipegenie.grammar import parse_pipe_grammar
from pipegenie.logging._logging import Logbook
from pipegenie.logging._stats import MultiStatistics, Statistics
from pipegenie.syntax_tree._encoding import SyntaxTreeSchema

from pipegenie.utils import serialize_estimator # Add this new import

# --- NEW IMPORTS FOR METALEARNING ---
from pipegenie.metalearning.metafeatures import (
    BaseMetafeatureCalculator,
    ClassificationMetafeatureCalculator,
    RegressionMetafeatureCalculator
)
from pipegenie.metalearning.metabase import MetaBase
from typing import Union # Add Union
# ------------------------------------

if TYPE_CHECKING:
    from collections.abc import Callable
    from queue import Queue
    from typing import Optional

    from numpy.typing import ArrayLike

    from pipegenie.evolutionary.crossover import CrossoverBase
    from pipegenie.evolutionary.mutation import MutationBase
    from pipegenie.evolutionary.replacement_og import ReplacementBase
    from pipegenie.evolutionary.selection import SelectionBase
    from pipegenie.model_selection import BaseCrossValidator
    from pipegenie.pipeline import Pipeline
    from pipegenie.voting import BaseVoting

# Ignore all warnings
warnings.filterwarnings("ignore")

# Show warnings from the "pipegenie" package
warnings.filterwarnings("default", module="^pipegenie\.")


class BasePipegenie(ABC):
    r"""
    Base class for all pipegenie estimators.

    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword arguments
    (no \*args or \*\*kwargs).

    This is a minimal interface that all estimators should implement.

    Parameters
    ----------
    grammar : str
        Path of the file containing the grammar used to generate the pipelines.

    grammar_type : str
        Format used to define the grammar.
        Supported formats: "evoflow-xml".
        You can add new formats by creating a new parser and registering it in the factory.

    pop_size : int
        Size of the population used in the evolutionary process.
        Must be greater than 0.

    generations : int
        Number of generations used in the evolutionary process.
        Must be greater or equal than 0.
        If 0, only the initial population will be evaluated.

    fitness : function
        Fitness function used to evaluate the pipelines.

    nderiv : int
        Number of derivations used to generate the pipelines.
        Must be greater than 0.

    selection : SelectionBase instance
        Selection method used to select the individuals that
        will be used to generate the offspring.

    crossover : CrossoverBase instance
        Crossover function used to generate new individuals.

    mutation : MutationBase instance
        Mutation function used to generate new individuals.

    mutation_elite : MutationBase instance
        Mutation function used to generate new individuals from the elite.

    replacement : ReplacementBase instance
        Replacement function used to create the new population.

    use_double_mutation : bool
        Indicates if the mutation_elite function should be used to generate new individuals
        from the elite instead of the mutation function.
        No crossover will be applied.
        If False, the crossover function will be applied and then the mutation function.

    elite_size : int
        Maximun number of pipelines to be stored in the elite.
        must be in the range [1, pop_size].

    timeout : int
        Maximun time allowed for the evolutionary process.
        Must be greater than 0.

    eval_timeout : int or None
        Maximun time allowed for the evaluation of a pipeline.
        It has to be greater than 0 or None.
        If None, the evaluation of a pipeline will not have a time limit.

    cross_validator : BaseCrossValidator instance
        Cross-validator used to evaluate the pipelines.

    maximization : bool
        Indicates if the fitness function should be maximized or minimized.

    early_stopping_threshold : float
        Threshold used to determine if there is improvement in the elite average fitness.
        Must be greater or equal than 0.0.

    early_stopping_generations : int or None
        Number of generations without improvement in the elite average fitness
        before stopping the evolutionary process.
        It has to be greater than 0 or None.
        If None, the evolutionary process will not stop based on generations without improvement.

    early_stopping_time : int or None
        Maximun time without improvement in the elite average fitness
        before stopping the evolutionary process.
        It has to be greater than 0 or None.
        If None, the evolutionary process will not stop based on time without improvement.

    seed : int or None
        Seed used to initialize the random number generator.
        If None, a random seed will be used.

    outdir : str
        Path of the directory where the results will be stored.

    n_jobs : int
        Number of processes used to evaluate the population.
        Must be greater than 0.
        If -1, the number of processes will be equal to the number of cores of the machine.

    verbose : bool, default=True
        Controls the verbosity of the output. If False, only high-level status
        messages are printed to the console. The detailed generational log is
        always saved to `evolution.txt`.


        metabase_path : str, default=None
    #        The directory path to store and retrieve meta-learning knowledge. If None,
    #        meta-learning is disabled.
    #
    #    metalearning_seeding_strategy : int or float, default=0.25
    #        Controls how the initial population is seeded from the meta-base.
    #        - If float (e.g., 0.25): Seeds this percentage of the population (25%).
    #        - If int (e.g., 10): Seeds a fixed number of individuals.

    """

    def __init__(
        self,
        grammar: str,
        *,
        grammar_type: str,
        pop_size: int,
        generations: int,
        fitness: 'Callable[..., object]',
        nderiv: int,
        selection: 'SelectionBase',
        crossover: 'CrossoverBase',
        mutation: 'MutationBase',
        mutation_elite: 'MutationBase',
        replacement: 'ReplacementBase',
        use_double_mutation: bool,
        elite_size: int,
        timeout: int,
        eval_timeout: 'Optional[int]',
        cross_validator: 'BaseCrossValidator',
        maximization: bool,
        early_stopping_threshold: float,
        early_stopping_generations: 'Optional[int]',
        early_stopping_time: 'Optional[int]',
        seed: 'Optional[int]',
        outdir: str,
        n_jobs: int,
        verbose: bool = True, # Added
        # --- METALEARNING PARAMETERS ---
        enable_metalearning: bool = True,
        save_to_metabase: bool = False,
        metabase_path: Optional[str] = None,
        metalearning_seeding_strategy: Union[int, float] = 0.25, # Added
        # ---------------------------------------
        ensemble_file_format: str = 'txt', # <-- ADD THIS
        **kwargs: object,
    ):
        self.grammar = grammar
        self.grammar_type = grammar_type
        self.pop_size = pop_size
        self.generations = generations
        self.fitness = fitness
        self.nderiv = nderiv
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.mutation_elite = mutation_elite
        self.replacement = replacement
        self.use_double_mutation = use_double_mutation
        self.elite_size = elite_size
        self.timeout = timeout
        self.eval_timeout = eval_timeout
        self.cross_validator = cross_validator
        self.maximization = maximization
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_generations = early_stopping_generations
        self.early_stopping_time = early_stopping_time
        self.outdir = outdir
        self.n_jobs = n_jobs
        self.verbose = verbose # Added

        # --- NEW INITIALIZATIONS ---
        self.enable_metalearning = enable_metalearning
        self.save_to_metabase = save_to_metabase
        self.metabase_path = metabase_path
        self.metalearning_seeding_strategy = metalearning_seeding_strategy # Added
        self.metabase: Optional[MetaBase] = None
        self.metafeature_calculator: Optional[BaseMetafeatureCalculator] = None # Type hint updated

        self.ensemble_file_format = ensemble_file_format # <-- ADD THIS

        if self.ensemble_file_format not in ['txt', 'json']:
            raise ValueError("`ensemble_file_format` must be either 'txt' or 'json'.")
        
        # Only activate if the feature is enabled AND a path is provided
        #if self.enable_metalearning and self.metabase_path is not None:
        if self.metabase_path is not None:
            self.metabase = MetaBase(self.metabase_path)
            # Decide which calculator to use based on the class type
            if "PipegenieClassifier" in self.__class__.__name__:
                self.metafeature_calculator = ClassificationMetafeatureCalculator()
            elif "PipegenieRegressor" in self.__class__.__name__:
                self.metafeature_calculator = RegressionMetafeatureCalculator()
            else:
                # Fallback or error if another class inherits BasePipegenie in the future
                self.enable_metalearning = False
                self.evolution_logger.warning("    - Meta-learning disabled: unknown task type.")
        else:
            self.enable_metalearning = False

        self.seed = seed if seed is not None else random.randint(0, 2**32)
        random.seed(self.seed)

        # Ensure that the cross-validator is initialized with the same seed
        self.cross_validator.set_random_state(random_state=self.seed)

        arguments = vars(self).copy()
        arguments.update(kwargs)

        self.outdir_path = Path(self.outdir)
        self.outdir_path.mkdir(parents=True, exist_ok=True)

        self.cpu_count = cpu_count() if self.n_jobs == -1 else self.n_jobs

        self._create_loggers()
        
        # Log the configuration to file
        with self.outdir_path.joinpath("config.txt").open("w", encoding="utf-8") as log:
            for key, value in arguments.items():
                if hasattr(value, '__name__'):
                    log.write(f"{key}: {value.__name__}\n")
                else:
                    log.write(f"{key}: {value}\n")
        
        # Log key configuration to evolution logger
        self.evolution_logger.info("[1/5] ⚙️  Configuration")
        self.evolution_logger.info(f"    - Generations: {self.generations}, Population Size: {self.pop_size}")
        self.evolution_logger.info(f"    - Elite Size: {self.elite_size}, Timeout: {self.timeout}s")
        self.evolution_logger.info(f"    - Fitness Function: {self.fitness.__name__}, Maximization: {self.maximization}")
        self.evolution_logger.info(f"    - Output Directory: {self.outdir_path.resolve()}")
        self.evolution_logger.info(f"    - Random Seed: {self.seed}")

        self._init_statistics()

        root, terms, non_terms, self.pset, _ = parse_pipe_grammar(
            grammar,
            grammar_type,
            self.seed,
        )
        self.schema = SyntaxTreeSchema(root, nderiv, terms, non_terms)


    def _create_initial_population(self, data: 'dict[str, ArrayLike]') -> list[Individual]:
        """Creates the initial population, using meta-learning if enabled."""
        meta_learned_pop = []
        num_to_seed = 0

        # --- UPDATED SEEDING LOGIC ---
        if self.enable_metalearning and self.metabase is not None:
        
            self.evolution_logger.info("    - Using meta-learning to warm-start population...")

            # Determine the number of individuals to seed
            if isinstance(self.metalearning_seeding_strategy, float):
                if not 0.0 < self.metalearning_seeding_strategy <= 1.0:
                    raise ValueError("Seeding strategy (float) must be between 0.0 and 1.0.")
                num_to_seed = int(self.pop_size * self.metalearning_seeding_strategy)
            elif isinstance(self.metalearning_seeding_strategy, int):
                if not 0 < self.metalearning_seeding_strategy <= self.pop_size:
                    raise ValueError(f"Seeding strategy (int) must be between 1 and pop_size ({self.pop_size}).")
                num_to_seed = self.metalearning_seeding_strategy
            else:
                raise TypeError("`metalearning_seeding_strategy` must be an int or a float.")
            
            if num_to_seed > 0:
                
                
                # 1. Calculate meta-features for the current dataset
                query_metafeatures = self.metafeature_calculator.calculate(data["X"], data["y"])

                # 2. Query meta-base for best individuals
                #meta_learned_pop = self.metabase.find_similar_pipelines(query_metafeatures, n=num_to_seed)
                task_type = self._get_task_type()
                meta_learned_pop = self.metabase.find_similar_pipelines(
                    query_metafeatures, task_type, n=num_to_seed
                )
                
                if meta_learned_pop:
                    self.evolution_logger.info(f"    - Seeding population with {len(meta_learned_pop)} pipeline(s) from past runs.")
        # -----------------------------------

        # Generate the rest of the population randomly
        num_random = self.pop_size - len(meta_learned_pop)
        random_pop = [
            Individual(
                self.schema.create_syntax_tree(),
                Fitness((1.0,) if self.maximization else (-1.0,)),
            ) for _ in range(num_random)
        ]

        # Combine the seeded and random populations
        initial_pop = meta_learned_pop + random_pop
        
        # Shuffle the initial population to mix meta-learned and random individuals
        random.shuffle(initial_pop)
        
        return initial_pop



    def _create_loggers(self) -> None:
        # Create a logger for individual evaluation
        self.individuals_logger = logging.getLogger("pipegenie_individuals")
        self.individuals_logger.setLevel(logging.INFO)
        # Prevent double logging if fit is called multiple times
        if not self.individuals_logger.hasHandlers():
            ind_handler = logging.FileHandler(self.outdir_path.joinpath("individuals.tsv"), mode="w")
            self.individuals_logger.addHandler(ind_handler)
            self.individuals_logger.info("pipeline\tfitness\tfit_time")

        # --- NEW LOGGING SETUP FOR EVOLUTION ---
        self.evolution_logger = logging.getLogger("pipegenie_evolution")
        self.evolution_logger.setLevel(logging.INFO)

        # Prevent adding handlers multiple times if fit is called again
        if not self.evolution_logger.hasHandlers():
            # 1. Console handler (depends on verbosity)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING if not self.verbose else logging.INFO)
            # Use a formatter that only shows the message for the console
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            self.evolution_logger.addHandler(console_handler)

            # 2. Clean file handler for evolution.txt (table and final times only)
            clean_log_path = self.outdir_path.joinpath("evolution.txt")
            clean_file_handler = logging.FileHandler(clean_log_path, mode="w")
            clean_file_handler.setLevel(logging.INFO) # Capture everything
            # This filter will only allow messages that are tables or contain time info
            clean_file_handler.addFilter(TableAndTimeFilter())
            self.evolution_logger.addHandler(clean_file_handler)

            # 3. Full, verbose file handler for debugging
            full_log_path = self.outdir_path.joinpath("evolution_full.log")
            full_file_handler = logging.FileHandler(full_log_path, mode="w")
            full_file_handler.setLevel(logging.INFO) # Log everything
            self.evolution_logger.addHandler(full_file_handler)
        # --- END OF NEW SETUP ---


    """ def _create_loggers(self) -> None:
        
        #Create the loggers used to store the results of the evolutionary process.
        
        # Create a logger for individual evaluation
        self.individuals_logger = logging.getLogger("pipegenie_individuals")
        self.individuals_logger.setLevel(logging.INFO)
        # Prevent double logging if fit is called multiple times
        if not self.individuals_logger.hasHandlers():
            ind_handler = logging.FileHandler(self.outdir_path.joinpath("individuals.tsv"), mode="w")
            self.individuals_logger.addHandler(ind_handler)
            self.individuals_logger.info("pipeline\tfitness\tfit_time")

        # Create a logger for the evolutionary process
        self.evolution_logger = logging.getLogger("pipegenie_evolution")
        self.evolution_logger.setLevel(logging.INFO)  # Capture all levels

        # Prevent double logging if fit is called multiple times
        if not self.evolution_logger.hasHandlers():
            # File handler always logs everything (INFO and above)
            file_handler = logging.FileHandler(self.outdir_path.joinpath("evolution.txt"), mode="w")
            file_handler.setLevel(logging.INFO)
            self.evolution_logger.addHandler(file_handler)

            # Console handler's verbosity depends on the `verbose` flag
            console_handler = logging.StreamHandler(sys.stdout)
            if self.verbose:
                console_handler.setLevel(logging.INFO)
            else:
                # Only show WARNING messages and above (our status messages)
                console_handler.setLevel(logging.WARNING)
            self.evolution_logger.addHandler(console_handler)
 """

    def _create_worker_logger(self, queue: 'Queue[object]') -> logging.Logger:
        """
        Create a logger used to store the logs generated by the worker processes.

        Parameters
        ----------
        queue : Queue
            Queue used to store the logs generated by the worker processes.

        Returns
        -------
        worker_logger : Logger
            Logger used to store the logs generated by the worker processes.
        """
        worker_logger = logging.getLogger("pipegenie_worker")

        if not worker_logger.hasHandlers():
            handler = logging.handlers.QueueHandler(queue)
            worker_logger.addHandler(handler)

        worker_logger.setLevel(logging.INFO)
        return worker_logger

    def _close_loggers(self) -> None:
        """
        Close the loggers used to store the results of the evolutionary process.
        """
        for handler in self.evolution_logger.handlers[:]:
            handler.close()
            self.evolution_logger.removeHandler(handler)

        for handler in self.individuals_logger.handlers[:]:
            handler.close()
            self.individuals_logger.removeHandler(handler)

    def _init_statistics(self) -> None:
        """
        Initialize the statistics used to evaluate the population and the elite.
        """
        stat_fit = Statistics(key=lambda ind: ind.fitness.values)
        # use string representation of the individual to calculate
        # the size because the pipeline object may not be valid
        stat_size = Statistics(key=lambda ind: len(str(ind).split(";")))
        stats = MultiStatistics(fitness=stat_fit, size=stat_size)
        stats.register('min', np.nanmin)
        stats.register('max', np.nanmax)
        stats.register('avg', np.nanmean)
        stats.register('std', np.nanstd)
        self.stats = stats

        stat_fit_elite = Statistics(key=lambda ind: ind.fitness.values)
        stat_size_elite = Statistics(key=lambda ind: len(str(ind).split(";")))
        stats_elite = MultiStatistics(fitness_elite=stat_fit_elite, size_elite=stat_size_elite)
        stats_elite.register('min', np.nanmin)
        stats_elite.register('max', np.nanmax)
        stats_elite.register('avg', np.nanmean)
        stats_elite.register('std', np.nanstd)
        self.stats_elite = stats_elite

    def _get_task_type(self) -> str:
        """Helper to determine the task type from the class name."""
        if "Classifier" in self.__class__.__name__:
            return "classification"
        elif "Regressor" in self.__class__.__name__:
            return "regression"
        return "unknown"

    def _fit(self, data: 'dict[str, ArrayLike]') -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        data : dict
            Dictionary containing the training input samples and the target values.
        """
        start_time_obj = datetime.datetime.now()
        start_time_float = time()

        start_time = datetime.datetime.now()
        self.evolution_logger.info(f"--- PipeGenie Run Started ---")
        self.evolution_logger.info(f"🕒 Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.elite_avg_fitness = np.nan
        self.generations_without_improvement = 0
        self.last_improvement_time = time()
        self.elite = self._create_elite_object()

        self.evolution_logger.info("[2/5] 🌱 Initializing evolutionary process...")
        self._evolve(data)
        
        self.evolution_logger.info("[3/5] ✅ Evolution finished.")

        if self.elite is None or len(self.elite) == 0:
            raise RuntimeError("The elite is empty when it should have, at least, one member")
        
        # --- SAVE RUN TO METABASE ---
        #if self.enable_metalearning and self.metabase is not None:
        if self.metabase is not None and self.save_to_metabase:
            self.evolution_logger.info("    - Saving run results to meta-knowledge base...")
            metafeatures = self.metafeature_calculator.calculate(data["X"], data["y"])
            #self.metabase.save_run(metafeatures, self.elite)
            task_type = self._get_task_type()
            
            self.metabase.save_run(metafeatures, self.elite, task_type)
        # --------------------------

        for ind in self.elite:
            if not hasattr(ind, 'pipeline') or ind.pipeline is None:
                ind.create_pipeline(self.pset)

        self.evolution_logger.info("[4/5] 👑 Generating and evaluating final ensemble...")
        self._generate_ensemble(data)
        self.evolution_logger.info("    - Ensemble models and weights saved to `ensemble.txt`.")
        
        

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        
        self.evolution_logger.info("[5/5] ✨ PipeGenie run complete.")
        # --- LOG FINAL TIME TO BE CAPTURED BY THE FILTER ---
        end_time_obj = datetime.datetime.now()
        elapsed_seconds = time() - start_time_float

        self.evolution_logger.info(f"--- PipeGenie Run Summary ---")
        self.evolution_logger.info(f"Start Time: {start_time_obj.strftime('%Y-%m-%d %H:%M:%S')}")
        self.evolution_logger.info(f"End Time: {end_time_obj.strftime('%Y-%m-%d %H:%M:%S')}")
        self.evolution_logger.info(f"Total Elapsed Time: {elapsed_seconds:.2f} seconds")
        self.evolution_logger.info(f"--- End of Run ---")
        # --------------------------------------------------------
        """ self.evolution_logger.info(f"🕒 End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.evolution_logger.info(f"⏱️ Total Elapsed Time: {elapsed_time}")
        self.evolution_logger.info(f"--- End of Run ---") """
        
        self._close_loggers()




    @abstractmethod
    def _evaluate(
        self,
        ind: 'Individual',
        data: 'dict[str, ArrayLike]',
        start: float,
        queue: 'Queue',
    ) -> tuple:
        """
        Evaluate an individual.

        Parameters
        ----------
        ind : Individual
            Individual to be evaluated.

        data : dict
            Dictionary containing the training input samples and the target values.

        start : float
            Time when the evaluation started.

        queue : Queue
            Queue used to store the logs generated by the worker processes.

        Returns
        -------
        result : tuple
            Tuple containing the fitness values, the prediction and the runtime of the individual.
        """
        raise NotImplementedError("Method '_evaluate' must be implemented in subclass")

    def _evolve(self, data: 'dict[str, ArrayLike]') -> None:
        """
        Evolve the population using the evolutionary process.

        Parameters
        ----------
        data : dict
            Dictionary containing the training input samples and the target values.
        """
        start = time() # To control the timeout

        headers = ["gen", "nevals"]
        headers.extend(self.stats.fields)
        headers.extend(self.stats_elite.fields)

        chapter_headers = {category: self.stats[category].fields
                           for category in self.stats.fields}
        chapter_headers.update({category: self.stats_elite[category].fields
                                for category in self.stats_elite.fields})

        logbook = Logbook(headers=headers, chapter_headers=chapter_headers)

        # --- CALL THE NEW METHOD TO CREATE THE POPULATION ---
        population = self._create_initial_population(data)
        # ----------------------------------------------------

        """ population = [
            Individual(
                self.schema.create_syntax_tree(),
                Fitness((1.0,) if self.maximization else (-1.0,)),
            ) for _ in range(self.pop_size)
        ] """

        chunksize = 1 if self.cpu_count == 1 else ceil((self.pop_size / self.cpu_count) * 0.25)

        manager = Manager()
        q = manager.Queue()
        listener = logging.handlers.QueueListener(q, *self.individuals_logger.handlers)
        listener.start()

        evaluate = partial(self._evaluate, data=data, start=start, queue=q)

        with ProcessPoolExecutor(max_workers=self.cpu_count) as pool:
            results = pool.map(evaluate, population, chunksize=chunksize)

        for ind, result in zip(population, results, strict=True):
            ind.fitness.values, ind.prediction, ind.runtime = result

        valid_population = [ind for ind in population if ind.fitness.valid]

        if len(valid_population) > 0:
            self.elite.update(valid_population)

        record = self.stats.compile(population)
        record_elite = self.stats_elite.compile(self.elite)
        logbook.record(gen=0, nevals=len(population), **record, **record_elite)
        report = logbook.stream

        # This will only be printed to console if verbose=True
        self.evolution_logger.info(report)

        apply_operators = (self._apply_operators_double_mut if self.use_double_mutation
                           else self._apply_operators_cx_mut)

        for gen in range(1, self.generations + 1):
            if (time() - start) > self.timeout:
                self.evolution_logger.warning(f"Timeout of {self.timeout}s reached. Stopping evolution.")
                return

            offspring = apply_operators(population)
            evals = [ind for ind in offspring if not ind.fitness.valid]

            with ProcessPoolExecutor(max_workers=self.cpu_count) as pool:
                results = pool.map(evaluate, offspring, chunksize=chunksize)

            for ind, result in zip(offspring, results, strict=True):
                ind.fitness.values, ind.prediction, ind.runtime = result

            population = self.replacement.replace(population, offspring, self.elite.elite)
            self.elite.update(population)

            record = self.stats.compile(population)
            record_elite = self.stats_elite.compile(self.elite)
            logbook.record(gen=gen, nevals=len(evals), **record, **record_elite)
            report = logbook.stream

            # This will only be printed to console if verbose=True
            self.evolution_logger.info(report)

            if self._should_early_stop():
                return

        listener.stop()

    @abstractmethod
    def _create_ensemble_object(
        self,
        estimators: 'list[tuple[str, Pipeline]]',
        weights: 'ArrayLike',
    ) -> 'BaseVoting':
        """
        Create the ensemble object used to combine the predictions of the estimators.

        Parameters
        ----------
        estimators : list of (str, estimator) tuples
            List of tuples with the estimators to be included in the ensemble.

        weights : list of floats
            List of weights used to combine the predictions of the estimators.

        Returns
        -------
        ensemble : BaseVoting instance
            The ensemble model.
        """
        raise NotImplementedError("Method '_create_ensemble' must be implemented in subclass")

    @abstractmethod
    def _generate_ensemble(self, data: 'dict[str, ArrayLike]') -> None:
        """
        Generate the ensemble model using the individuals of the elite.

        Parameters
        ----------
        data : dict
            Dictionary containing the training input samples and the target values.
        """
        raise NotImplementedError("Method '_generate_ensemble' must be implemented in subclass")

    def _create_elite_object(self) -> 'DiverseElite':
        """
        Create the elite object used to store the best individuals of the population.

        By default, the elite object is a DiverseElite object with a diversity weight of 0.

        Returns
        -------
        elite : DiverseElite
            The elite object.
        """
        return DiverseElite(self.elite_size, div_weight=0)

    def _should_early_stop(self) -> bool:
        """
        Check if the evolutionary process should stop based on the early stopping criteria.

        Returns
        -------
        stop : bool
            Indicates if the evolutionary process should stop.
        """
        if self.early_stopping_generations is None and self.early_stopping_time is None:
            return False

        current_avg_fitness = float(np.mean([ind.fitness.values[0] for ind in self.elite]))

        if (current_avg_fitness - self.elite_avg_fitness) < self.early_stopping_threshold:
            self.generations_without_improvement += 1

            if self.early_stopping_generations is not None and \
                self.generations_without_improvement >= self.early_stopping_generations:
                msg = (f"Early stopping due to no improvement in the elite average fitness for "
                       f"{self.generations_without_improvement} generations")
                self.evolution_logger.warning(msg)
                return True

            current_time = time()

            if self.early_stopping_time is not None and \
                (current_time - self.last_improvement_time) >= self.early_stopping_time:
                msg = (f"Early stopping due to no improvement in the elite average fitness for "
                       f"{current_time - self.last_improvement_time:.2f} seconds")
                self.evolution_logger.warning(msg)
                return True
        else:
            self.generations_without_improvement = 0
            self.elite_avg_fitness = current_avg_fitness
            self.last_improvement_time = time()

        return False

    def _apply_operators_cx_mut(self, population: list['Individual']) -> list['Individual']:
        offspring = self.selection.select(population, len(population))
        offspring = [ind.clone() for ind in offspring]

        for i in range(1, len(offspring), 2):
            offspring[i - 1], offspring[i], modified = self.crossover.cross(
                offspring[i - 1],
                offspring[i],
                self.schema,
            )

            if modified:
                offspring[i - 1].reset()
                offspring[i].reset()

        for i, _ in enumerate(offspring):
            offspring[i], modified = self.mutation.mutate(offspring[i], self.schema)

            if modified and hasattr(offspring[i], "pipeline"):
                offspring[i].reset()

        return offspring

    def _apply_operators_double_mut(self, population: list['Individual']) -> list['Individual']:
        offspring = [ind.clone() for ind in population]

        for i, _ in enumerate(offspring):
            if offspring[i] in self.elite:
                offspring[i], modified = self.mutation_elite.mutate(offspring[i], self.schema)
            else:
                offspring[i], modified = self.mutation.mutate(offspring[i], self.schema)

            if modified and hasattr(offspring[i], "pipeline"):
                offspring[i].reset()

        return offspring

    def save_model(self, filename: str) -> None:
        """
        Save the created ensemble to a file.

        Parameters
        ----------
        filename : str
            Name of the file where the ensemble will be saved.
        """
        # Check if the ensemble model has been created
        if not hasattr(self, "ensemble"):
            raise RuntimeError("The ensemble model has not been created yet")

        # Ensure the directory exists
        file_path = Path(self.outdir_path) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with self.outdir_path.joinpath(filename).open("wb") as file:
            pickle.dump(self.ensemble, file)

# --- ADD THIS NEW FILTER CLASS AT THE END OF THE FILE ---
class TableAndTimeFilter(logging.Filter):
    """A logging filter that allows only multiline table-like messages or time summaries."""
    def filter(self, record):
        msg = record.getMessage()
        # Allow messages that are multi-line (our stats table)
        is_table = '\n' in msg
        # Allow messages that contain keywords for the final summary
        is_summary = "Time:" in msg or "Elapsed Time:" in msg
        
        return is_table or is_summary
# -------------------------------------------------------