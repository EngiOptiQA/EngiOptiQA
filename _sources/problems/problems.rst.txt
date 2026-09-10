.. Problems Overview

============================
Problems
============================

.. automodule::  engioptiqa.problems

Abstract Problem Class
---------------------------------

The abstract `Problem` class serves as the base for all problem formulations within the *EngiOptiQA* framework.
It defines the common interface and structure that concrete problem classes must implement.
This includes methods for:

1. **Discretization of problem variables:**

  - Encode continuous or discrete problem variables in terms of binary variables.

2. **Generation of problem formulation:**

  - Build an *Amplify* `Model` (based on a polynomial in binary variables) that represents the objective and constraints.
  - The model is then provided to the any *EngiOptiQA* solver for optimization.

Furthermore, specific problem formulations can override existing methods, e.g., for post-processing and analyzing
results returned by the solvers, such as:

- **Analysis of results:**

  - Post-process the raw solution (binary variables) returned by the solver. This may involve decoding the binary
    solution back into the original problem variables, evaluating the objective function or other quantities of interest.


.. currentmodule:: engioptiqa.problems.problem

.. autoclass:: Problem
   :members:

Specific Problem Formulations
-----------------------------

.. toctree::
   :maxdepth: 1

   structural/structural


