.. EngiOptiQA documentation master file, created by
   sphinx-quickstart on Thu Feb 15 10:49:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EngiOptiQA's documentation!
======================================

*EngiOptiQA* is a software project that allows to perform **Engi**\neering **Opti**\mization with **Q**\uantum **A**\lgorithms.

Overview
========

*EngiOptiQA* consists of three main modules:

1. :ref:`Solvers Module<solvers-module>`
2. :ref:`Problems Module<problems-module>`
3. :ref:`Variables Module<variables-module>`

.. _solvers-module:

Solvers Module
================

This module contains solvers for *Polynomial Unconstrained Binary Optimization (PUBO)* problems.
There are currently two types of solvers available: *annealing solvers* and *variational quantum algorithms (VQAs)*.
Furthermore, the module includes a *brute-force solver* for testing purposes.
For the annealing solvers, there are two alternative implementations based on the `Fixstars Amplify SDK <https://amplify.fixstars.com/en/sdk>`_ or the `D-Wave Ocean SDK <https://www.dwavesys.com/solutions-and-products/ocean/>`_, respectively.
Please note that the D‑Wave implementation requires transforming the problem into a *Quadratic Unconstrained Binary Optimization (QUBO)* formulation before solving.
For the VQAs, there is an implementation of the *Quantum Approximate Optimization Algorithm (QAOA)* based on `PennyLane <https://pennylane.ai/>`_.


.. toctree::
   :maxdepth: 2

   annealing_solvers

.. _problems-module:

Problems Module
===============

This modules provides classes for setting up *PUBO* problems.
There exists a general abstract problem class, which can be used for the implementation of specific problem formulations.
So far, this includes structural problems such as the analysis and design optimization of a one-dimensional rod or truss
structures.

.. toctree::
   :maxdepth: 3

   problems/problems

.. _variables-module:

Variables Module
================

This module includes classes for the representation of variables in logical qubits.
At the moment, there exist different options for representing real-valued variables.

.. toctree::
   :maxdepth: 2

   variables

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




