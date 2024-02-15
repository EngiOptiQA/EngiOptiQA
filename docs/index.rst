.. EngiOptiQA documentation master file, created by
   sphinx-quickstart on Thu Feb 15 10:49:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EngiOptiQA's documentation!
======================================

*EngiOptiQA* is a software project that allows to perform **Engi**\neering **Opti**\mization with **Q**\uantum **A**\nnealing.

Overview
========

*EngiOptiQA* consists of three main modules:

1. :ref:`Annealing Solvers Module<solvers-module>`
2. :ref:`Problems Module<problems-module>`
3. :ref:`Variables Module<variables-module>`

.. _solvers-module:

Annealing Solvers Module
========================

This module contains solvers for *Quadratic Unconstrained Binary Optimization (QUBO)* problems.
There are two alternative implementations based on the `Fixstars Amplify SDK <https://amplify.fixstars.com/en/sdk>`_ or the `DWave Ocean SDK <https://www.dwavesys.com/solutions-and-products/ocean/>`_, respectively.

.. toctree::
   :maxdepth: 2

   annealing_solvers

.. _problems-module:

Problems Module
===============

This modules provides classes for setting up *QUBO* problems. 
So far, this includes the one-dimensional problem of a rod.

.. toctree::
   :maxdepth: 2

   problems

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




