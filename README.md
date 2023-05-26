# Nonlinear Time Series Analysis

Version 0.1.0

## Description
A program for nonlinear time series analaysis. This program includes example chaotic systems (currently Lorenz and Thomas system of ODEs), for which sample trajectories can be simulated and analyzed. 
Includes plotting functions (e.g. autocorrelation, recurrence plots). More functionalities (e.g. state space reconstruction, sparse identification of nonlinear dynamics) are under construction.

## Support

Having problems with this package? Send an email to f.slegers@uu.nl.

## Roadmap

This program is still under construction. Future releases will include more examples of dynamical systems, preprocessing steps for time series analysis and more 
methods for nonlinear time series analysis, like state space reconstruction and spare identification of nonlinear dynamics.

## Project organization

```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── requirements.txt
├── bin                <- Compiled and external code, ignored by git (PG)
│   └── external       <- Any external source code, ignored by git (RO)
├── config             <- Configuration files (HW)
├── data               <- All project data, ignored by git
│   ├── processed      <- The final, canonical data sets for modeling. (PG)
│   ├── raw            <- The original, immutable data dump. (RO)
│   └── temp           <- Intermediate data that has been transformed. (PG)
├── docs               <- Documentation notebook for users (HW)
│   ├── manuscript     <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports        <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── results
│   ├── figures        <- Figures for the manuscript or reports (PG)
│   └── output         <- Other output for the manuscript or reports (PG)
└── src                <- Source code for this project (HW)

```

## Installation

This project is created in Python version 3.1.1.

### Requirements

The requirements.txt file contains the packages this program depends on. 

## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)


## Citation

Please [cite this project as described here](/CITATION.md).

