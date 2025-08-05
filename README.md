# GridPath-Mercosur Regional Coordination 2025
Data and code for forthcoming work "Regional coordination can alleviate the cost burden of a low-carbon electricity system".

This repository contains model input data, processed model output data, and all original code for data processing, analysis, and figure generation.

Model scenario outputs have been deposited at Zenodo and are publicly available as of the date of publication at https://doi.org/10.5281/zenodo.15096839.

The GridPath model is open source and is accessible at https://doi.org/10.5281/zenodo.5822994; the specific model version utilized in this paper is accessible at https://github.com/blue-marble/gridpath/releases/tag/v0.8.1.


## Repository overview (within data_inputs_scripts.zip)

| Item  | Purpose        |
|-------|----------------|
| model_inputs/	 | Contains the "csvs_12day_p2" folder which holds all pre-processed data inputs necessary to reproduce the GridPath scenarios run in this experiment, except for the Stranded Coordination appendix. Files should be removed from the compressed archive prior to running the model. |
| model_run_scripts/ | Contains scripts and scenario setup files required to reproduce the experiment. |
| figure_data/  | Post-processed model outputs necessary to generate the figures presented in this analysis. For full, raw model outputs, see the associated Zenodo repository. |
| stranded_coord_appendix/  | Self-contained set of extra model inputs, post-processed output data, and figure generation scripts necessary to reproduce the Stranded Coordination sensitivity analysis. |
| main_data_prep.py  | Python code necessary to reproduce the analysis. This script takes the model output available in Zenodo and processes it into the data contained in figure_data/ for figure generation. |
| main_figures.py  | Python code to generate figures shown in this study. |

## Running the model and reproducing the experiment

The files and scripts presented in the model_inputs/ and model_run_scripts/ folder, along with a compatible version of GridPath can be used to fully reproduce the experiment. See above for the model version used in this paper.

For first time users of GridPath, extensive documentation and setup guides are available on [Github](https://github.com/blue-marble/gridpath) and [Read the Docs](https://gridpath.readthedocs.io/). Reference this documentation for necessary software dependencies.

GridPath may take 30-60 minutes to install. The GridPath scenarios used in this experiment generally require the use of large computing resources to fully reproduce, but the model includes several [example problems](https://gridpath.readthedocs.io/en/latest/usage.html#examples) which can be run on an average desktop computer. Depending on the sample problem tested, this can be run in seconds to minutes.

Note also that a mathematical solver is required to run GridPath. There are many commercial solvers available; in this study, Gurobi v10.0.1 is used. Gurobi is freely available for academic use. Although more powerful solvers are generally recommended for reproducing this experiment, the demo problems referenced above may be solved with simpler methods.

Generally, running GridPath consists of: 
1. preparing scenario-specific input data and formatting to match the "csvs" folder. Here, the "csvs_12day_p2" contains experiment-specific data.
2. preparing a scenarios.csv file containing all scenario configurations.
3. creating a new database, importing input csv's into the database, and creating scenarios from the built database. The script model_run_scripts/prep_model_scenarios.sh completes these tasks.
4. loading the specific scenario inputs into the database and running GridPath for the desired scenario(s). The script model_run_scripts/run_model_scenarios.sh completes these tasks. The expected output is a folder of csv files containing the model outputs.
