#!/bin/bash

#SBATCH --job-name=GP_merc
#SBATCH -n 2
#SBATCH --time=72:00:00
#SBATCH --mem=90GB
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --output=outfiles/%A.%a.out
#SBATCH --error=errfiles/%A.%a.err
#SBATCH --array=1-80%1

#run this script with: sbatch <filename>
# MAKE SURE ARRAY OPTION IS UNCOMMENTED IF NEEDED
# MAKE SURE TO UPDATE scenario_list AND RUN prep_model_scenarios BEFORE THIS SCRIPT

# assumes virtual environment "my_genv" is created" and gurobi is available as a module
source ../../my_genv/bin/activate
module load gurobi/10.0.1
date

# sleep in between running scenarios if scheduler unable to handle job array
#x=${SLURM_ARRAY_TASK_ID}
#m=2000
#sleep $(((x - 1) * m))

#Get scenarios (assumes "scenario_list" is a plain text file where each row contains a scenario name)
readarray -t SCENARIOS < ../run_scripts/scenario_list
SCENARIO_NAME=${SCENARIOS[$SLURM_ARRAY_TASK_ID-1]}

#Step4: load scenario inputs to the database (can specify a single scenario name if running just one scenario)
cd ../gridpath
python get_scenario_inputs.py --database ../db/io.db --scenario $SCENARIO_NAME

#Step5: Run GridPath for the desired scenario (specify solver, scenario, and database name)
cd ../gridpath
python run_end_to_end.py --scenario $SCENARIO_NAME --database ../db/io.db --solver gurobi
date