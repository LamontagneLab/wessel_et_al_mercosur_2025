#!/bin/bash

#SBATCH --job-name=GP_DB
#SBATCH -n 1
#SBATCH --time=9:00:00
#SBATCH --mem=32GB
#SBATCH --output=outfiles/%A.%a.out
#SBATCH --error=errfiles/%A.%a.err

#run this script with: sbatch <filename>

# assumes virtual environment "my_genv" is created" and gurobi is available as a module
source ../../my_genv/bin/activate
module load gurobi/10.0.1
date


#Step1: Create database (can specify name, defaults to io.db)
cd ../db
python create_database.py --database ../db/io.db


#Step2: import csvs into database (stay at: .\db):
python utilities/port_csvs_to_db.py --database io.db --csv_location csvs_12day_p2


#Step3: import scenarios from database (imports all unless specified)
cd utilities
python scenario.py --database ../io.db --csv_path ../scenarios.csv --csv_location csvs_12day_p2
