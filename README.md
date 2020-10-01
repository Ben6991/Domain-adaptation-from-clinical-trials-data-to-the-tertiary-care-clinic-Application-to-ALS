# Domain adaptation from clinical trials data to the tertiary care clinic Application to ALS
In this repository, you can find all the code necessary to execute the 3 experiments explained in the paper.
The scripts to run the pipeline for each database and model are placed under the 'ModelComparison' folder.
Besides basic python libraries, it is necessary to import also the python file called 'TSFunctions", which contains many fundamental functions for processing the temporal data.

For example,

import TSFunctions as ts
ts.some_function()

Note that TASMC's data is confidential. So, only the processed PRO-ACT database is given in this repository.
