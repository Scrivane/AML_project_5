# AML_project_5



This project is organized into **6 main folders**, each containing different parts of the code.  

### 1. Install Dependencies
The project uses **Poetry** for dependency management. To install Poetry, follow the official guide:  
https://python-poetry.org/docs/  

Once installed, navigate to the project’s root directory and run:

poetry install 

This will install all required dependencies.

### 2. Run code

To run the code use VS Code with the Jupyter extension.

Open the desired notebook (.ipynb) and either:

Click Run All, or

Run specific sections as needed.

Always make sure you are in the correct directory before running a notebook. For example:
cd ./Cifar_100_Federated
This command moves you into the Cifar_100_Federated directory.

### 3. project structure

The main code directories are:

Cifar_100_Baseline (contains baseline_cifar_config_search.ipynb used for param search  and baseline_cifar_search_best_scheduler.ipynb  to test different schedulers  )

Cifar_100_Federated (contains Heterogeneous_checkpoints_federated.ipynb  to run test in a federated setting with different shardings, J values and different levels of skewness )

Shakespeare_Baseline (contains federated baseline shakespeare_baseline_config_search.ipynb for param search  and shakespeare_baseline_best_scheduler.ipynb to test different schedulers  )

Shakespeare_Federated ( contains Shakespeare_iid_and_non_iid_final.ipynb to run test in a federated setting with different shardings, J values and different levels of skewness)

graph  (which contains scripts to generate summary graphs)

/TODO → reserved for personal contributions

Each folder contains:

Jupyter notebooks (.ipynb) that can be run, logs of previous runs, which are also used for generating summary graphs and saved last run checkpoints.
