# Glacier Thickness Prediction

This project aims to develop a model for predicting glacier thickness using multi-source observation data. The data comes from the **GLATHIDA** database, and the project includes preprocessing, model training, hyperparameter tuning, and model interpretability analysis.

## Project Structure

The directory structure of this project is organized as follows:

```

GlacierThicknessPrediction/
│
├── data/                        # Directory for storing the original data files
│   ├── glathida\_on\_grid\_RGI01.csv
│   ├── glathida\_on\_grid\_RGI02.csv
│   ├── ...                      # Other RGI data files
│
├── output/   # Output directory for data processing results
│   ├── configs/                  # Configuration files folder
│   ├── global\_all.csv        # Raw data file containing all records
│   ├── global\_clean.csv      # Cleaned data file
│   ├── global\_outliers.csv   # Data file with outliers removed
│   └── runs/                     # Folder for run results
│       └── exp\_glathida\_v1/      # Output files for the current experiment
│           ├── config\_snapshot.yaml   # Configuration snapshot file
│           ├── feature\_list.json      # Feature list
│           ├── feature\_report.csv    # Feature report
│           ├── processing\_log.json   # Processing log
│           ├── rgi\_counts.csv        # RGI data count
│           ├── rgi\_removed\_summary.csv  # Summary of removed data
│           └── scaler.pkl           # Standardizer
├── runs\_demo/  # Directory for storing outputs from subsequent stages of modeling
│   ├── best\_params\_global.json    # Best global parameters for model tuning
│   ├── df\_meta.csv                # Metadata for the dataset
│   ├── figs/                      # Folder for figures
│   ├── logs\_global.csv            # Logs for global experiment
│   ├── s1\_manifest.json           # Manifest for the current experiment
│   ├── stage3\_supervised\_fast\_thesis/  # Output of stage 3 experiment
│   ├── stage4\_best\_blend/         # Output of stage 4 best blend
│   ├── stage5\_clean\_vs\_all/       # Comparison results between clean and all data
│   └── stage6\_explain/            # Explanation results of the model
├── README.md                    # Project overview document
├── requirements.txt
├── Hyperparameter\_Tuning\_with\_Comments.ipynb  # Notebook for hyperparameter tuning with comments
├── Hyperparameter\_Tuning.ipynb  # Notebook for hyperparameter tuning
├── Model\_Interpretability\_Visualization.ipynb  # Notebook for model interpretability visualization
├── Modeling\_Pipeline\_and\_Interpretability.ipynb  # Notebook for modeling pipeline and interpretability
└── Data\_Processing.ipynb        # Jupyter Notebook for data preprocessing

```

## Description

This project involves several key steps:

1. **Data Preprocessing**: 
   - The raw data from **GLATHIDA** is preprocessed, including cleaning, handling missing values, outlier detection, and feature engineering.
   - Data files are processed and stored in the `output/` directory, such as `global_all.csv`, `global_clean.csv`, and `global_outliers.csv`.

2. **Model Training**: 
   - The project uses various machine learning models such as **XGBoost**, **LightGBM**, **CatBoost**, and **MLP** for glacier thickness prediction.
   - The training and hyperparameter tuning results are stored in the `runs_demo/` directory.

3. **Model Interpretation**: 
   - The models are interpreted using SHAP values, feature importance, and other explainability techniques to understand the predictions and feature relationships.
   - The `stage6_explain/` folder contains the results and explanations for model predictions.

## Data Sources

The data used in this project comes from the **GLATHIDA** database, which provides glacier thickness measurements across various regions globally. The data is divided into multiple files for different regions, such as:

- `glathida_on_grid_RGI01.csv`
- `glathida_on_grid_RGI02.csv`
- ...

These files are stored in the `data/` directory.

## Setup

### Prerequisites

To run this project, you need to have the following dependencies installed:

* Python 3.x
* Jupyter Notebook
* Required libraries:

  * `numpy`
  * `pandas`
  * `scikit-learn`
  * `xgboost`
  * `lightgbm`
  * `catboost`
  * `shap`
  * `matplotlib`
  * `seaborn`
  * `optuna`
  * `torch`
  * `joblib`
  * `pyyaml`
  * `scipy`

You can install the required libraries using `pip`:

```
pip install -r requirements.txt
```

### Running the Notebooks

1. **Data Processing**: 
   - The notebook `Data_Processing.ipynb` is used to preprocess the data, including cleaning and feature engineering. You can run this notebook to prepare the data for modeling.

2. **Hyperparameter Tuning**:
   - The notebooks `Hyperparameter_Tuning_with_Comments.ipynb` and `Hyperparameter_Tuning.ipynb` perform hyperparameter tuning using **Optuna** to find the best parameters for the models.

3. **Modeling and Interpretability**:
   - The notebooks `Model_Interpretability_Visualization.ipynb` and `Modeling_Pipeline_and_Interpretability.ipynb` build the models and interpret them using various techniques such as SHAP values and feature importance.

## Output

The processed data, model results, and logs will be saved in the `output/` and `runs_demo/` directories.

### Example output files:
- `output/configs/global_all.csv` - The full dataset before cleaning.
- `output/configs/global_clean.csv` - The dataset after cleaning.
- `runs_demo/stage3_supervised_fast_thesis/` - Output for the stage 3 experiment (supervised model).
- `runs_demo/stage4_best_blend/` - Output for the best model blend.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the **GLATHIDA** dataset and various machine learning models to predict glacier thickness. Special thanks to the contributors and researchers who made the data publicly available.

---

Feel free to modify the content further according to your specific requirements.
```

