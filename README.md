# Linear Regression Project - Red Wine Analysis

![image](https://github.com/user-attachments/assets/dab599b8-ad0d-4d41-90fc-66cfcc13f85c)


## Introduction

This project contains the analysis (Linear Regression) of the Red Wine dataset, downloaded from:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data as a `winequality-red.csv` file.

The analysis focuses on building an explanatory linear regression model to identify the key factors that determine wine quality ratings. This multivariate analysis examines relationships between:

**Chemical Properties:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, sulfur dioxide levels, density, pH, sulphates, and alcohol content <br>
**Quality Ratings:** Expert evaluations on a scale from 0-10 (actual range 3-8 in the dataset)

The primary goal is to quantify how specific wine chemistry characteristics impact quality perceptions, providing insights that could inform winemaking decisions and quality control processes.

## Dataset Information

The Wine Quality dataset contains physicochemical and sensory data for 1,599 samples of Portuguese "Vinho Verde" red wine. This dataset was created by Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos, and José Reis in 2009, as documented in their paper "Modeling wine preferences by data mining from physicochemical properties." (https://www.sciencedirect.com/science/article/pii/S0167923609001377)

**1. fixed acidity:** The concentration of non-volatile acids in the wine (primarily tartaric acid), measured in g/dm³. These acids come primarily from grapes and remain relatively stable throughout the winemaking process.

**2. volatile acidity:** The amount of acetic acid in the wine, measured in g/dm³. High levels of volatile acidity can lead to an unpleasant vinegar taste.

**3. citric acid:** The concentration of citric acid in the wine, measured in g/dm³. Citric acid can add freshness and flavor to wines when present in small quantities.

**4. residual sugar:** The amount of sugar remaining after fermentation stops, measured in g/dm³. It's rare to find wines with less than 1 g/L, and wines with greater than 45 g/L are considered sweet.

**5. chlorides:** The salt content in the wine, measured in g/dm³. High chloride concentrations can give wine a salty taste.

**6. free sulfur dioxide:** The free form of SO₂ that exists in equilibrium between molecular SO₂ and bisulfite ion, measured in mg/dm³. It prevents microbial growth and oxidation.

**7. total sulfur dioxide:** The sum of free and bound forms of SO₂, measured in mg/dm³. In low concentrations, SO₂ is mostly undetectable in wines, but at free SO₂ concentrations over 50 ppm, it becomes evident in the nose and taste of wine.

**8. density:** The density of the wine, measured in g/cm³. This is close to that of water depending on the alcohol and sugar content.

**9. pH:** Describes how acidic or basic the wine is on a scale from 0 (very acidic) to 14 (very basic). Most wines are between 3-4 on the pH scale.

**10. sulphates:** The amount of potassium sulphate in the wine, measured in g/dm³. Sulphates act as an antimicrobial and antioxidant.

**11. alcohol:** The percentage of alcohol content in the wine by volume.

**12. quality:** The target variable, based on sensory data. It's the median of at least 3 evaluations made by wine experts, rated on a scale from 0 (very bad) to 10 (excellent).


## Target Variable, Modeling Approach 📊 and Hypothesis Testing Framework 🧪

The target variable is **wine quality** (`quality`) representing expert sensory evaluations based on the median rating from at least three wine experts. This ordinal variable ranges from 3 to 8 in our dataset, with most wines rated between 5-6.

**<ins>Model Development Strategy:</ins>**

  **1. Initial Model:** Focus on hypothesis-driven variables (alcohol and volatile acidity) <br>
  **2. Expanded Model:** Use backward selection to identify additional significant predictors <br>
  **3. Model Comparison:** Evaluate both approaches to understand incremental value of additional variables

**<ins>Hypothesis Testing:</ins>**

* **Null Hypothesis (H₀):** There is no relationship between alcohol content, volatile acidity, and wine quality (the coefficients of alcohol and volatile acidity in predicting wine quality equal zero). <br>

* **Alternative Hypothesis (H₁):** Higher alcohol content and lower volatile acidity are independently associated with higher perceived wine quality.


## 🚀 Analytical Plan

**1. Data Loading and Preparation:**

  * Load Portuguese red wine quality dataset (1,599 samples)
  * Remove duplicate observations to ensure data integrity
  * Apply outlier detection and removal using z-score methodology
  * Validate data quality and completeness <br />

**2. Exploratory Data Analysis (EDA):**

  * Analyze distribution of wine quality ratings and chemical properties
  * Create correlation matrix to identify relationships between variables
  * Examine multicollinearity using Variance Inflation Factor (VIF)
  * Visualize key relationships between predictors and wine quality <br />

**3. Hypothesis Formulation and Testing:**

  * Formulate hypothesis: Higher alcohol content and lower volatile acidity independently predict higher wine quality
  * Split data into 80% training and 20% testing sets
  * Build initial linear regression model with hypothesis variables (alcohol, volatile acidity)
  * Test statistical significance using p-values and confidence intervals <br />

**4. Model Development and Selection:**

  * Apply backward selection starting with correlated variables
  * Systematically remove non-significant predictors (p > 0.05)
  * Compare initial hypothesis-driven model vs. expanded backward selection model
  * Evaluate model assumptions through residual analysis and diagnostic plots <br />

**5. Model Validation and Performance:**

  * Test both models on hold-out dataset to assess generalizability
  * Calculate R-squared and RMSE metrics for model comparison
  * Create actual vs. predicted visualizations
  * Compare model performance on raw vs. cleaned data <br />

**6. Results Interpretation and Conclusions:**

  * Interpret coefficient magnitudes, directions, and confidence intervals
  * Assess practical significance of chemical properties on wine quality
  * Address original hypothesis with statistical evidence
  * Identify limitations and suggest improvements for future analysis
  * Provide actionable insights for wine production and quality assessment
 
## Project Files and Structure

* `red_wine_functions.py`: This is a Python module contains all the visualization and analysis functions needed that can be called and run in the notebook file.
* `Red_Wine_Analysis.ipynb`: A complete Jupyter notebook for performing the analysis with proper statistical testing and visualization.
* `winequality-red.csv`: The raw data file needed for the analysis that needs to be loaded in the analysis notebook.

## Prerequisites

* Python 3.x
* Required Python packages:
  * pandas
  * numpy
  * seaborn
  * matplotlib
  * statsmodel
  * plotly
  * scikit-learn
* Jupyter Notebook

## Requirements

### Installation Instructions and Cloning the Repository

Follow these steps to set up the project environment and install the required dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/LauraLechermann/Linear-Regression-Project__Red-Wine-Analysis.git
    ```
2. Navigate to the project directories:
    ```bash
    cd Red_Wine_Analysis
    ```
3. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. You are now ready to run the project!
   
7. Follow these steps to open and run the Jupyter Notebook:
   
   Start the Jupyter Notebook by running the following command in your terminal:
   ```bash
     jupyter notebook Red_Wine_Analysis.ipynb
   ```
 This will open Jupyter Notebook in your default web browser.


## Importing the original dataset into Jupyter Notebooks for Analysis:

* Download the Red Wine dataset from: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data as a `winequality-red.csv` file and save the file in the same directory as the Jupyter Notebook file
* Load the dataset into a DataFrame in Jupyter Notebooks with the necessary packages:
```bash
import pandas as pd
# Load dataset into Pandas DataFrame
df = pd.read_csv('winequality-red.csv')

#Display the first 5 rows of the dataset
df.head()
```
* Proceed with data inspection lookiing for duplicates, missing values and outliers before proceeding with the exploratory data analysis (EDA)

## Visualizations

The Jupyter Notebook contains visualizations and graphs plotted with funtions that can be found in a separate `red_wine_functions.py` file. Each visualation function is called separately in the Jupyter Notebook file, e.g. when visualizing the distribution of players in each variant:

```bash
viz.plot_feature_distribution(df3)
```
If needed, `filename='feature_distribution.png` can be used to save each plot as a png. file for further use. If this is not needed it can be removed.

When running the Jupyter Notebook file, make sure the `red_wine_functions.py` function is in the same directory as the Jupyter Notebook file to run the analysis successfully!
