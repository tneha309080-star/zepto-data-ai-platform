# Zepto Data AI Platform

A three-module data and AI capstone project covering web scraping and data engineering, exploratory data analysis and machine learning, and an AI support assistant.

## Project Structure

```text
zepto-data-ai-platform/
│
├── data_pipeline/
│   ├── scrape_books.py
│   ├── books_clean.csv
│   ├── books.db
│   └── sql_outputs.txt
│
├── analytics/
│   ├── analysis.py
│   ├── titanic.csv
│   ├── best_titanic_pipeline.joblib
│   └── analysis outputs and charts
│
├── support_assistant/
│   └── support assistant files
│
├── README.md
└── requirements.txt
```

---

# 1. Setup

Clone the repository and move into the project directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 2. Module 1 — Data Pipeline

## Objective

Scrape book information from `books.toscrape.com`, clean the data, store it in a normalized SQLite database, and demonstrate SQL and pandas analysis.

## Run

From the project root:

```bash
python data_pipeline/scrape_books.py
```

Then:

```bash
python data_pipeline/database.py
```

## Data captured

The scraper collects:

* Book title
* Price in GBP
* Star rating
* Availability
* Category

The pipeline collects at least 60 books across multiple categories.

## Cleaning

The raw price is converted to a floating-point value.

Star ratings such as `One`, `Two`, `Three`, `Four`, and `Five` are converted to integers from 1–5.

Availability is converted into an `in_stock` Boolean field.

Parsing failures are handled without crashing the pipeline.

## GBP to INR conversion

A fixed project baseline is used:

```text
1 GBP = 105.50 INR
```

Therefore:

```text
price_inr = price_gbp × 105.50
```

No external exchange-rate API is required.

## Database

SQLite is used with normalized tables for:

* `categories`
* `books`

The `books.category_id` field references `categories.category_id`.

The pipeline demonstrates SQL operations including:

* SELECT / WHERE
* ORDER BY
* LIMIT
* DISTINCT
* IN
* BETWEEN
* JOIN
* GROUP BY / aggregate functions

SQL query strings and outputs are stored in:

```text
data_pipeline/sql_outputs.txt
```

Selected SQL results are also loaded into pandas with `pd.read_sql()`.

The JOIN result is independently reproduced using `pd.merge()` on in-memory DataFrames.

---

# 3. Module 2 — Analytics and Machine Learning

## Dataset

The classic Titanic dataset is loaded with Seaborn and saved locally as:

```text
analytics/titanic.csv
```

The saved CSV provides an offline reproducibility fallback.

The cleaned dataset contains 889 observations after removing two rows with missing embarkation information.

## Missing-value handling

The original dataset contains:

| Column        | Missing | Percentage | Treatment         |
| ------------- | ------: | ---------: | ----------------- |
| `age`         |     177 |     19.87% | Median imputation |
| `embarked`    |       2 |      0.22% | Drop rows         |
| `embark_town` |       2 |      0.22% | Drop rows         |
| `deck`        |     688 |     77.22% | Drop column       |

The rules used were:

* Below 5% missing → drop affected rows
* 5–30% missing → impute
* Above 30% missing → explicitly remove the column when appropriate

The `age` median used for imputation was 28.0.

## Exploratory Data Analysis

The analysis includes:

* Dataset information
* Descriptive statistics
* Shape and missing-value analysis
* Age histogram and box plot
* Fare histogram and box plot
* IQR outlier analysis
* Fare mean, median, mode, and skewness
* Survival rates by sex
* Survival rates by passenger class
* Survival rates by sex and passenger class
* Boolean masking using `&` and `|`
* Six-variable correlation matrix
* Correlation heatmap
* Four multivariate visualizations
* EDA-only z-score standardization

The six-variable correlation analysis uses:

```text
survived
pclass
age
sibsp
parch
fare
```

The two strongest absolute correlations were:

* `pclass` and `fare`: approximately -0.5482
* `sibsp` and `parch`: approximately 0.4145

The EDA z-score transformation is used only for exploratory analysis and is not fed into the modeling pipeline.

## Classification

A stratified 80/20 train-test split is used so that the class distribution is maintained between training and testing data.

The modeling preprocessing:

* imputes numeric missing values with the median
* imputes categorical missing values with the most frequent value
* standardizes numeric features
* one-hot encodes categorical features
* fits preprocessing only on training data

Three classifiers were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

### Original classifier results

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8146 |    0.7966 | 0.6912 | 0.7402 |  0.8680 |
| Decision Tree       |   0.7921 |    0.8163 | 0.5882 | 0.6838 |  0.8248 |
| Random Forest       |   0.7865 |    0.7344 | 0.6912 | 0.7121 |  0.8146 |

Confusion matrices and ROC curves are saved in the `analytics/` directory.

The Decision Tree is also visualized using `plot_tree()` with feature and class labels.

## Class imbalance experiment

The following approaches were compared:

| Method                | Precision | Recall |     F1 |
| --------------------- | --------: | -----: | -----: |
| Baseline              |    0.7966 | 0.6912 | 0.7402 |
| Class Weight Balanced |    0.7879 | 0.7647 | 0.7761 |
| SMOTE                 |    0.7937 | 0.7353 | 0.7634 |

Class weighting produced the largest increase in recall and F1 in this experiment, while precision decreased slightly.

SMOTE was applied only within the training pipeline, leaving the test set untouched.

## Random Forest Grid Search

Random Forest hyperparameters were tuned using 5-fold `GridSearchCV`.

The search covered:

* `n_estimators`
* `max_depth`
* `max_features`

Best parameters:

```text
n_estimators = 300
max_depth = 5
max_features = sqrt
```

Best cross-validation F1:

```text
0.7590
```

Out-of-bag score:

```text
0.8256
```

Tuned Random Forest test results:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8146 |
| Precision | 0.8182 |
| Recall    | 0.6618 |
| F1        | 0.7317 |
| ROC-AUC   | 0.8373 |

The GridSearch results are saved to:

```text
analytics/random_forest_grid_search_results.csv
```

## Regression

A multivariate Linear Regression model predicts passenger fare using the other available passenger features.

Results:

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 18.3735 |
| RMSE        | 41.2921 |
| R²          |  0.3609 |
| Adjusted R² |  0.2558 |

The model explains approximately 36.1% of the variance in the test-set fare values.

The residual plot is saved as:

```text
analytics/fare_regression_residuals.png
```

The residual plot should be reviewed to assess whether the residual spread appears approximately constant or shows a funnel-like pattern indicative of heteroscedasticity.

## Final classifier

The standard Logistic Regression pipeline is used as the saved deployment pipeline.

It achieved:

* Accuracy: 0.8146
* F1: 0.7402
* ROC-AUC: 0.8680

The class-weight-balanced version is also documented because it produced higher recall and F1 in the imbalance experiment.

The complete fitted Logistic Regression pipeline, including preprocessing and classifier, is saved as:

```text
analytics/best_titanic_pipeline.joblib
```

The saved pipeline was reloaded successfully and generated a prediction directly from raw feature data without manually applying preprocessing.

---

# 4. Module 3 — Support Assistant

The `support_assistant/` module contains the AI support-assistant implementation.

Run it according to the instructions contained in that module.

---

# 5. Reproducibility

The project uses a consolidated:

```text
requirements.txt
```

The Titanic dataset is saved locally after initial loading so that subsequent analysis can use the same dataset without requiring another dataset download.

Generated charts, CSV outputs, SQL results, and the fitted machine-learning pipeline are stored under their respective module directories.

---

# 6. Design Decisions

### Data pipeline

Requests and BeautifulSoup are used for web scraping. SQLite provides a lightweight relational database with a normalized category/book relationship.

### Analytics

EDA preprocessing is kept separate from machine-learning preprocessing. The ML preprocessing is encapsulated in scikit-learn pipelines so that transformations are fitted only on training data.

### Model evaluation

Classification models use the same stratified test split for comparable evaluation. Accuracy, precision, recall, F1, and ROC-AUC are reported together because each metric captures a different aspect of classifier performance.

Class imbalance is explicitly evaluated using baseline, class weighting, and SMOTE.

### Deployment

The final classifier is stored as one complete pipeline containing both preprocessing and the estimator. This allows raw input data to be passed directly to the reloaded model.

---

# 7. Running the Project

From the repository root:

```bash
python data_pipeline/scrape_books.py
python data_pipeline/database.py
python analytics/analysis.py
```

Run the support assistant using the instructions provided in:

```text
support_assistant/
```

---

# 8. Git Workflow

The project is maintained in a single public GitHub repository.

The required Git workflow includes:

1. Create a feature branch from `main`.
2. Make at least two commits on the feature branch.
3. Merge the feature branch into `main`.
4. Verify the history with:

```bash
git log --graph --oneline --all
```

This provides visible evidence of the feature branch and merge history.
