# Zepto Data AI Platform

A three-module data and AI capstone project covering web scraping and data engineering, exploratory data analysis and machine learning, and an AI support assistant.

Project Overview

This project implements an end-to-end Zepto Data & AI Platform with three independent modules:

Data Pipeline — scrapes and cleans book data, converts GBP prices to INR using the required fixed exchange rate, stores normalized data in SQLite, and validates SQL results against pandas.
Analytics & ML — performs Titanic dataset profiling, EDA, statistical analysis, classification, imbalance handling, hyperparameter tuning, regression, model evaluation, and pipeline persistence.
Support Assistant — implements a retrieval-augmented Zepto policy assistant using local sentence-transformer embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

Each module is independently runnable while sharing the project's common Python environment and dependency file.

## Quick Start

From the project root:

```powershell
# Install dependencies
pip install -r requirements.txt

# Module 1 — Data Pipeline
python data_pipeline/scrape_books.py
python data_pipeline/database.py

# Module 2 — Analytics & ML
python analytics/analysis.py

# Module 3 — Support Assistant
cd support_assistant
python build_index.py
$env:MOCK_LLM="1"
python -m uvicorn main:app --reload --port 7860
```

The analytics module saves the Titanic dataset locally as `analytics/titanic.csv` so that the committed dataset can be used as an offline fallback after the initial dataset load.

The support assistant requires the ChromaDB index to be built before starting the API.


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

The support_assistant/ module implements a policy-focused Zepto customer support assistant using local embeddings, ChromaDB, LangGraph, FastAPI, and a deterministic mock LLM mode.

Architecture

The support assistant follows this flow:

Zepto policy documents
        |
        v
Document chunking
        |
        v
SentenceTransformer
(all-MiniLM-L6-v2)
        |
        v
ChromaDB
(cosine similarity)
        |
        v
LangGraph
        |
        +-----------------------------+
        |                             |
        v                             v
classify_intent              general_question
        |                             |
        |                             v
        |                      direct_answer
        |
        v
policy_question
        |
        v
retrieve_and_answer
        |
        v
Pydantic response
        |
        v
FastAPI /ask
Components
Component	Implementation
Policy documents	support_assistant/docs/doc_01.txt to doc_08.txt
Chunking	Paragraph-based chunking
Embedding model	all-MiniLM-L6-v2
Vector database	ChromaDB
Similarity	Cosine similarity
Orchestration	LangGraph StateGraph
API	FastAPI
Validation	Pydantic
Default LLM mode	Deterministic mock mode
API port	7860
LangGraph nodes

The graph contains three nodes:

classify_intent — classifies the query as either policy_question or general_question.
retrieve_and_answer — embeds policy questions, retrieves the top 3 relevant chunks from ChromaDB, and generates a context-based response.
direct_answer — handles general questions that are outside the current Zepto policy scope.

Policy questions are routed to retrieval, while general questions are routed directly to the fallback response.

Document ingestion

The eight policy documents are stored individually under:

support_assistant/docs/
├── doc_01.txt
├── doc_02.txt
├── doc_03.txt
├── doc_04.txt
├── doc_05.txt
├── doc_06.txt
├── doc_07.txt
└── doc_08.txt

Build the ChromaDB index with:

cd support_assistant
python build_index.py

The script:

Loads the local all-MiniLM-L6-v2 embedding model.
Reads all eight policy documents.
Splits documents into paragraph-based chunks.
Generates normalized embeddings.
Stores the chunks and metadata in the persistent ChromaDB collection zepto_policies.

The generated vector database is stored in:

support_assistant/chroma_db/
Running the API

The default implementation uses deterministic mock mode so that the project can run without an external LLM API key.

From support_assistant/:

$env:MOCK_LLM="1"
python -m uvicorn main:app --reload --port 7860

The API is available at:

http://127.0.0.1:7860

FastAPI documentation is available at:

http://127.0.0.1:7860/docs
API endpoint
POST /ask

Request body:

{
  "query": "What is the delivery policy?"
}

Example PowerShell request:

Invoke-RestMethod `
  -Uri "http://127.0.0.1:7860/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the delivery policy?"}'

A policy query is classified as policy_question, sent through the retrieval node, and answered using the most relevant retrieved policy context.

Example response:

{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes...",
  "sources": [
    "doc_01.txt",
    "doc_02.txt",
    "doc_05.txt"
  ],
  "confidence": 0.9
}

A general question such as:

{
  "query": "Tell me a joke"
}

is routed to direct_answer and returns the deterministic fallback response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 0.8
}
Mock LLM mode

The project defaults to:

MOCK_LLM=1

Mock mode provides deterministic responses for reproducible evaluation and does not require an external LLM API key.

The retrieval stage remains real: policy queries are embedded using all-MiniLM-L6-v2 and matched against the ChromaDB collection using cosine similarity.

A real LLM integration can be enabled as an optional extension without changing the retrieval and LangGraph architecture.

Response schema

The API validates responses using Pydantic:

{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}

confidence is constrained to the range 0.0 to 1.0.

Docker

Build the support assistant image from the project root:

docker build -t zepto-support-assistant -f support_assistant/Dockerfile .

Run the container:

docker run --rm -p 7860:7860 zepto-support-assistant

The Docker image runs Uvicorn on port 7860.

Design decisions
Local embeddings: all-MiniLM-L6-v2 avoids requiring an external embedding API.
ChromaDB: provides persistent vector storage and cosine-similarity retrieval.
LangGraph: makes intent classification, retrieval, and direct-answer routing explicit as separate nodes.
Mock mode: makes the application deterministic and reproducible for evaluation without requiring an external LLM key.
Top-3 retrieval: provides multiple relevant sources while keeping the retrieved context compact.
Source metadata: each stored chunk records its originating document so responses can expose source files.
Negative constraint: the assistant is instructed not to invent Zepto policies outside the retrieved context.

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
