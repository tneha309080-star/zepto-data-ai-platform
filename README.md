# Zepto Data & AI Platform

An end-to-end AI/ML project covering data engineering, analytics and machine learning, and a retrieval-augmented customer support assistant. The project is organized as **one repository containing three independent but related modules**.

## Project Overview

This project implements an end-to-end **Zepto Data & AI Platform** with three modules:

1. **Data Pipeline** — scrapes and cleans catalog data, converts GBP prices to INR using the required fixed exchange rate, stores normalized data in SQLite, and validates SQL results against pandas.
2. **Analytics & ML** — profiles and cleans the Titanic dataset, performs EDA and statistical analysis, trains and evaluates classification models, handles class imbalance, tunes a Random Forest, performs fare regression, and saves a reusable ML pipeline.
3. **Support Assistant** — implements a policy-focused retrieval-augmented assistant using local sentence-transformer embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

All three modules live in one public GitHub repository and can be run independently.

---

## Repository Structure

```text
zepto-data-ai-platform/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_pipeline/
│   ├── scrape_books.py
│   ├── database.py
│   ├── books_raw.csv
│   ├── books_clean.csv
│   ├── books.db
│   └── sql_outputs.txt
│
├── analytics/
│   ├── analysis.py
│   ├── titanic.csv
│   ├── best_titanic_pipeline.joblib
│   ├── classification_results.csv
│   ├── class_imbalance_results.csv
│   ├── random_forest_grid_search_results.csv
│   ├── regression_results.csv
│   ├── final_classifier_comparison.csv
│   ├── final_regression_comparison.csv
│   └── saved charts and plots
│
└── support_assistant/
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    ├── build_index.py
    ├── graph.py
    ├── main.py
    ├── Dockerfile
    └── chroma_db/
```

---

# Setup

## Requirements

The project uses **one consolidated `requirements.txt`** at the repository root.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The main dependencies include:

* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
* imbalanced-learn
* joblib
* requests
* beautifulsoup4
* sentence-transformers
* chromadb
* langgraph
* fastapi
* uvicorn
* pydantic
* langchain-groq

---

# Quick Start

From the project root:

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Module 1
python data_pipeline\scrape_books.py
python data_pipeline\database.py

# Module 2
python analytics\analysis.py

# Module 3
cd support_assistant
python build_index.py
python -m uvicorn main:app --reload --port 7860
```

The three modules are independent. Module 1 and Module 2 generate their own outputs, while Module 3 requires the ChromaDB index to be built before starting the API.

---

# Module 1 — Data Pipeline

**Location:** `/data_pipeline`

## Objective

This module demonstrates a complete data-engineering workflow:

```text
Web scraping
    ↓
Raw data
    ↓
Cleaning and type conversion
    ↓
GBP → INR conversion
    ↓
Normalized SQLite database
    ↓
SQL analysis
    ↓
pandas validation
```

## Data Source

The data source is [Books to Scrape](http://books.toscrape.com/), a public website designed for scraping practice.

The pipeline uses:

* `requests`
* `BeautifulSoup`
* `pandas`
* `sqlite3`

The first five catalogue pages are scraped, producing **100 books**, which exceeds the required minimum of 60.

## Run

From the project root:

```powershell
python data_pipeline\scrape_books.py
```

Then:

```powershell
python data_pipeline\database.py
```

## Scraped Fields

The raw dataset contains:

* `title`
* `price_gbp`
* `star_rating`
* `availability`
* `category`

The cleaned dataset contains:

* `title`
* `price_gbp`
* `rating`
* `availability`
* `in_stock`
* `category`
* `price_inr`

## Cleaning Decisions

### Price

Currency symbols are removed and the value is converted to a numeric `float`.

Unexpected numeric parsing failures are handled using numeric conversion with coercion and median imputation.

### Rating

The textual values:

```text
One
Two
Three
Four
Five
```

are mapped to:

```text
1
2
3
4
5
```

### Availability

Availability text is converted into a boolean `in_stock` field.

### Category

Category information is extracted from the product detail page. If a category extraction produces an invalid UI value such as `Add a comment`, it is normalized to `Unknown` rather than storing the UI text as a category.

This keeps the dataset usable while explicitly representing an extraction failure.

## Currency Conversion

The required fixed project rate is:

```text
```
