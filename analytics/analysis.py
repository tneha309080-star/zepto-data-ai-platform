import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# STEP 1: LOAD TITANIC DATASET
# ==========================================

df = sns.load_dataset("titanic")

# Save raw dataset for offline use
df.to_csv("analytics/titanic.csv", index=False)

# ==========================================
# STEP 2: RAW DATA INSPECTION
# ==========================================

print("\n========== DATA INFO ==========")
df.info()

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()
missing_percent = (df.isnull().mean() * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})

print(missing_report[missing_report["missing_count"] > 0])

# ==========================================
# STEP 3: DATA CLEANING
# ==========================================

# 1. Drop columns with more than 30% missing values
# deck has 77.22% missing values, so it is removed.
df_clean = df.drop(columns=["deck"]).copy()

# 2. Drop rows where columns have less than 5% missing values
# embarked and embark_town each have only 0.22% missing values.
df_clean = df_clean.dropna(
    subset=["embarked", "embark_town"]
).copy()

# 3. Impute age because it has 19.87% missing values.
# Median is used because it is less sensitive to extreme ages.
age_median = df_clean["age"].median()

df_clean["age"] = df_clean["age"].fillna(age_median)

# ==========================================
# VERIFY CLEANING
# ==========================================

print("\n========== CLEANING RESULTS ==========")

print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

print("\nAge median used for imputation:", age_median)

# ==========================================
# STEP 4: AGE AND FARE ANALYSIS
# ==========================================

import matplotlib.pyplot as plt

# ------------------------------------------
# AGE HISTOGRAM
# ------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df_clean["age"], bins=20)
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.title("Distribution of Passenger Age")
plt.tight_layout()
plt.savefig("analytics/age_histogram.png")
plt.close()

# ------------------------------------------
# AGE BOXPLOT
# ------------------------------------------

plt.figure(figsize=(8, 4))
plt.boxplot(df_clean["age"], vert=False)
plt.xlabel("Age")
plt.title("Age Box Plot")
plt.tight_layout()
plt.savefig("analytics/age_boxplot.png")
plt.close()

# ------------------------------------------
# FARE HISTOGRAM
# ------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df_clean["fare"], bins=30)
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.title("Distribution of Passenger Fare")
plt.tight_layout()
plt.savefig("analytics/fare_histogram.png")
plt.close()

# ------------------------------------------
# FARE BOXPLOT
# ------------------------------------------

plt.figure(figsize=(8, 4))
plt.boxplot(df_clean["fare"], vert=False)
plt.xlabel("Fare")
plt.title("Fare Box Plot")
plt.tight_layout()
plt.savefig("analytics/fare_boxplot.png")
plt.close()

print("\n========== AGE AND FARE CHARTS ==========")
print("Age histogram saved.")
print("Age box plot saved.")
print("Fare histogram saved.")
print("Fare box plot saved.")

# ==========================================
# IQR OUTLIER ANALYSIS
# ==========================================

def calculate_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return q1, q3, iqr, lower_bound, upper_bound, len(outliers)


age_q1, age_q3, age_iqr, age_lower, age_upper, age_outliers = \
    calculate_iqr_outliers(df_clean["age"])

fare_q1, fare_q3, fare_iqr, fare_lower, fare_upper, fare_outliers = \
    calculate_iqr_outliers(df_clean["fare"])

print("\n========== IQR OUTLIER ANALYSIS ==========")

print("\nAge:")
print("Q1:", age_q1)
print("Q3:", age_q3)
print("IQR:", age_iqr)
print("Lower bound:", age_lower)
print("Upper bound:", age_upper)
print("Outlier count:", age_outliers)

print("\nFare:")
print("Q1:", fare_q1)
print("Q3:", fare_q3)
print("IQR:", fare_iqr)
print("Lower bound:", fare_lower)
print("Upper bound:", fare_upper)
print("Outlier count:", fare_outliers)

# ==========================================
# FARE STATISTICS
# ==========================================

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()
fare_skewness = df_clean["fare"].skew()

print("\n========== FARE STATISTICS ==========")

print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode.tolist())
print("Skewness:", fare_skewness)

if fare_mean > fare_median:
    print("Interpretation: Mean > median, indicating positive/right skew.")
elif fare_mean < fare_median:
    print("Interpretation: Mean < median, indicating negative/left skew.")
else:
    print("Interpretation: Mean and median are approximately equal.")

# ==========================================
# STEP 5: SURVIVAL ANALYSIS
# ==========================================

print("\n========== SURVIVAL RATE BY SEX ==========")

survival_by_sex = df_clean.groupby("sex")["survived"].mean()

print(survival_by_sex)

print("\n========== SURVIVAL RATE BY PCLASS ==========")

survival_by_class = df_clean.groupby("pclass")["survived"].mean()

print(survival_by_class)

print("\n========== SURVIVAL RATE BY SEX AND PCLASS ==========")

survival_by_sex_class = (
    df_clean
    .groupby(["sex", "pclass"])["survived"]
    .mean()
)

print(survival_by_sex_class)


# ==========================================
# BOOLEAN MASKING WITH &
# ==========================================

female_first_class = df_clean[
    (df_clean["sex"] == "female") &
    (df_clean["pclass"] == 1)
]

print("\n========== FEMALE AND FIRST CLASS ==========")
print("Number of passengers:", len(female_first_class))
print(
    "Survival rate:",
    female_first_class["survived"].mean()
)


# ==========================================
# BOOLEAN MASKING WITH |
# ==========================================

first_or_second_class = df_clean[
    (df_clean["pclass"] == 1) |
    (df_clean["pclass"] == 2)
]

print("\n========== FIRST OR SECOND CLASS ==========")
print("Number of passengers:", len(first_or_second_class))
print(
    "Survival rate:",
    first_or_second_class["survived"].mean()
)

# ==========================================
# STEP 6: CORRELATION ANALYSIS
# ==========================================

corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df_clean[corr_cols].corr()

print("\n========== CORRELATION MATRIX ==========")
print(corr_matrix)


# ------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Selected Titanic Variables")
plt.tight_layout()

plt.savefig("analytics/correlation_heatmap.png")
plt.close()

print("\nCorrelation heatmap saved to:")
print("analytics/correlation_heatmap.png")


# ------------------------------------------
# FIND TWO STRONGEST ABSOLUTE CORRELATIONS
# ------------------------------------------

corr_pairs = []

for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        variable_1 = corr_cols[i]
        variable_2 = corr_cols[j]
        correlation = corr_matrix.loc[variable_1, variable_2]

        corr_pairs.append(
            (variable_1, variable_2, correlation)
        )

corr_pairs_sorted = sorted(
    corr_pairs,
    key=lambda x: abs(x[2]),
    reverse=True
)

print("\n========== STRONGEST CORRELATIONS ==========")

for pair in corr_pairs_sorted[:2]:
    print(
        f"{pair[0]} vs {pair[1]}: "
        f"{pair[2]:.4f}"
    )

# ==========================================
# STEP 7: MULTIVARIATE VISUALIZATIONS
# CHART 1: SURVIVAL BY SEX AND PCLASS
# ==========================================

plt.figure(figsize=(8, 6))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("analytics/survival_by_sex_pclass.png")
plt.close()

print("\nChart 1 saved to:")
print("analytics/survival_by_sex_pclass.png")

# ==========================================
# CHART 2: AGE VS FARE BY SURVIVAL
# ==========================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Age vs Fare by Survival and Sex")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.savefig("analytics/age_vs_fare_survival.png")
plt.close()

print("\nChart 2 saved to:")
print("analytics/age_vs_fare_survival.png")

# ==========================================
# CHART 3: FARE BY PCLASS AND SURVIVAL
# ==========================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df_clean,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title("Fare Distribution by Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()
plt.savefig("analytics/fare_by_pclass_survival.png")
plt.close()

print("\nChart 3 saved to:")
print("analytics/fare_by_pclass_survival.png")

# ==========================================
# CHART 4: SURVIVAL BY AGE GROUP AND SEX
# ==========================================

df_clean["age_group"] = pd.cut(
    df_clean["age"],
    bins=[0, 12, 18, 35, 60, float("inf")],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
    include_lowest=True
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_clean,
    x="age_group",
    y="survived",
    hue="sex",
    errorbar=None
)

plt.title("Survival Rate by Age Group and Sex")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("analytics/survival_by_age_group_sex.png")
plt.close()

print("\nChart 4 saved to:")
print("analytics/survival_by_age_group_sex.png")

# ==========================================
# STEP 8: EDA-ONLY STANDARDIZATION
# ==========================================

eda_df = df_clean.copy()

# Before standardization
print("\n========== BEFORE STANDARDIZATION ==========")

print("Age:")
print(f"Mean: {eda_df['age'].mean():.4f}")
print(f"Std:  {eda_df['age'].std():.4f}")

print("\nFare:")
print(f"Mean: {eda_df['fare'].mean():.4f}")
print(f"Std:  {eda_df['fare'].std():.4f}")

# Z-score standardization
eda_df["age_zscore"] = (
    (eda_df["age"] - eda_df["age"].mean())
    / eda_df["age"].std()
)

eda_df["fare_zscore"] = (
    (eda_df["fare"] - eda_df["fare"].mean())
    / eda_df["fare"].std()
)

# After standardization
print("\n========== AFTER STANDARDIZATION ==========")

print("Age z-score:")
print(f"Mean: {eda_df['age_zscore'].mean():.4f}")
print(f"Std:  {eda_df['age_zscore'].std():.4f}")

print("\nFare z-score:")
print(f"Mean: {eda_df['fare_zscore'].mean():.4f}")
print(f"Std:  {eda_df['fare_zscore'].std():.4f}")

# ==========================================
# STEP 9: STRATIFIED TRAIN/TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

# Features and target
X = df_clean.drop(columns=["survived"])
y = df_clean["survived"]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN/TEST SPLIT ==========")

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))

# ==========================================
# STEP 10: ML PREPROCESSING PIPELINE
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Remove columns that should not be used for prediction
# survived = target
# alive = direct representation of target (data leakage)
# age_group = EDA-only feature
X_model = df_clean.drop(
    columns=["survived", "alive", "age_group"],
    errors="ignore"
).copy()

y_model = df_clean["survived"].copy()

# Train/test split for modeling
X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_model,
    test_size=0.20,
    random_state=42,
    stratify=y_model
)

# Numerical and categorical columns
numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked",
    "class",
    "who",
    "adult_male",
    "embark_town",
    "alone"
]

# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Fit preprocessing ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the already-fitted preprocessing
X_test_processed = preprocessor.transform(X_test)

print("\n========== ML PREPROCESSING ==========")

print("Original training shape:", X_train.shape)
print("Original testing shape:", X_test.shape)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nPreprocessing fitted on training data only.")
print("Test data was transformed using the training-fitted preprocessing.")

# ==========================================
# STEP 11: THREE CLASSIFICATION MODELS
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------
# Logistic Regression
# ------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]
)

# ------------------------------------------
# Decision Tree
# ------------------------------------------

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ))
    ]
)

# ------------------------------------------
# Random Forest
# ------------------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# ------------------------------------------
# Train all three models
# ------------------------------------------

logistic_pipeline.fit(X_train, y_train)

decision_tree_pipeline.fit(X_train, y_train)

random_forest_pipeline.fit(X_train, y_train)

print("\n========== CLASSIFICATION MODELS TRAINED ==========")

print("Logistic Regression: trained")
print("Decision Tree: trained")
print("Random Forest: trained")

# ==========================================
# DECISION TREE VISUALIZATION
# ==========================================

# Get the fitted preprocessing step
tree_preprocessor = decision_tree_pipeline.named_steps["preprocessor"]

# Get transformed feature names
feature_names = tree_preprocessor.get_feature_names_out()

# Get the trained decision tree
tree_model = decision_tree_pipeline.named_steps["classifier"]

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Did Not Survive", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Titanic Survival Prediction")
plt.tight_layout()

plt.savefig("analytics/decision_tree.png", dpi=150)
plt.close()

print("\nDecision tree visualization saved to:")
print("analytics/decision_tree.png")

# ==========================================
# STEP 12: CLASSIFICATION MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree_pipeline,
    "Random Forest": random_forest_pipeline
}

results = []

plt.figure(figsize=(8, 6))

for model_name, model in models.items():

    # Predictions
    y_pred = model.predict(X_test)

    # Probability for positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n========== {model_name.upper()} ==========")

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    # Save individual confusion matrix
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"]
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(f"analytics/{filename}", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(1)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

# Final ROC plot

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Titanic Classifiers")
plt.legend()

plt.tight_layout()
plt.savefig("analytics/roc_curves.png", dpi=150)
plt.close()

# Comparison table
results_df = pd.DataFrame(results)

print("\n========== CLASSIFICATION COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "analytics/classification_results.csv",
    index=False
)

print("\nClassification results saved to:")
print("analytics/classification_results.csv")
print("analytics/roc_curves.png")

# ==========================================
# STEP 13: CLASS IMBALANCE EXPERIMENT
# ==========================================

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ------------------------------------------
# 1. Baseline Logistic Regression
# ------------------------------------------

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)

baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)


# ------------------------------------------
# 2. Logistic Regression with class_weight
# ------------------------------------------

balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

balanced_model.fit(X_train, y_train)

balanced_pred = balanced_model.predict(X_test)

balanced_precision = precision_score(y_test, balanced_pred)
balanced_recall = recall_score(y_test, balanced_pred)
balanced_f1 = f1_score(y_test, balanced_pred)


# ------------------------------------------
# 3. Logistic Regression with SMOTE
# ------------------------------------------

smote_model = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

smote_model.fit(X_train, y_train)

smote_pred = smote_model.predict(X_test)

smote_precision = precision_score(y_test, smote_pred)
smote_recall = recall_score(y_test, smote_pred)
smote_f1 = f1_score(y_test, smote_pred)


# ------------------------------------------
# Comparison table
# ------------------------------------------

imbalance_results = pd.DataFrame([
    {
        "Method": "Baseline",
        "Precision": baseline_precision,
        "Recall": baseline_recall,
        "F1": baseline_f1
    },
    {
        "Method": "Class Weight Balanced",
        "Precision": balanced_precision,
        "Recall": balanced_recall,
        "F1": balanced_f1
    },
    {
        "Method": "SMOTE",
        "Precision": smote_precision,
        "Recall": smote_recall,
        "F1": smote_f1
    }
])

print("\n========== CLASS IMBALANCE EXPERIMENT ==========")

print(
    imbalance_results.round(4).to_string(index=False)
)

imbalance_results.to_csv(
    "analytics/class_imbalance_results.csv",
    index=False
)

print("\nClass imbalance results saved to:")
print("analytics/class_imbalance_results.csv")

# ============================================================
# STEP 14: RANDOM FOREST GRID SEARCH + OOB SCORE
# ============================================================

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

print("\n========== RANDOM FOREST GRID SEARCH ==========")

# Random Forest pipeline
rf_grid_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            oob_score=True,
            bootstrap=True,
            n_jobs=-1
        )
    )
])

# Hyperparameter grid
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"]
}

# Grid search
grid_search = GridSearchCV(
    estimator=rf_grid_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=False
)

# Fit ONLY on training data
grid_search.fit(X_train, y_train)

# Best results
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1:")
print(f"{grid_search.best_score_:.4f}")

# Get the fitted best Random Forest
best_rf_pipeline = grid_search.best_estimator_
best_rf_model = best_rf_pipeline.named_steps["classifier"]

print("\nOOB Score:")
print(f"{best_rf_model.oob_score_:.4f}")

# Evaluate best model on untouched test data
best_rf_pred = best_rf_pipeline.predict(X_test)
best_rf_prob = best_rf_pipeline.predict_proba(X_test)[:, 1]

best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
best_rf_precision = precision_score(y_test, best_rf_pred)
best_rf_recall = recall_score(y_test, best_rf_pred)
best_rf_f1 = f1_score(y_test, best_rf_pred)
best_rf_auc = roc_auc_score(y_test, best_rf_prob)

print("\nBest Random Forest Test Results:")
print(f"Accuracy:  {best_rf_accuracy:.4f}")
print(f"Precision: {best_rf_precision:.4f}")
print(f"Recall:    {best_rf_recall:.4f}")
print(f"F1:        {best_rf_f1:.4f}")
print(f"ROC-AUC:   {best_rf_auc:.4f}")

# Save GridSearch results
grid_results = pd.DataFrame(grid_search.cv_results_)

grid_results.to_csv(
    "analytics/random_forest_grid_search_results.csv",
    index=False
)

print("\nGridSearch results saved to:")
print("analytics/random_forest_grid_search_results.csv")

# ============================================================
# STEP 15: MULTIVARIATE LINEAR REGRESSION FOR FARE
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n========== MULTIVARIATE LINEAR REGRESSION ==========")

# Remove EDA-only columns and fare target
regression_df = df_clean.drop(
    columns=["fare", "age_group"],
    errors="ignore"
).copy()

X_reg = regression_df
y_reg = df_clean["fare"].copy()

# Train/test split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Identify numeric and categorical predictors
reg_numeric_features = [
    col for col in X_reg_train.select_dtypes(include=["int64", "float64"]).columns
]

reg_categorical_features = [
    col for col in X_reg_train.select_dtypes(
        include=["object", "category", "str", "bool"]
    ).columns
]
print("\nRegression numeric features:")
print(reg_numeric_features)

print("\nRegression categorical features:")
print(reg_categorical_features)

# Regression preprocessing
reg_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

reg_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

reg_preprocessor = ColumnTransformer([
    ("numeric", reg_numeric_pipeline, reg_numeric_features),
    ("categorical", reg_categorical_pipeline, reg_categorical_features)
])

# Complete regression pipeline
regression_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
])

# Fit on training data only
regression_pipeline.fit(X_reg_train, y_reg_train)

# Predict test data
y_reg_pred = regression_pipeline.predict(X_reg_test)

# Metrics
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

# Number of predictors after one-hot encoding
X_reg_test_processed = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_reg_test)

n = len(y_reg_test)
p = X_reg_test_processed.shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\nRegression Results:")
print(f"MAE:         {mae:.4f}")
print(f"RMSE:        {rmse:.4f}")
print(f"R-squared:   {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")

print(f"\nTest observations (n): {n}")
print(f"Predictors after encoding (p): {p}")

# Residuals
residuals = y_reg_test - y_reg_pred

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_reg_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Linear Regression Residual Plot")
plt.tight_layout()
plt.savefig(
    "analytics/fare_regression_residuals.png",
    dpi=150
)
plt.close()

print("\nResidual plot saved to:")
print("analytics/fare_regression_residuals.png")

# Save regression metrics
regression_results = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R-squared",
        "Adjusted R-squared"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        adjusted_r2
    ]
})

regression_results.to_csv(
    "analytics/regression_results.csv",
    index=False
)

print("\nRegression results saved to:")
print("analytics/regression_results.csv")

# ============================================================
# STEP 16: FINAL MODEL COMPARISON
# ============================================================

print("\n========== FINAL MODEL COMPARISON ==========")

# Classifier comparison
classifier_final_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "Accuracy": [
        0.8146,
        0.7921,
        0.7865,
        best_rf_accuracy
    ],
    "Precision": [
        0.7966,
        0.8163,
        0.7344,
        best_rf_precision
    ],
    "Recall": [
        0.6912,
        0.5882,
        0.6912,
        best_rf_recall
    ],
    "F1": [
        0.7402,
        0.6838,
        0.7121,
        best_rf_f1
    ],
    "ROC-AUC": [
        0.8680,
        0.8248,
        0.8146,
        best_rf_auc
    ]
})

print("\nClassifier Comparison:")
print(classifier_final_results.to_string(index=False))

classifier_final_results.to_csv(
    "analytics/final_classifier_comparison.csv",
    index=False
)

# Regression results kept separate because classification
# and regression metrics are not directly comparable.
final_regression_results = pd.DataFrame({
    "Model": ["Multivariate Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R-squared": [r2],
    "Adjusted R-squared": [adjusted_r2]
})

print("\nRegression Results:")
print(final_regression_results.to_string(index=False))

final_regression_results.to_csv(
    "analytics/final_regression_comparison.csv",
    index=False
)

print("\nFinal comparison files saved:")
print("analytics/final_classifier_comparison.csv")
print("analytics/final_regression_comparison.csv")


# ============================================================
# FINAL CLASSIFIER INTERPRETATION
# ============================================================

print("\n========== FINAL CLASSIFIER INTERPRETATION ==========")

print(
    "Logistic Regression achieved the highest test ROC-AUC "
    f"({0.8680:.4f}) and accuracy ({0.8146:.4f}) among the "
    "three original classifiers."
)

print(
    "The tuned Random Forest achieved the highest test precision "
    f"({best_rf_precision:.4f}), but its recall ({best_rf_recall:.4f}) "
    f"and F1 ({best_rf_f1:.4f}) were lower than Logistic Regression."
)

print(
    "The class-weight-balanced Logistic Regression produced the "
    "highest recall (0.7647) and F1 (0.7761) in the "
    "class-imbalance experiment."
)

print(
    "For the final deployed classifier, the model should be selected "
    "according to the project's primary evaluation metric and the "
    "relative importance of precision versus recall."
)

# ============================================================
# STEP 17: SAVE AND RELOAD COMPLETE FITTED PIPELINE
# ============================================================

import joblib

print("\n========== SAVE AND RELOAD BEST PIPELINE ==========")

# Fit the complete Logistic Regression pipeline
# on the training data.
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

final_pipeline.fit(X_train, y_train)

# Save the complete pipeline
pipeline_path = "analytics/best_titanic_pipeline.joblib"

joblib.dump(
    final_pipeline,
    pipeline_path
)

print(f"\nPipeline saved to:")
print(pipeline_path)

# Reload the complete pipeline
loaded_pipeline = joblib.load(pipeline_path)

print("\nPipeline successfully reloaded.")

# Use completely raw, unprocessed test input.
raw_sample = X_test.iloc[[0]]

print("\nRaw sample input:")
print(raw_sample)

# Prediction using the reloaded pipeline.
# No manual encoding or scaling is performed here.
prediction = loaded_pipeline.predict(raw_sample)
prediction_probability = loaded_pipeline.predict_proba(raw_sample)[:, 1]

print("\nPrediction from reloaded pipeline:")
print(f"Predicted class: {prediction[0]}")
print(f"Survival probability: {prediction_probability[0]:.4f}")

print(
    "\nThe reloaded pipeline accepted raw feature data directly "
    "because preprocessing and the classifier are stored together."
)