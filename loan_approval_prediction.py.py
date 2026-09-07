# ==============================================================
# LOAN APPROVAL PREDICTION SYSTEM USING MACHINE LEARNING
# ==============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==============================================================
# 1. PROJECT TITLE
# ==============================================================

print("\n" + "=" * 70)
print("          LOAN APPROVAL PREDICTION SYSTEM")
print("             USING MACHINE LEARNING")
print("=" * 70)


# ==============================================================
# 2. DATA COLLECTION
# ==============================================================

print("\n[1] DATA COLLECTION")
print("-" * 70)


# The program looks for loan_data.csv in the same folder
# as this Python program.

dataset_file = "loan_data.csv"


# If dataset is available, load it.
# Otherwise, create a sample dataset so the program can run.

if os.path.exists(dataset_file):

    df = pd.read_csv(dataset_file)

    print("Bank loan dataset loaded successfully.")
    print("Dataset file:", dataset_file)

else:

    print("loan_data.csv was not found.")
    print("Creating a sample loan dataset for demonstration...")

    np.random.seed(42)

    number_of_records = 500

    data = {

        "Loan_ID":
        ["LP" + str(1000 + i)
         for i in range(number_of_records)],

        "Gender":
        np.random.choice(
            ["Male", "Female"],
            number_of_records
        ),

        "Married":
        np.random.choice(
            ["Yes", "No"],
            number_of_records
        ),

        "Dependents":
        np.random.choice(
            ["0", "1", "2", "3+"],
            number_of_records
        ),

        "Education":
        np.random.choice(
            ["Graduate", "Not Graduate"],
            number_of_records
        ),

        "Self_Employed":
        np.random.choice(
            ["Yes", "No"],
            number_of_records
        ),

        "ApplicantIncome":
        np.random.randint(
            1500, 15000,
            number_of_records
        ),

        "CoapplicantIncome":
        np.random.randint(
            0, 8000,
            number_of_records
        ),

        "LoanAmount":
        np.random.randint(
            50, 500,
            number_of_records
        ),

        "Loan_Amount_Term":
        np.random.choice(
            [120, 180, 240, 300, 360],
            number_of_records
        ),

        "Credit_History":
        np.random.choice(
            [0, 1],
            number_of_records,
            p=[0.2, 0.8]
        ),

        "Property_Area":
        np.random.choice(
            ["Urban", "Rural", "Semiurban"],
            number_of_records
        )
    }

    df = pd.DataFrame(data)


    # Generate loan status using simple realistic rules

    df["Loan_Status"] = np.where(
        (
            (df["Credit_History"] == 1) &
            (df["ApplicantIncome"] >= 3000) &
            (df["LoanAmount"] <= 400)
        ),
        "Y",
        "N"
    )


    # Save sample dataset

    df.to_csv(
        dataset_file,
        index=False
    )

    print("Sample dataset created successfully.")


print("\nNumber of records:", len(df))

print("\nFirst 5 records:")
print(df.head())


# ==============================================================
# 3. DATASET INFORMATION
# ==============================================================

print("\n[2] DATASET INFORMATION")
print("-" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)


# ==============================================================
# 4. CLEAN COLUMN NAMES
# ==============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)


# ==============================================================
# 5. CHECK REQUIRED TARGET COLUMN
# ==============================================================

if "Loan_Status" not in df.columns:

    print("\nERROR: Loan_Status column was not found.")

    print(
        "\nYour CSV must contain a column named Loan_Status."
    )

    input("\nPress Enter to exit...")

    raise SystemExit


# ==============================================================
# 6. DATA PREPROCESSING
# ==============================================================

print("\n[3] DATA PREPROCESSING")
print("-" * 70)

print("\nMissing values before preprocessing:")

print(df.isnull().sum())


# --------------------------------------------------------------
# Remove Loan_ID
# --------------------------------------------------------------

if "Loan_ID" in df.columns:

    df = df.drop(
        "Loan_ID",
        axis=1
    )


# --------------------------------------------------------------
# Convert Dependents
# --------------------------------------------------------------

if "Dependents" in df.columns:

    df["Dependents"] = (
        df["Dependents"]
        .astype(str)
        .str.strip()
    )

    df["Dependents"] = (
        df["Dependents"]
        .replace(
            {
                "3+": "3",
                "nan": np.nan,
                "None": np.nan
            }
        )
    )

    df["Dependents"] = pd.to_numeric(
        df["Dependents"],
        errors="coerce"
    )


# --------------------------------------------------------------
# Numerical columns
# --------------------------------------------------------------

numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents"
]


for column in numerical_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if df[column].isnull().any():

            median_value = df[column].median()

            if pd.isna(median_value):

                median_value = 0

            df[column] = df[column].fillna(
                median_value
            )


# --------------------------------------------------------------
# Categorical columns
# --------------------------------------------------------------

categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]


for column in categorical_columns:

    if column in df.columns:

        df[column] = df[column].astype(str)

        df[column] = df[column].replace(
            ["nan", "None", ""],
            np.nan
        )

        if df[column].isnull().any():

            mode_values = df[column].mode()

            if len(mode_values) > 0:

                df[column] = df[column].fillna(
                    mode_values[0]
                )

            else:

                df[column] = df[column].fillna(
                    "Unknown"
                )


# ==============================================================
# 7. PROCESS LOAN STATUS
# ==============================================================

df["Loan_Status"] = (
    df["Loan_Status"]
    .astype(str)
    .str.strip()
)


# Convert common loan status formats to 0 and 1

loan_status_mapping = {
    "Y": 1,
    "N": 0,
    "Yes": 1,
    "No": 0,
    "Approved": 1,
    "Rejected": 0,
    "1": 1,
    "0": 0
}


df["Loan_Status"] = (
    df["Loan_Status"]
    .map(loan_status_mapping)
)


# Remove rows where target could not be converted

df = df.dropna(
    subset=["Loan_Status"]
)

df["Loan_Status"] = df["Loan_Status"].astype(int)


print("\nMissing values after preprocessing:")

print(df.isnull().sum())


# ==============================================================
# 8. ENCODE CATEGORICAL VARIABLES
# ==============================================================

print("\n[4] FEATURE ENGINEERING")
print("-" * 70)

label_encoders = {}


for column in categorical_columns:

    if column in df.columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

        label_encoders[column] = encoder


print("\nCategorical variables encoded successfully.")


# ==============================================================
# 9. EXPLORATORY DATA ANALYSIS
# ==============================================================

print("\n[5] EXPLORATORY DATA ANALYSIS")
print("-" * 70)


# --------------------------------------------------------------
# Graph 1: Loan Status Distribution
# --------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Loan_Status",
    data=df
)

plt.title(
    "Loan Approval Distribution"
)

plt.xlabel(
    "Loan Status (0 = Rejected, 1 = Approved)"
)

plt.ylabel(
    "Number of Applicants"
)

plt.tight_layout()

plt.savefig(
    "loan_status_distribution.png"
)

plt.show()


# --------------------------------------------------------------
# Graph 2: Credit History vs Loan Approval
# --------------------------------------------------------------

if "Credit_History" in df.columns:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        x="Credit_History",
        hue="Loan_Status",
        data=df
    )

    plt.title(
        "Credit History vs Loan Approval"
    )

    plt.xlabel(
        "Credit History"
    )

    plt.ylabel(
        "Number of Applicants"
    )

    plt.tight_layout()

    plt.savefig(
        "credit_history_vs_approval.png"
    )

    plt.show()


# --------------------------------------------------------------
# Graph 3: Income vs Loan Approval
# --------------------------------------------------------------

if "ApplicantIncome" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x="Loan_Status",
        y="ApplicantIncome",
        data=df
    )

    plt.title(
        "Applicant Income vs Loan Approval"
    )

    plt.xlabel(
        "Loan Status"
    )

    plt.ylabel(
        "Applicant Income"
    )

    plt.tight_layout()

    plt.savefig(
        "income_vs_approval.png"
    )

    plt.show()


# --------------------------------------------------------------
# Graph 4: Education vs Loan Approval
# --------------------------------------------------------------

if "Education" in df.columns:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        x="Education",
        hue="Loan_Status",
        data=df
    )

    plt.title(
        "Education vs Loan Approval"
    )

    plt.xlabel(
        "Education"
    )

    plt.ylabel(
        "Number of Applicants"
    )

    plt.tight_layout()

    plt.savefig(
        "education_vs_approval.png"
    )

    plt.show()


# ==============================================================
# 10. FEATURE SELECTION
# ==============================================================

print("\n[6] FEATURE SELECTION")
print("-" * 70)


possible_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]


features = [
    column
    for column in possible_features
    if column in df.columns
]


X = df[features]

y = df["Loan_Status"]


print("\nSelected features:")

for feature in features:

    print("-", feature)


# ==============================================================
# 11. TRAIN-TEST SPLIT
# ==============================================================

print("\n[7] TRAIN-TEST SPLIT")
print("-" * 70)


# Make sure there are enough samples

if len(df) < 10:

    print(
        "Dataset contains too few records."
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


# Use stratify only if both classes have enough records

class_counts = y.value_counts()


if (
    len(class_counts) == 2
    and class_counts.min() >= 2
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


print(
    "\nTraining data:",
    len(X_train)
)

print(
    "Testing data:",
    len(X_test)
)


# ==============================================================
# 12. FEATURE SCALING
# ==============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==============================================================
# 13. MODEL TRAINING
# ==============================================================

print("\n[8] MODEL TRAINING")
print("-" * 70)


# --------------------------------------------------------------
# Logistic Regression
# --------------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=2000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_scaled
)


# --------------------------------------------------------------
# Decision Tree
# --------------------------------------------------------------

decision_tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

decision_tree_prediction = (
    decision_tree_model.predict(
        X_test
    )
)


# --------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_prediction = (
    random_forest_model.predict(
        X_test
    )
)


# --------------------------------------------------------------
# Naive Bayes
# --------------------------------------------------------------

naive_bayes_model = GaussianNB()

naive_bayes_model.fit(
    X_train_scaled,
    y_train
)

naive_bayes_prediction = (
    naive_bayes_model.predict(
        X_test_scaled
    )
)


# --------------------------------------------------------------
# Support Vector Machine
# --------------------------------------------------------------

svm_model = SVC(
    kernel="linear"
)

svm_model.fit(
    X_train_scaled,
    y_train
)

svm_prediction = (
    svm_model.predict(
        X_test_scaled
    )
)


print(
    "\nAll machine learning models trained successfully."
)


# ==============================================================
# 14. MODEL EVALUATION
# ==============================================================

print("\n[9] MODEL EVALUATION")
print("-" * 70)


def evaluate_model(
    model_name,
    actual_values,
    predicted_values
):

    accuracy = accuracy_score(
        actual_values,
        predicted_values
    )

    precision = precision_score(
        actual_values,
        predicted_values,
        zero_division=0
    )

    recall = recall_score(
        actual_values,
        predicted_values,
        zero_division=0
    )

    f1 = f1_score(
        actual_values,
        predicted_values,
        zero_division=0
    )

    print(
        "\n" + model_name
    )

    print(
        "Accuracy  :",
        round(accuracy * 100, 2),
        "%"
    )

    print(
        "Precision :",
        round(precision, 2)
    )

    print(
        "Recall    :",
        round(recall, 2)
    )

    print(
        "F1 Score  :",
        round(f1, 2)
    )

    return accuracy


logistic_accuracy = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_prediction
)


decision_tree_accuracy = evaluate_model(
    "Decision Tree",
    y_test,
    decision_tree_prediction
)


random_forest_accuracy = evaluate_model(
    "Random Forest",
    y_test,
    random_forest_prediction
)


naive_bayes_accuracy = evaluate_model(
    "Naive Bayes",
    y_test,
    naive_bayes_prediction
)


svm_accuracy = evaluate_model(
    "Support Vector Machine",
    y_test,
    svm_prediction
)


# ==============================================================
# 15. MODEL COMPARISON
# ==============================================================

print("\n[10] MODEL COMPARISON")
print("-" * 70)


model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
    "SVM"
]


model_accuracies = [
    logistic_accuracy,
    decision_tree_accuracy,
    random_forest_accuracy,
    naive_bayes_accuracy,
    svm_accuracy
]


comparison = pd.DataFrame(
    {
        "Model": model_names,
        "Accuracy (%)":
        [
            round(
                value * 100,
                2
            )
            for value in model_accuracies
        ]
    }
)


print(
    "\n",
    comparison.to_string(
        index=False
    )
)


# Model comparison graph

plt.figure(
    figsize=(10, 5)
)

sns.barplot(
    x="Model",
    y="Accuracy (%)",
    data=comparison
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "model_comparison.png"
)

plt.show()


# ==============================================================
# 16. SELECT BEST MODEL
# ==============================================================

best_index = np.argmax(
    model_accuracies
)

best_model_name = model_names[
    best_index
]


print(
    "\nBest Model:",
    best_model_name
)


# ==============================================================
# 17. CONFUSION MATRIX
# ==============================================================

if best_model_name == "Logistic Regression":

    best_prediction = logistic_prediction

elif best_model_name == "Decision Tree":

    best_prediction = decision_tree_prediction

elif best_model_name == "Random Forest":

    best_prediction = random_forest_prediction

elif best_model_name == "Naive Bayes":

    best_prediction = naive_bayes_prediction

else:

    best_prediction = svm_prediction


cm = confusion_matrix(
    y_test,
    best_prediction
)


print(
    "\nConfusion Matrix:"
)

print(cm)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - " +
    best_model_name
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()


# ==============================================================
# 18. NEW APPLICANT PREDICTION
# ==============================================================

print("\n" + "=" * 70)
print("             NEW APPLICANT PREDICTION")
print("=" * 70)


print(
    "\nEnter the applicant details."
)

try:

    # Gender

    gender_input = input(
        "\nGender (Male/Female): "
    ).strip()


    # Married

    married_input = input(
        "Married (Yes/No): "
    ).strip()


    # Dependents

    dependents_input = input(
        "Dependents (0/1/2/3+): "
    ).strip()


    # Education

    education_input = input(
        "Education (Graduate/Not Graduate): "
    ).strip()


    # Self employed

    self_employed_input = input(
        "Self Employed (Yes/No): "
    ).strip()


    # Income

    applicant_income = float(
        input(
            "Applicant Income: "
        )
    )


    # Co-applicant income

    coapplicant_income = float(
        input(
            "Coapplicant Income: "
        )
    )


    # Loan amount

    loan_amount = float(
        input(
            "Loan Amount: "
        )
    )


    # Loan term

    loan_term = float(
        input(
            "Loan Amount Term: "
        )
    )


    # Credit history

    credit_history = float(
        input(
            "Credit History (1 = Good, 0 = Bad): "
        )
    )


    # Property area

    property_area_input = input(
        "Property Area "
        "(Urban/Semiurban/Rural): "
    ).strip()


    # ----------------------------------------------------------
    # Encode new applicant
    # ----------------------------------------------------------

    def encode_input(
        column,
        value
    ):

        if column not in label_encoders:

            return 0

        encoder = label_encoders[
            column
        ]

        value = str(value).strip()


        # Direct match

        if value in encoder.classes_:

            return encoder.transform(
                [value]
            )[0]


        # Case-insensitive match

        for category in encoder.classes_:

            if (
                str(category).lower()
                == value.lower()
            ):

                return encoder.transform(
                    [category]
                )[0]


        return 0


    # Create applicant dictionary

    applicant_data = {}


    if "Gender" in features:

        applicant_data["Gender"] = (
            encode_input(
                "Gender",
                gender_input
            )
        )


    if "Married" in features:

        applicant_data["Married"] = (
            encode_input(
                "Married",
                married_input
            )
        )


    if "Dependents" in features:

        if dependents_input == "3+":

            applicant_data["Dependents"] = 3

        else:

            applicant_data["Dependents"] = int(
                dependents_input
            )


    if "Education" in features:

        applicant_data["Education"] = (
            encode_input(
                "Education",
                education_input
            )
        )


    if "Self_Employed" in features:

        applicant_data["Self_Employed"] = (
            encode_input(
                "Self_Employed",
                self_employed_input
            )
        )


    if "ApplicantIncome" in features:

        applicant_data["ApplicantIncome"] = (
            applicant_income
        )


    if "CoapplicantIncome" in features:

        applicant_data["CoapplicantIncome"] = (
            coapplicant_income
        )


    if "LoanAmount" in features:

        applicant_data["LoanAmount"] = (
            loan_amount
        )


    if "Loan_Amount_Term" in features:

        applicant_data["Loan_Amount_Term"] = (
            loan_term
        )


    if "Credit_History" in features:

        applicant_data["Credit_History"] = (
            credit_history
        )


    if "Property_Area" in features:

        applicant_data["Property_Area"] = (
            encode_input(
                "Property_Area",
                property_area_input
            )
        )


    # Arrange columns exactly like training data

    new_applicant = pd.DataFrame(
        [applicant_data]
    )

    new_applicant = new_applicant[
        features
    ]


    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    if best_model_name in [
        "Logistic Regression",
        "Naive Bayes",
        "SVM"
    ]:

        new_applicant_processed = (
            scaler.transform(
                new_applicant
            )
        )

        prediction = best_prediction_model = (
            {
                "Logistic Regression":
                logistic_model,

                "Naive Bayes":
                naive_bayes_model,

                "SVM":
                svm_model
            }[
                best_model_name
            ].predict(
                new_applicant_processed
            )
        )

    elif best_model_name == "Decision Tree":

        prediction = decision_tree_model.predict(
            new_applicant
        )

    else:

        prediction = random_forest_model.predict(
            new_applicant
        )


    # ----------------------------------------------------------
    # Display result
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("                    PREDICTION RESULT")
    print("=" * 70)


    if prediction[0] == 1:

        print(
            "\n          LOAN APPROVED"
        )

        print(
            "\nThe machine learning model predicts"
            " that the loan application can be approved."
        )

    else:

        print(
            "\n          LOAN REJECTED"
        )

        print(
            "\nThe machine learning model predicts"
            " that the loan application should be rejected."
        )


except Exception as error:

    print(
        "\nUnable to process applicant details."
    )

    print(
        "Error:",
        error
    )


# ==============================================================
# 19. CONCLUSION
# ==============================================================

print("\n" + "=" * 70)
print("                       CONCLUSION")
print("=" * 70)

print("""
The Loan Approval Prediction System successfully applies
machine learning techniques to predict whether a loan
application should be approved or rejected.

The system performs data collection, data preprocessing,
feature engineering, exploratory data analysis, feature
selection, model training and model evaluation.

Logistic Regression, Decision Tree, Random Forest,
Naive Bayes and Support Vector Machine algorithms are
used for classification.

The models are evaluated using Accuracy, Precision,
Recall and F1-Score. A confusion matrix is also generated
to understand the prediction results.

The system can help financial institutions speed up loan
processing and support consistent risk assessment.

The prediction depends on the quality of historical data
and should be used as a decision-support system rather
than the sole basis for financial decisions.
""")


print("\n" + "=" * 70)
print("             PROJECT EXECUTION COMPLETED")
print("=" * 70)

input(
    "\nPress Enter to close the program..."
)
