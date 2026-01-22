import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

# --------------------------------------------------
# TITLE & DESCRIPTION
# --------------------------------------------------
st.title("🏦 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** to predict loan approval."
)

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

df = load_data()

# --------------------------------------------------
# DATA PREPROCESSING (FULL & SAFE)
# --------------------------------------------------

# Drop ID
df.drop("Loan_ID", axis=1, inplace=True)

# Handle missing values
cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Education"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

# Encode categorical variables
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# Target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# One-hot encoding
df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

# Log transform (outlier handling)
df["ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
df["LoanAmount"] = np.log1p(df["LoanAmount"])

# --------------------------------------------------
# FEATURES & TARGET (USED BY UI)
# --------------------------------------------------
X = df[
    [
        "ApplicantIncome",
        "LoanAmount",
        "Credit_History",
        "Self_Employed",
        "Property_Area_Semiurban",
        "Property_Area_Urban",
    ]
]
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# TRAIN SVM MODELS
# --------------------------------------------------
@st.cache_resource
def train_models():
    linear = SVC(kernel="linear", probability=True)
    poly = SVC(kernel="poly", degree=3, probability=True)
    rbf = SVC(kernel="rbf", probability=True)

    linear.fit(X_train, y_train)
    poly.fit(X_train, y_train)
    rbf.fit(X_train, y_train)

    return linear, poly, rbf

svm_linear, svm_poly, svm_rbf = train_models()
# Predictions
y_pred_linear = svm_linear.predict(X_test)
y_pred_poly = svm_poly.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

# Accuracy scores
acc_linear = accuracy_score(y_test, y_pred_linear)
acc_poly = accuracy_score(y_test, y_pred_poly)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

# --------------------------------------------------
# SIDEBAR — USER INPUTS
# --------------------------------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input(
    "Applicant Income", min_value=0, step=500
)

loan_amount = st.sidebar.number_input(
    "Loan Amount", min_value=0, step=10
)

credit_history = st.sidebar.selectbox(
    "Credit History", ["Yes", "No"]
)

employment = st.sidebar.selectbox(
    "Employment Status", ["Employed", "Self Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

st.sidebar.subheader("⚙️ Model Selection")

kernel_choice = st.sidebar.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

predict_btn = st.sidebar.button("🔍 Check Loan Eligibility")

# --------------------------------------------------
# INPUT PREPROCESSING (SAME AS TRAINING)
# --------------------------------------------------
credit_history = 1 if credit_history == "Yes" else 0
self_employed = 1 if employment == "Self Employed" else 0

property_semiurban = 1 if property_area == "Semiurban" else 0
property_urban = 1 if property_area == "Urban" else 0

app_income = np.log1p(app_income)
loan_amount = np.log1p(loan_amount)

input_data = np.array([[
    app_income,
    loan_amount,
    credit_history,
    self_employed,
    property_semiurban,
    property_urban
]])

input_scaled = scaler.transform(input_data)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
if kernel_choice == "Linear SVM":
    model = svm_linear
elif kernel_choice == "Polynomial SVM":
    model = svm_poly
else:
    model = svm_rbf

st.subheader("📈 Model Accuracy Comparison")

st.write(f"🔹 **Linear SVM Accuracy:** {acc_linear * 100:.2f}%")
st.write(f"🔹 **Polynomial SVM Accuracy:** {acc_poly * 100:.2f}%")
st.write(f"🔹 **RBF SVM Accuracy:** {acc_rbf * 100:.2f}%")

best_model = max(
    [("Linear SVM", acc_linear), ("Polynomial SVM", acc_poly), ("RBF SVM", acc_rbf)],
    key=lambda x: x[1]
)

st.success(f"🏆 Best Performing Model: **{best_model[0]}**")

# --------------------------------------------------
# OUTPUT SECTION
# --------------------------------------------------
if predict_btn:
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0].max() * 100

    st.subheader("📌 Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
        explanation = (
            "Based on credit history and income pattern, "
            "the applicant is likely to repay the loan."
        )
    else:
        st.error("❌ Loan Rejected")
        explanation = (
            "Based on credit history and income pattern, "
            "the applicant is unlikely to repay the loan."
        )

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Model Confidence:** {confidence:.2f}%")

    st.info(f"📊 **Business Explanation:** {explanation}")
