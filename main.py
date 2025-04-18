# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# %%
# Load the dataset
#file_path = r"C:\Users\ganna\PycharmProjects\DS-ASSIGNMENT-2\loan_approval_dataset.csv"
file_path = r"C:\Users\elena\Downloads\archive\loan_approval_dataset.csv"

columns = [
    "Applicant_ID", "Age", "Income", "Credit_Score", "Loan_Amount", "Loan_Term",
    "Interest_Rate", "Employment_Status", "Debt_to_Income_Ratio", "Marital_Status",
    "Number_of_Dependents", "Property_Ownership", "Loan_Purpose", "Previous_Defaults"
]
data = pd.read_csv(file_path, header=0, names=columns)

# Display the first few rows of the dataset
print(data.head())

# %%
# Categorical Encoding
categorical_columns = ["Employment_Status", "Marital_Status","Property_Ownership", "Loan_Purpose"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


# %%
# Split the Data
X = data.drop(columns=['Loan_Purpose'])
y = data['Loan_Purpose']
print(X)
print(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Feature Scaling (Critical for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler!

# 2. Initialize and train KNN (start with k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 3. Predict and evaluate
y_pred_knn = knn.predict(X_test_scaled)

# 4. Metrics
print("\n=== KNN Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))


# %%
#NAIVE BAYES

# initialize the naive bayes classifier
nb_classifier = GaussianNB()

# train the classifier on the training data
nb_classifier.fit(X_train, y_train)

# make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# %%
# DECISION TREE

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))


# %%
# RANDOM FOREST IMPLEMENTATION
from sklearn.ensemble import RandomForestClassifier

# 1. Initialize Random Forest
# (Use 100 trees and set random_state for reproducibility)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train the model (no scaling needed for tree-based models!)
rf.fit(X_train, y_train)

# 3. Predict and evaluate
y_pred_rf = rf.predict(X_test)

# 4. Metrics
print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

