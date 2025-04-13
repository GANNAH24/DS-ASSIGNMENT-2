# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# %%
# Load the dataset
file_path = r"C:\Users\ganna\PycharmProjects\DS-ASSIGNMENT-2\loan_approval_dataset.csv"
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