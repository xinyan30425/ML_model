import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("/Users/xinyanliu/Desktop/NEU/Apriqot/BRFSS/new_england_combined_dataset_processed.csv")

# Initialize a dictionary to hold LabelEncoders for each categorical column
label_encoders = {}
categorical_columns = ['RACE', 'YEAR', 'STATE']

# Encode categorical variables using LabelEncoder and store them in the dictionary
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define your features and target variable
X = df[['POVERTY', 'SEX', 'EMPLOYMENT_STATUS', 'OVER_65', 'RACE', 'HAVE_DISABILITY', 'YEAR', 'STATE']]
Y = df['ASTHMA']  # Replace 'TARGET_VARIABLE' with 'ASTHMA', 'DIABETES', or 'COGNITIVE_DECLINE' as needed

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, predictions))

# 'new_data.csv' is the new dataset and 'model' is the trained Random Forest model
new_df = pd.read_csv("/Users/xinyanliu/Desktop/NEU/Apriqot/ACS/newengland_people_18_20_22_combined_processed.csv")
# Apply the stored LabelEncoders to the new dataset's categorical columns
for col, le in label_encoders.items():
    if col in new_df.columns:
        new_df[col] = le.transform(new_df[col].astype(str))

required_columns = ['POVERTY', 'SEX', 'EMPLOYMENT_STATUS', 'OVER_65', 'RACE', 'HAVE_DISABILITY', 'YEAR', 'STATE']  # list of columns used in the trained model
new_data_subset = new_df[required_columns]

predictions = model.predict(new_data_subset)

# Optionally, save the predictions to a CSV
new_df['ASTHMA_Predictions'] = predictions
new_df.to_csv("/Users/xinyanliu/Desktop/NEU/Apriqot/ACS/predictions.csv", index=False)

