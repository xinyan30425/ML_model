import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the dataset
df = pd.read_csv("/Users/xinyanliu/Desktop/NEU/Apriqot/BRFSS/new_england_combined_dataset.csv",low_memory=False) # ensure all columns have the appropriate data types

# Ensure the RACE column is treated as categorical with all expected categories
expected_races = ['White', 'Black', 'AmericanIndian', 'Asian', 'Hawaiian','Other']
df['RACE'] = pd.Categorical(df['RACE'], categories=expected_races)

# Convert the RACE and AGEG variables to dummy variables, prefixing to avoid name conflicts
df = pd.get_dummies(df, columns=['RACE', 'AGEG'], prefix=['RACE', 'AGEG'], drop_first=True)

# Ensure all predictor variables are correctly formatted, especially for categorical variables like 'YEAR'
df['YEAR'] = df['YEAR'].astype('category')

df.dropna(subset=['YEAR', 'DIABETES', 'OVER_65', 'SEX', 'EMPLOYMENT_STATUS', 'POVERTY', 'HAVE_DISABILITY'], inplace=True)

# Reset index of the DataFrame
df.reset_index(drop=True, inplace=True)

# Update the model formula to include the AGEG dummy variables
ageg_columns = ' + '.join(df.filter(like='AGEG_').columns)

# Define the model formula
model_formula = f"DIABETES ~ OVER_65 + SEX + EMPLOYMENT_STATUS + POVERTY + HAVE_DISABILITY + RACE_Black + RACE_AmericanIndian + RACE_Asian + RACE_Hawaiian + RACE_Other + {ageg_columns}"

# Fit the GLMM model
# 'YEAR' is treated as a random effect. We use 're_formula' to specify the random effects structure. An empty formula means we only consider random intercepts for YEAR.
model = smf.mixedlm(model_formula, df, groups=df["YEAR"], re_formula="~1")

result = model.fit()

# Print the results
print(result.summary())
