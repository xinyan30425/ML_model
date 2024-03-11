import pymc3 as pm
import pandas as pd
import theano.tensor as tt
import os

# Load the dataset
df = pd.read_csv("/home/xliuvm1/new_england_combined_dataset_processed.csv")

# Convert 'YEAR' to categorical and get its codes
year_codes, _ = pd.factorize(df['YEAR'])
df['YEAR_CODE'] = year_codes
unique_years = df['YEAR'].nunique()

# Ensure the RACE column is treated as categorical with all expected categories
expected_races = ['White', 'Black', 'AmericanIndian', 'Asian', 'Hawaiian','Other']
df['RACE'] = pd.Categorical(df['RACE'], categories=expected_races)

# Convert the RACE, AGEG, and YEAR variables to dummy variables, prefixing to avoid name conflicts
df = pd.get_dummies(df, columns=['RACE', 'AGEG', 'YEAR'], prefix=['RACE', 'AGEG', 'YEAR'], drop_first=True)

# Drop any rows with missing values to ensure data integrity for modeling
df.dropna(inplace=True)

# Ensure all data is numeric for PyMC3 and Theano processing
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)

# Define the model in PyMC3
with pm.Model() as hierarchical_model:
    # Priors for fixed effects coefficients
    β = pm.Normal('β', mu=0, sigma=100, shape=df.drop(columns=['ASTHMA']).shape[1])

    # Priors for random effects
    σ_re = pm.InverseGamma('σ_re', alpha=0.01, beta=0.01)

    # Non-centered parameterization for random effects
    re_year_prior = pm.Normal('re_year_prior', mu=0, sigma=1, shape=unique_years)

    # The line for 're_year' has been adjusted to reflect that 're_year_prior' is already representing our prior belief about the random effect
    re_year = pm.Deterministic('re_year', re_year_prior * σ_re)

    # Expected value parameter (linear predictor)
    η = tt.dot(df.drop(columns=['ASTHMA']).values, β) + re_year[df['YEAR_CODE'].values]

    # Link function and probability of having asthma
    p = pm.math.sigmoid(η)

    # Likelihood (sampling distribution) of the data
    Y_obs = pm.Bernoulli('Y_obs', p=p, observed=df['ASTHMA'].values)

    # Perform MCMC
    trace = pm.sample(1000, tune=1000, target_accept=0.9, cores=1)

# Save the trace
pm.save_trace(trace, directory="/home/xliuvm1/outputs/ML_model", overwrite=True)



# trace = pm.load_trace('/Users/xinyanliu/Desktop/NEU/Apriqot/ML_model')

# with hierarchical_model:  # Make sure you have defined the model in the same way as before
#     summary = pm.summary(trace)
#     print(summary)


# Post-processing to calculate the estimated diabetes prevalence through post-stratification
# This will vary depending on how you've stratified your data
# You would sum the probabilities for each category and normalize by the population size or weight


