import numpy as np
from scipy.stats import chi2_contingency

# Your data
gender = ['m', 'f', 'm', 'f', 'm', 'f', 'm', 'f']
job = ['teacher', 'programming', 'teacher', 'police', 'teacher', 'programming', 'teacher', 'programming']

# Create a contingency table
observed = np.array([gender, job]).T
observed_table = np.array([[np.sum((observed[:, 0] == 'm') & (observed[:, 1] == 'teacher')),
                           np.sum((observed[:, 0] == 'm') & (observed[:, 1] == 'programming')),
                           np.sum((observed[:, 0] == 'm') & (observed[:, 1] == 'police'))],
                          [np.sum((observed[:, 0] == 'f') & (observed[:, 1] == 'teacher')),
                           np.sum((observed[:, 0] == 'f') & (observed[:, 1] == 'programming')),
                           np.sum((observed[:, 0] == 'f') & (observed[:, 1] == 'police'))]])

# Perform the chi-squared test
chi2, _, _, _ = chi2_contingency(observed_table)

# Calculate Cramér's V
n = np.sum(observed_table)
min_dim = min(observed_table.shape) - 1
v = np.sqrt(chi2 / (n * min_dim))

# Output the results
print(f"Cramer's V: {v}")
