import numpy as np
from scipy.stats import pointbiserialr

# Your data
gender = ['m', 'f', 'm', 'f', 'm', 'm', 'm', 'f', 'f', 'f']
salary = [10, 4, 15, 3, 9, 11, 9, 3, 2, 1]

# Convert gender to numerical values (0 for 'm', 1 for 'f')
gender_numeric = np.array([0 if g == 'm' else 1 for g in gender])

# Calculate point-biserial correlation
correlation, p_value = pointbiserialr(gender_numeric, salary)

# Calculate R-squared
r_squared = correlation**2

print(f"Correlation coefficient: {correlation:.4f}")
print(f"R-squared: {r_squared:.2%}")
