import numpy as np

# 计算对数似然值
log_likelihood_guess1 = np.log(0.5) + np.log(0.3) + np.log(0.65)
log_likelihood_guess2 = np.log(0.5) + np.log(0.65) + np.log(0.65) + np.log(0.5)

print(f"Manual MLE for guess 1: {log_likelihood_guess1}")
print(f"Manual MLE for guess 2: {log_likelihood_guess2}")
