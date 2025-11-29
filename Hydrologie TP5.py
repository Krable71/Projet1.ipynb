import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt


# QUESTION 1

Q = np.array([
    23.5, 78.2, 100.5, 94.7, 124.8, 160.7, 130.8, 88.7, 67.5, 104.2,
    124.8, 110.9, 100.7, 104.9, 55.7, 38.1, 158.4, 144.2, 96.1, 77.1,
    130.7, 114.9, 96.5, 99.4, 103.4
])

# Statistiques
mu = 101.176
sigma = 33.176
n = len(Q)

print("QUESTION 1 : Statistiques \n")

print(f"Moyenne μ = {mu:.3f} m³/s")
print(f"Ecart-type σ = {sigma:.3f} m³/s\n")


# d) — Probabilité d'excéder 180 et 200 m³/s


def proba_exceed(x, mu, sigma):
    z = (x - mu) / sigma
    p = 1 - stats.norm.cdf(z)
    T = np.inf if p == 0 else 1 / p
    return p, T, z

levels = [180, 200]
for L in levels:
    p, Tret, z = proba_exceed(L, mu, sigma)

    print(f"Débit {L} m³/s : z={z:.3f}, P={p*100:.6f}, T={Tret:.1f} ans")


# e) — Débits de conception T=10,25,100 ans


Tr_values = [10, 25, 100]

print(" Débits de conception (normale)")

for Tr in Tr_values:
    p_non_exceed = 1 - 1/Tr  # P(Q ≤ q_T)
    z = stats.norm.ppf(p_non_exceed)
    Qt = mu + sigma * z
    print(f"T={Tr} ans : Q_T = {Qt:.2f} m³/s (z={z:.3f})")

print("\n")

# ------------------------------------------------------------
# QUESTION 2 — Données intensités (2011–2019)
# ------------------------------------------------------------

I = np.array([10.0, 5.2, 16.6, 20.9, 8.0, 5.0, 19.1, 36.4, 7.6])
n2 = len(I)

# ------------------------------------------------------------
# a) — Probabilités de Weibull P = m / (n+1)
# ------------------------------------------------------------

order = np.argsort(I)              # indices triés (croissant)
I_sorted = I[order]                # intensités triées
ranks = np.arange(1, n2+1)         # m = 1,2,...,n
P_weibull = ranks / (n2 + 1)       # probabilités de Weibull

table = pd.DataFrame({
    "I (cm/h)": I_sorted,
    "rang m": ranks,
    "P_Weibull": P_weibull
})

print("a) — Table Weibull ")
print(table, "\n")


# b) — Paper-probability plots (Normal & Lognormal)


z_vals = stats.norm.ppf(P_weibull)

# Ajustement linéaire (Normal)
slope_norm, intercept_norm, r_norm, p_norm, se_norm = stats.linregress(z_vals, I_sorted)
R2_norm = r_norm**2

# Ajustement linéaire (Lognormal)
lnI = np.log(I_sorted)
slope_ln, intercept_ln, r_ln, p_ln, se_ln = stats.linregress(z_vals, lnI)
R2_lognormal = r_ln**2

print("b) — Qualité d'ajustement")
print(f"R² Normal     = {R2_norm:.4f}")
print(f"R² Lognormal  = {R2_lognormal:.4f}")
print("→ La distribution lognormale ajuste mieux les intensités.\n")

# ------------------------------------------------------------
# Tracés graphiques
# ------------------------------------------------------------

# Normal
plt.figure(figsize=(7,5))
plt.scatter(z_vals, I_sorted)
plt.plot(z_vals, intercept_norm + slope_norm*z_vals)
plt.xlabel("Quantile normal (z)")
plt.ylabel("Intensité I (cm/h)")
plt.title("Probabilité papier — Distribution Normal")
plt.grid(True)
plt.show()

# Lognormal
plt.figure(figsize=(7,5))
plt.scatter(z_vals, lnI)
plt.plot(z_vals, intercept_ln + slope_ln*z_vals)
plt.xlabel("Quantile normal (z)")
plt.ylabel(" Intensité I (cm/h) ")
plt.title("Probabilité papier — Distribution Lognormal")
plt.grid(True)
plt.show()

