import os
import numpy as np
import matplotlib.pyplot as plt

# Explicit imports from your package
from qmcpy.integrand.financial_option import FinancialOption
from qmcpy.stopping_criterion import CubQMCSobolG, CubMCG, CubQMCCLT, CubQMCLatticeG
from qmcpy.discrete_distribution import Sobol, IIDStdUniform, Halton, Lattice
# -------------------
# SETUP OUTPUT FOLDER
# -------------------
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

# -------------------
# PARAMETERS
# -------------------
initPrice = 40
strike = 40
interest = 0.06
vol = 0.2
tfinal = 1
d = 12
tol_vec = np.array([0.5, 0.1, 0.05, 0.01, 0.005])  # Absolute tolerances

# -------------------
# SAMPLERS
# -------------------
samplers = {
    'Sobol': Sobol(dimension=d),               # OK
    'IID': IIDStdUniform(dimension=d),         # OK
    'Halton': Halton(dimension=d, replications=8),   # OK for Student-t
    'Lattice': Lattice(dimension=d, replications=None)  # Must be None
}


# -------------------
# DATA STORAGE
# -------------------
n_required = {key: np.zeros_like(tol_vec) for key in samplers.keys()}
t_required = {key: np.zeros_like(tol_vec) for key in samplers.keys()}
prices = {key: 0.0 for key in samplers.keys()}

# -------------------
# INTEGRATION WITH STOPPING CRITERION
# -------------------
for key, sampler in samplers.items():
    print(f"Running sampler: {key}")
    integrand = FinancialOption(
        sampler,
        option="AMERICAN",
        call_put="PUT",
        start_price=initPrice,
        strike_price=strike,
        interest_rate=interest,
        volatility=vol,
        t_final=tfinal
    )

    for ii, abs_tol in enumerate(tol_vec):
        print(f"  Tolerance {abs_tol}")
        if key == 'Sobol':
            price, data = CubQMCSobolG(integrand, abs_tol=abs_tol).integrate()
        elif key == 'IID':
            price, data = CubMCG(integrand, abs_tol=abs_tol).integrate()
        elif key == 'Halton':
            price, data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
        elif key == 'Lattice':
            price, data = CubQMCLatticeG(integrand, abs_tol=abs_tol).integrate()

        prices[key] = price
        n_required[key][ii] = data.n_total
        t_required[key][ii] = data.time_integrate
        print(f"    Price: {price:.4f}, #paths: {data.n_total}, time: {data.time_integrate:.2f}s")

# -------------------
# SAVE GRAPHS
# -------------------
# Number of samples
plt.figure(figsize=(10, 6))
for key in samplers.keys():
    plt.loglog(tol_vec, n_required[key], '.-', label=key)
plt.xlabel('Absolute Error Tolerance')
plt.ylabel('Number of Samples Required')
plt.title('American Put Option - Sample Requirements')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
samples_path = os.path.join(output_dir, "american_put_samples.png")
plt.savefig(samples_path, dpi=300)
plt.show()
print(f"Saved graph to {samples_path}")

# Computation time
plt.figure(figsize=(10, 6))
for key in samplers.keys():
    plt.plot(tol_vec, t_required[key], '.-', label=key)
plt.xlabel('Absolute Error Tolerance')
plt.ylabel('Time Elapsed (s)')
plt.title('American Put Option - Computation Time')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
time_path = os.path.join(output_dir, "american_put_time.png")
plt.savefig(time_path, dpi=300)
plt.show()
print(f"Saved graph to {time_path}")
