"""
This code is modified based on
    https://github.com/plgreenLIRU/Gaussian-Process-SparseGP-Tutorial
"""

import numpy as np
import matplotlib.pyplot as plt

from time import time
from gp_algs import SparseVariationalGP, ExactGP
from sklearn.metrics import mean_squared_error

# Make some 1D training data
total_num = 500  # 500 training points

gp_time = []
spgp_time = []
init_points = [10, 32, 100, 312, 500, 1000, 3120] # Shorter version
# Uncomment below for long example (~15hrs) used in presentation
# init_points = [10, 32, 100, 312, 500, 1000, 3120, 10000, 31200]
for total_num in init_points:
    x_range = 10
    # raw_points_x = np.linspace(0, x_range, total_num)    # Inputs evenly spaced between 0 and 10
    raw_points_x = np.random.rand(total_num) * x_range    # Inputs randomly spaced between 0 and 100
    raw_points_f = np.sin(raw_points_x)             # True function (f = sin(x))
    raw_points_y = raw_points_f + 0.1 * np.random.randn(total_num)  # Observations


    # Initial hyperparameters
    initial_l = 0.5        # Lengthscale
    initial_sigma = 0.2    # Noise standard deviation


    # Train sparse GP
    inducing_num = 6              # No. sparse points
    candidate_num = 2      # No. of candidate sets of sparse points analysed


    print("##################### Sparse Variational GP #####################")
    sgp = SparseVariationalGP(variational=True)

    # Load data to GP regression
    sgp.load_data(raw_points_x, raw_points_y, total_num)

    # Initialize the trainer and train the GP
    sgp.init_trainer(initial_l, initial_sigma, inducing_num)
    # lb_best, elapsed_time = sgp.train(candidate_num)
    lb_best, elapsed_time = sgp.train_greedy(candidate_num, False)
    spgp_time.append(elapsed_time)

    # Print results
    print('Maximum lower bound:', np.round(lb_best, 3))
    print('Hyperparameters:', np.round(sgp.width_l, 3), np.round(sgp.sigma, 3))
    print('Training time:', np.round(elapsed_time, 3))


    print("##################### Original GP #####################")

    gp = ExactGP()
    gp.load_data(raw_points_x, raw_points_y, total_num)
    gp.init_trainer(initial_l, initial_sigma)
    elapsed_time = gp.train()
    gp_time.append(elapsed_time)

    # Print results
    print('Hyperparameters:', np.round(gp.width_l, 3), np.round(gp.sigma, 3))
    print('Training time:', np.round(elapsed_time, 3))

# Plot results
plt.grid()
plt.plot(init_points, spgp_time, 'red', label='Sparse Variational GP', linewidth=3, zorder=5, marker='o')
plt.plot(init_points, gp_time, 'black', label='Exact GP', linewidth=3, zorder=5, marker='d')
plt.xlabel('Training Points')
plt.ylabel('Time, s')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
