"""
This code is modified based on
    https://github.com/plgreenLIRU/Gaussian-Process-SparseGP-Tutorial
"""

import numpy as np
import matplotlib.pyplot as plt
from gp_algs import SparseVariationalGP, ExactGP, kmpp_gp
from sklearn.metrics import mean_squared_error
from time import time

# Make some 1D training data
total_num = 2000  # 500 training points
x_range = 10
# raw_points_x = np.linspace(0, x_range, total_num)    # Inputs evenly spaced between 0 and 10
raw_points_x = np.random.rand(total_num) * x_range    # Inputs randomly spaced between 0 and 100
raw_points_f = np.sin(raw_points_x)             # True function (f = sin(x))
raw_points_y = raw_points_f + 0.1 * np.random.randn(total_num)  # Observations

# Initial hyperparameters
initial_l = 0.5        # Lengthscale
initial_sigma = 0.2    # Noise standard deviation

# Points to be predicted
predict_x = np.linspace(0, x_range, 200)  # Make predictions
predict_num = len(predict_x)              # No. points where we make predictions
predict_y_mean = np.zeros(predict_num)     # Initialise GP mean predictions
predict_y_std = np.zeros(predict_num)      # Initialise GP std predictions

# Train sparse GP
inducing_num = 8                # No. sparse points
candidate_num = 100      # No. of candidate sets of sparse points analysed

sgp, gp, kmpp = True, True, True

# Initialize Sparse GP regression
if sgp:
    print("##################### Sparse Variational GP #####################")
    sgp = SparseVariationalGP()

    # Load data to GP regression
    sgp.load_data(raw_points_x, raw_points_y, total_num)

    # Initialize the trainer and train the GP
    sgp.init_trainer(initial_l, initial_sigma, inducing_num)
    lb_best, elapsed_time = sgp.train(candidate_num)
    # lb_best, elapsed_time = sgp.train_greedy(candidate_num)

    # Print results
    print('Maximum lower bound:', np.round(lb_best, 3))
    print('Hyperparameters:', np.round(sgp.width_l, 3), np.round(sgp.sigma, 3))
    print('Elapsed time:', np.round(elapsed_time, 3))

    for n in range(predict_num):
        local_x = predict_x[n]
        predict_y_mean[n], predict_y_std[n] = sgp.predict(local_x)

    t0 = time()
    y_pred = np.array([sgp.predict(x_)[0][0] for x_ in raw_points_x])
    t1 = time()
    print('RMSE:', np.round(mean_squared_error(raw_points_y, y_pred, squared=False), 3))
    print('Prediction time:', np.round(t1 - t0, 3))

    # Plot results
    plt.figure()
    plt.grid()
    plt.plot(raw_points_x, raw_points_y, '.', color='blue', label='Full dataset')
    plt.plot(sgp.inducing_points_x, sgp.inducing_points_y, 'o',
             markeredgecolor='black', markerfacecolor='red',
             markeredgewidth=1.5, markersize=10, label='Sparse dataset')
    plt.plot(predict_x, predict_y_mean, 'black', label='Sparse GP')
    plt.plot(predict_x, predict_y_mean + 3 * predict_y_std, 'black')
    plt.plot(predict_x, predict_y_mean - 3 * predict_y_std, 'black')

    plt.xlabel('x')
    plt.title("Sparse variational GP")
    plt.legend()
    plt.show()

if gp:
    print("##################### Original GP #####################")

    gp = ExactGP()
    gp.load_data(raw_points_x, raw_points_y, total_num)
    gp.init_trainer(initial_l, initial_sigma)
    elapsed_time = gp.train()

    # Print results
    print('Hyperparameters:', np.round(gp.width_l, 3), np.round(gp.sigma, 3))
    print('Elapsed time:', np.round(elapsed_time, 3))

    for n in range(predict_num):
        local_x = predict_x[n]
        predict_y_mean[n], predict_y_std[n] = gp.predict(local_x)

    t0 = time()
    y_pred = np.array([gp.predict(x_)[0][0] for x_ in raw_points_x])
    t1 = time()
    print('RMSE:', np.round(mean_squared_error(raw_points_y, y_pred, squared=False), 3))
    print('Prediction time:', np.round(t1 - t0, 3))


    # Plot results
    plt.figure()
    plt.grid()
    plt.plot(raw_points_x, raw_points_y, '.', color='blue', label='Input data')
    plt.plot(predict_x, predict_y_mean, 'black', label='Exact GP')
    plt.plot(predict_x, predict_y_mean + 3 * predict_y_std, 'black')
    plt.plot(predict_x, predict_y_mean - 3 * predict_y_std, 'black')
    plt.xlabel('x')
    plt.title("Exact GP")
    plt.legend()
    plt.show()

if kmpp:
    print("##################### kmeans++ heuristical sparse GP #####################")
    kmppgp = kmpp_gp.SparseKmppGP()

    # Load data to GP regression
    kmppgp.load_data(raw_points_x, raw_points_y, total_num)

    # Initialize the trainer and train the GP
    kmppgp.init_trainer(initial_l, initial_sigma, inducing_num)
    elapsed_time = kmppgp.train(candidate_num)
    # lb_best, elapsed_time = sgp.train_greedy(candidate_num)

    # Print results
    print('Hyperparameters:', np.round(kmppgp.width_l, 3), np.round(kmppgp.sigma, 3))
    print('Elapsed time:', np.round(elapsed_time, 3))

    for n in range(predict_num):
        local_x = predict_x[n]
        predict_y_mean[n], predict_y_std[n] = kmppgp.predict(local_x)

    t0 = time()
    y_pred = np.array([kmppgp.predict(x_)[0][0] for x_ in raw_points_x])
    t1 = time()
    print('RMSE:', np.round(mean_squared_error(raw_points_y, y_pred, squared=False), 3))
    print('Prediction time:', np.round(t1 - t0, 3))

    # Plot results
    plt.figure()
    plt.grid()
    plt.plot(raw_points_x, raw_points_y, '.', color='blue', label='Full dataset')
    plt.plot(kmppgp.inducing_points_x, kmppgp.inducing_points_y, 'o',
             markeredgecolor='black', markerfacecolor='red',
             markeredgewidth=1.5, markersize=10, label='Sparse dataset')
    plt.plot(predict_x, predict_y_mean, 'black', label='Sparse GP')
    plt.plot(predict_x, predict_y_mean + 3 * predict_y_std, 'black')
    plt.plot(predict_x, predict_y_mean - 3 * predict_y_std, 'black')

    plt.xlabel('x')
    plt.title("Sparse kmeans++ GP")
    plt.legend()
    plt.show()