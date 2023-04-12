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
x_range = 10
# raw_points_x = np.linspace(0, x_range, total_num)    # Inputs evenly spaced between 0 and 10
raw_points_x = np.random.rand(total_num) * x_range    # Inputs randomly spaced between 0 and 100
raw_points_f = np.sin(raw_points_x)             # True function (f = sin(x))
raw_points_y = raw_points_f + 0.1 * np.random.randn(total_num)  # Observations

# Plot results
plt.figure()
plt.grid()
plt.plot(raw_points_x, raw_points_y, '.', color='blue', label='Full dataset', zorder=0)
plt.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), 'black', label='Target Function', linewidth=3, zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')

plt.title("Function")
plt.legend()
plt.show()

# Initial hyperparameters
initial_l = 0.5        # Lengthscale
initial_sigma = 0.2    # Noise standard deviation

# Points to be predicted
predict_x = np.linspace(0, x_range, 200)  # Make predictions
predict_num = len(predict_x)              # No. points where we make predictions
predict_y_mean = np.zeros(predict_num)     # Initialise GP mean predictions
predict_y_std = np.zeros(predict_num)      # Initialise GP std predictions

# Train sparse GP
inducing_num = 10              # No. sparse points
candidate_num = 100      # No. of candidate sets of sparse points analysed


print("##################### Sparse Variational GP #####################")
sgp = SparseVariationalGP(variational=True)

# Load data to GP regression
sgp.load_data(raw_points_x, raw_points_y, total_num)

# Initialize the trainer and train the GP
sgp.init_trainer(initial_l, initial_sigma, inducing_num)
# lb_best, elapsed_time = sgp.train(candidate_num)
lb_best, elapsed_time, rmse_svgp = sgp.train_greedy(candidate_num, True)
print(sgp.inducing_points_x)

# Print results
print('Maximum lower bound:', np.round(lb_best, 3))
print('Hyperparameters:', np.round(sgp.width_l, 3), np.round(sgp.sigma, 3))
print('Training time:', np.round(elapsed_time, 3))

start_time = time()
for n in range(predict_num):
    local_x = predict_x[n]
    predict_y_mean[n], predict_y_std[n] = sgp.predict(local_x)
print('Prediction time:', np.round(time() - start_time))

print("##################### Sparse GP #####################")
sgp = SparseVariationalGP(variational=False)

# Load data to GP regression
sgp.load_data(raw_points_x, raw_points_y, total_num)

# Initialize the trainer and train the GP
sgp.init_trainer(initial_l, initial_sigma, inducing_num)
# lb_best, elapsed_time = sgp.train(candidate_num)
lb_best, elapsed_time, rmse_dtc = sgp.train_greedy(candidate_num, True)
print(sgp.inducing_points_x)

# Print results
print('Maximum lower bound:', np.round(lb_best, 3))
print('Hyperparameters:', np.round(sgp.width_l, 3), np.round(sgp.sigma, 3))
print('Training time:', np.round(elapsed_time, 3))

start_time = time()
for n in range(predict_num):
    local_x = predict_x[n]
    predict_y_mean[n], predict_y_std[n] = sgp.predict(local_x)
print('Prediction time:', np.round(time() - start_time))

print("##################### Original GP #####################")

gp = ExactGP()
gp.load_data(raw_points_x, raw_points_y, total_num)
gp.init_trainer(initial_l, initial_sigma)
elapsed_time = gp.train()

# Print results
print('Hyperparameters:', np.round(gp.width_l, 3), np.round(gp.sigma, 3))
print('Training time:', np.round(elapsed_time, 3))
start_time = time()
for n in range(predict_num):
    local_x = predict_x[n]
    predict_y_mean[n], predict_y_std[n] = gp.predict(local_x)
print('Prediction time:', np.round(time() - start_time))
predict_y_points = []
for x in raw_points_x:
    ym, ys = gp.predict(x)
    predict_y_points.append(ym[0][0])
predict_y_points = np.array(predict_y_points)
rmsegp = mean_squared_error(raw_points_y, predict_y_points, squared=False)


# Plot results
plt.figure()
plt.grid()

plt.plot(range(1,len(rmse_svgp) + 1), rmse_svgp, 'red', marker='o', label='Sparse Variational GP', linewidth=3, zorder=5)
plt.plot(range(1,len(rmse_svgp) + 1), rmse_dtc, 'blue', marker='d', label='Sparse GP (DTC)', linewidth=3, zorder=5)

plt.plot([1,len(rmse_svgp)], [rmsegp, rmsegp], 'black', linestyle='--', label='Full GP', linewidth=3, zorder=5)

# plt.plot(predict_x, predict_y_mean + 3 * predict_y_std, 'black', linestyle="--", label="$3\sigma$")
# plt.plot(predict_x, predict_y_mean - 3 * predict_y_std, 'black', linestyle="--")

plt.xlabel('Number of Inducing Points')
plt.ylabel('RMSE')
plt.title("Sparse variational GP")
plt.legend()
plt.show()
