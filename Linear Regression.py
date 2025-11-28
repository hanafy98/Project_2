import numpy as np
import matplotlib.pyplot as plt


x_train = np.array ([1.0,2.0, 3.0, 4.0,])
y_train = np.array([500.0,1000.0,1500.0, 2000.0,])


plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population")
plt.xlabel("Population (10,000s)")
plt.ylabel("Profit ($10,000s)")
plt.show()


def compute_cost (x, y, w, b):

    m = x.shape[0]
    cost_sum = 0
    for i in range (m):
        f_wb = w * x[i] + b
        cost_sum = (f_wb - y[i]) ** 2

    total_cost = cost_sum / (2 * m)

    return total_cost
print(f" compute cost is: {compute_cost(x_train, y_train, 0, 0)}")

def compute_gradient (x, y, w, b):
    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0


    for i in range (m):
        fw_b = w * x[i] + b
        error = fw_b - y[i]
        dj_dw += error * x[i]
        dj_db += error

    dj_dw/= m
    dj_db/= m
    return dj_dw, dj_db

dj_dw, dj_db = compute_gradient(x_train, y_train, w = 0.0, b = 0.0)
print(dj_dw)
print(dj_db)


def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    w,b = w_init, b_init
    j_history = []
    for i in range (num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w-= alpha * dj_dw
        b-= alpha * dj_db

        if i % 100 == 0:
            j_history.append(compute_cost(x, y, w,b))
            print(f"Iter {i:4d}: J = {j_history[-1]:.3f}")
    return w, b, j_history

w_init, b_init = 0.0, 0.0
alpha = 0.01
iters = 4000

w_final, b_final, J_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iters
)
print("Learned parameters:", w_final, b_final)

preds = w_final * x_train + b_final

plt.scatter(x_train, y_train, c='r', marker='x')
plt.plot(x_train, preds, c='b')
plt.title("Linear Regression Fit")
plt.xlabel("Population (10,000s)")
plt.ylabel("Profit ($10,000s)")
plt.show()

for pop in [3.5, 7.0]:
    profit = w_final * pop + b_final
    print(f"Pop = {pop*10000:.0f}, Predicted profit = ${profit*10000:.2f}")
