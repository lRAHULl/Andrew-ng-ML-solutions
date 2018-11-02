from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', header=None)

X = data.iloc[:, [0]].values
y = data.iloc[:, [1]].values

theta = zeros((2, 1))

plt.scatter(X[:, [0]], y, c='r')
plt.show()

X = append(ones((97, 1)), X, axis=1)

m = len(X)
n = len(theta)

plt.scatter(X[:, [1]], y, c='r')
plt.show()


def computeCost(X, y, theta):
    h = dot(X, theta)

    J = dot(1/(2 * m), sum(square(h-y)))

    return J


print(computeCost(X, y, theta))

theta = [[-1], [2]]
print(computeCost(X, y, theta))


def gradientDescent(X, y, theta, alpha, num_iter):
    # theta = theta - alpha(delta)
    # delta = 1/m * sum((h - y) .* x)

    m = len(y)
    n = len(theta)
    J_history = zeros((num_iter, 1))

    for _ in range(0, num_iter):
        h = dot(X, theta)
        diff = h - y
        delta = 1 / m * sum(multiply(diff, X), axis=0)
        for i in range(0, 2):
            temp = theta[i] - dot(alpha, delta[i])
            theta[i] = temp

        J_history[_] = computeCost(X, y, theta)

    return theta, J_history


theta, J_history = gradientDescent(X, y, theta, 0.01, 1000)

predict1 = dot(array([[1, 3.5000]]), theta)
predict2 = dot(array([[1, 7.0000]]), theta)

print(theta)

print(f'The predict1 is {predict1[0][0] * 10000}.')
print(f'The predict1 is {predict2[0][0] * 10000}.')

dot(dot(inv(dot(transpose(X), X)), transpose(X)), y)

# theta0_vals = linspace(-10, 10, 100)
# theta1_vals = linspace(-1, 4, 100)

# J_vals = zeros((len(theta0_vals), len(theta1_vals)))


# for i in range(len(theta0_vals)):
#     for j in range(len(theta1_vals)):
#         t = array(theta0_vals[i], theta1_vals[j])
#         J_vals[i, j] = computeCost(X, y, t)

# J_vals = transpose(J_vals)

# plt.contourf(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
# plt.xlabel('theta_0')
# plt.ylabel('theta_1')
# plt.plot(theta[0], theta[1])
# plt.show()
