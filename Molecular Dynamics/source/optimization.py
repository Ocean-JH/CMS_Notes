import numpy as np
import matplotlib.pyplot as plt
"""
    Python implementation of optimization algorithm

    Author: jhwang@BUAA
"""


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 *(x[1] - x[0]**2)])


class OptimizationAlgorithm:
    def __init__(self, f, f_grad, max_iters=10000, tol=1e-6, alpha=0.5, beta=0.5):
        self.f = f
        self.f_grad = f_grad
        self.max_iters = max_iters
        self.tol = tol
        self.alpha = alpha
        self.beta = beta

    def backtrack_line_search(self, x, descent_dir, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha
        else:
            alpha = float(alpha)

        if beta is None:
            beta = self.beta
        else:
            beta = float(beta)

        t = 1.0
        while self.f(x + t*descent_dir) > self.f(x) + alpha*t*np.dot(self.f_grad(x), descent_dir):
            t *= beta
        return t

    def steepest_descent(self, init_x, max_iters=None, tol=None):
        if max_iters is None:
            max_iters = self.max_iters
        else:
            max_iters = int(max_iters)

        if tol is None:
            tol = self.tol
        else:
            tol = int(tol)

        x = init_x
        path = [x]
        print("Iteration\t\t\t\tPosition\t\t\tdelta")
        print("------------------------------------------------------------")
        for i in range(max_iters):
            g = -self.f_grad(x)
            d = g
            alpha = self.backtrack_line_search(x, d)
            new_x = x + alpha * d
            if i % 50 == 0:
                position = [x[0], x[1], self.f(x)]
                delta = np.linalg.norm(g)
                print("{}\t\t\t{}\t\t\t{}".format(i, position, delta))
            path.append(new_x)
            if np.linalg.norm(d) < tol:
                break
            x = new_x
        return np.array(path)

    def conjugate_gradient(self, init_x, max_iters=None, tol=None):
        if max_iters is None:
            max_iters = self.max_iters
        else:
            max_iters = int(max_iters)

        if tol is None:
            tol = self.tol
        else:
            tol = int(tol)

        x = init_x
        path = [x]
        print("Iteration\t\t\t\tPosition\t\t\tdelta")
        print("------------------------------------------------------------")
        g = -self.f_grad(x)
        d = g
        for i in range(max_iters):
            alpha = self.backtrack_line_search(x, d)
            new_x = x + alpha * d
            if i % 10 == 0:
                position = [x[0], x[1], self.f(x)]
                delta = np.linalg.norm(g)
                print("{}\t\t\t{}\t\t\t{}".format(i, position, delta))
            path.append(new_x)
            g_next = -self.f_grad(new_x)
            beta = np.dot(g_next, g_next) / np.dot(g, g)
            d = g_next + beta * d
            g = g_next
            if np.linalg.norm(g) < tol:
                break
            x = new_x
        return np.array(path)


if __name__ == "__main__":
    ini_x = np.array([-0.7, 5])
    optimizer = OptimizationAlgorithm(rosenbrock, rosenbrock_gradient)
    sd_path = optimizer.steepest_descent(ini_x)
    print("\n\n================================================================================\n\n")
    cg_path = optimizer.conjugate_gradient(ini_x)

    sd_value, cg_value = [], []
    for x in sd_path:
        sd_value.append(rosenbrock(x))

    for x in cg_path:
        cg_value.append(rosenbrock(x))

    sd_value = np.array(sd_value)
    cg_value = np.array(cg_value)

    print("Algo\t\t\t\tFinal position\t\t\t\tIterations")
    print("------------------------------------------------------------------------------------------------------------------------")
    print("Steepest Descent\t\t\t\t{}\t\t\t\t{}".format(sd_path[-1], len(sd_path) - 1))
    print("Conjugate Gradient\t\t\t\t{}\t\t\t\t{}".format(cg_path[-1], len(cg_path) - 1))

    # Contour plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

    ax.plot(sd_path[:, 0], sd_path[:, 1], sd_value, color='red', linewidth=2)
    ax.plot(cg_path[:, 0], cg_path[:, 1], cg_value, color='blue', linewidth=2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Comparison of efficiency between the steepest descent method and conjugate gradient method')

    plt.show()
