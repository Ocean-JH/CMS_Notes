import numpy as np
import matplotlib.pyplot as plt
"""
    Python implementation of integration algorithm

    Author: jhwang@BUAA
"""


class HarmonicOscillator:
    def __init__(self, k, m):
        self.k = k
        self.m = m

    def acceleration(self, x):
        return -self.k * x / self.m

    def analytical_solution(self, x0, v0, t):
        """
        Calculate the analytical solution for the 1D harmonic oscillator.

        Parameters:
        x0 : float
            Initial position.
        v0 : float
            Initial velocity.
        t : float
            Time at which to calculate the position and velocity.

        Returns:
        x : float
            Position at time t.
        v : float
            Velocity at time t.
        """
        w = np.sqrt(self.k / self.m)  # angular frequency
        x = x0 * np.cos(w*t) + v0/w * np.sin(w*t)
        v = -x0 * w * np.sin(w*t) + v0 * np.cos(w*t)
        return x, v


class ForwardEulerIntegrator:
    """
    A numerical integrator using the Forward Euler method.
    """

    def __init__(self, dt):
        """
        Initialize the integrator.

        Parameters:
        dt : float
            Time step for the numerical integration.
        """
        self.dt = dt

    def step(self, system, x, v):
        """
        Perform one integration step.

        Parameters:
        system : object
            The physical system to be integrated. It should have a method `acceleration(x)` that computes the acceleration.
        x : float
            Current position.
        v : float
            Current velocity.

        Returns:
        float, float
            Updated position and velocity.
        """
        a = system.acceleration(x)
        x_new = x + self.dt * v
        v_new = v + self.dt * a

        return x_new, v_new



class VerletIntegrator:
    def __init__(self, dt):
        self.dt = dt
        self.previous_x = None

    def step(self, system, x, v0=None):
        """
        Perform verlet integration on a system, stores x as previous_x and returns the new_x

        Args:
          system (class): simulation system class, should provide acceleration() method
          x (float): current position
          v0 (float): initial velocity

        Returns:
          new_x: float
              the position at the next time step
          current_v : float
              the velocity at the current time step
        """

        if self.previous_x is None:
            """
            On the first step, we can't do a full Verlet update because we
            don't have a previous_x. Instead, we estimate previous_x using a
            first-order Taylor expansion, taking into account initial velocity
            """
            self.previous_x = x - v0*self.dt + 0.5*system.acceleration(x) * self.dt ** 2

        # Calculate new position using Verlet algorithm
        a = system.acceleration(x)
        new_x = 2*x - self.previous_x + a * self.dt ** 2

        # Calculate the velocity for the current position
        if self.previous_x is not None:
            current_v = (new_x - self.previous_x) / (2*self.dt)
        else:
            current_v = v0

        # Update previous_x for the next step
        self.previous_x = x

        return new_x, current_v



class VelocityVerletIntegrator:
    """
    A numerical integrator using the Velocity Verlet method.
    """

    def __init__(self, dt):
        """
        Initialize the integrator.

        Parameters:
        dt : float
            Time step for the numerical integration.
        """
        self.dt = dt

    def step(self, system, x, v):
        """
        Perform one integration step.

        Parameters:
        system : object
            The physical system to be integrated. It should have a method `acceleration(x)` that computes the acceleration.
        x : float
            Current position.
        v : float
            Current velocity.

        Returns:
        float, float
            Updated position and velocity.
        """
        a = system.acceleration(x)
        x_new = x + self.dt * v + 0.5 * self.dt**2 * a
        a_new = system.acceleration(x_new)
        v_new = v + 0.5 * self.dt * (a + a_new)

        return x_new, v_new


class LeapfrogIntegrator:
    """
    LeapfrogIntegrator is a class for the Leapfrog integration method.

    Attributes:
        dt: The timestep for integration.

    Methods:
        step(system, x, v): Perform one step of Leapfrog integration.
    """
    def __init__(self, dt):
        self.dt = dt

    def step(self, system, x, v):
        """
        Parameters:
        x : float
            current position
        v : float
            velocity at current minus half timestep

        Returns:
        x_next : float
            position at next time step
        v_next : float
            velocity at next half timestep
        """
        a = system.acceleration(x)
        v_next = v + self.dt * a
        x_next = x + self.dt * v_next
        return x_next, v_next


def evolution(k, m, dt, x0, v0, T, algo: str, plot=False):
    """
    Visualize the time evolution of position and phase space trajectory.

    This function generates two subplots: one showing the position as a function
    of time, and one showing the phase space trajectory (velocity vs. position).

    Parameters:
    k : spring constant
    m : mass
    dt : time step
    x0 : initial position
    v0 : initial velocity
    T : total time

    Returns: times, positions, velocities
    """
    if algo not in ['euler', 'position verlet', 'velocity verlet', 'leapfrog']:
        raise Exception("algo must be one of the following: ['euler', 'position verlet', 'velocity verlet', 'leapfrog']")

    oscillator = HarmonicOscillator(k, m)

    if algo == 'euler':
        integrator = ForwardEulerIntegrator(dt)
    if algo == 'position verlet':
        integrator = VerletIntegrator(dt)
    if algo == 'velocity verlet':
        integrator = VelocityVerletIntegrator(dt)
    if algo == 'leapfrog':
        integrator = LeapfrogIntegrator(dt)

    times = []
    positions = []
    velocities = []

    if algo == 'leapfrog':
        x = x0
        v = v0 - 0.5 * dt * oscillator.acceleration(x0)
    else:
        x = x0
        v = v0

    step_num = int(T / dt)
    # Time evolution
    for i in range(step_num):
        times.append(i * dt)
        positions.append(x)
        velocities.append(v)
        x, v = integrator.step(oscillator, x, v)

    if plot is True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(times, positions)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position')
        ax1.set_title('Time Evolution of Position')

        ax2.plot(positions, velocities)
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Phase Space Trajectory')

        plt.tight_layout()
        plt.show()

    return times, positions, velocities



def compare_solutions(oscillator, integrator, x0, v0, T, dt):
    """
    Compare the numerical integration to the analytical solution.

    Parameters:
    oscillator : HarmonicOscillator
        The HarmonicOscillator system.
    integrator : VerletIntegrator
        The VerletIntegrator used for numerical integration.
    x0 : float
        Initial position.
    v0 : float
        Initial velocity.
    T : float
        Total time for the simulation.
    dt : float
        Time step for the numerical integration.

    Returns:
    None
    """
    times = np.arange(0, T, dt)
    num_positions = []
    num_velocities = []
    ana_positions = []
    ana_velocities = []

    x = x0
    v = v0
    for t in times:
        # Numerical solution
        num_positions.append(x)
        x, v = integrator.step(oscillator, x, v)
        num_velocities.append(v)

        # Analytical solution
        x_ana, v_ana = oscillator.analytical_solution(x0, v0, t)
        ana_positions.append(x_ana)
        ana_velocities.append(v_ana)

    # Calculate differences
    diff_positions = np.array(num_positions) - np.array(ana_positions)
    diff_velocities = np.array(num_velocities) - np.array(ana_velocities)
    return times, diff_positions, diff_velocities


def plot_differences(times, diff_positions, diff_velocities):
    """
    Plot the differences in position and velocity.

    Parameters:
    times : array
        Array of time points.
    diff_positions : array
        Array of absolute differences in position.
    diff_velocities : array
        Array of absolute differences in velocity.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(times, diff_positions)
    plt.xlabel('Time')
    plt.ylabel('Absolute difference in position')
    plt.title('Position Difference: Numerical vs Analytical')

    plt.subplot(122)
    plt.plot(times, diff_velocities)
    plt.xlabel('Time')
    plt.ylabel('Absolute difference in velocity')
    plt.title('Velocity Difference: Numerical vs Analytical')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Parameters
    k = 1.0
    m = 1.0
    dt = 0.1
    x0 = 2.0
    v0 = 0.0
    T = 100


    oscillator = HarmonicOscillator(k, m)

    verlet = VerletIntegrator(dt)
    leapfrog = LeapfrogIntegrator(dt)

    pv_times, pv_diff_positions, pv_diff_velocities = compare_solutions(oscillator, verlet, x0, v0, T, dt)
    lf_times, lf_diff_positions, lf_diff_velocities = compare_solutions(oscillator, leapfrog, x0, v0, T, dt)

    plot_differences(pv_times, pv_diff_positions, pv_diff_velocities)
    print("\n------------------------------------------------------------\n")
    print('Position ERROR =', max(pv_diff_positions))
    print('Velocity ERROR =', max(pv_diff_velocities))
    print(
        "\n\n========================================================================================================================\n\n")
    plot_differences(lf_times, lf_diff_positions, lf_diff_velocities)
    print("\n------------------------------------------------------------\n")
    print('Position ERROR =', max(lf_diff_positions))
    print('Velocity ERROR =', max(lf_diff_velocities))

    lf_times, lf_positions, lf_velocities = evolution(k, m, dt, x0, v0, T, algo='leapfrog', plot=True)
