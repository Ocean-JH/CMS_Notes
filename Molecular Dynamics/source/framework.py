import numpy as np
from typing import List
import matplotlib.pyplot as plt
"""
    Basic framework of molecular dynamics

    Author: jhwang@BUAA
"""


class Particle:
    def __init__(self, position):
        self._position = np.array(position, dtype=float)

    def __str__(self):
        return f"Particle position =\n{self.position}"

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = np.array(value)


class Box:
    def __init__(self, lower, upper):
        self.lower = np.array(lower,dtype=float)
        self.upper = np.array(upper,dtype=float)
        self.length = self.upper - self.lower

    def __str__(self):
        return f"Box dimension {self.length}"


class BoundaryCondition:
    """Abstract base class for boundary conditions."""

    def apply(self, particles: List[Particle], box: Box) -> None:
        raise NotImplementedError

    def displacement(self, position1: np.ndarray, position2: np.ndarray, box: Box) -> np.ndarray:
        raise NotImplementedError


class PeriodicBoundaryCondition(BoundaryCondition):
    """Implements periodic boundary conditions."""

    def apply(self, particles: List[Particle], box: Box) -> None:
        """
        Applies periodic boundary conditions to all particles in the system.

        Parameters
        ----------
        particles : List[Particle]
            List of all Particle instances in the system.
        box : Box
            The Box instance defining the boundaries of the system.

        Returns
        -------
        None

        Notes
        -----
        This method updates the positions of particles in the system
        such that they adhere to the periodic boundary conditions.
        """
        for p in particles:
            p.position = (p.position - box.lower) % box.length + box.lower

    def displacement(self, position1: np.ndarray, position2: np.ndarray, box: Box) -> np.ndarray:
        """
        Computes the shortest displacement vector between two positions in a periodic system.

        Parameters
        ----------
        position1 : np.array
            Position of the first particle.
        position2 : np.array
            Position of the second particle.
        box : Box
            The box object representing the system's boundary.

        Returns
        -------
        np.array
            The displacement vector from position2 to position1 taking into account periodic boundary conditions.
        """
        dr = position1 - position2
        dr = dr - np.rint(dr / box.length) * box.length
        return dr

    def displacement_list(self, particles: List[Particle], box: Box) -> np.ndarray:
        """
        Computes the shortest displacement vector between two positions in a periodic system.

        Parameters
        ----------
        particles : List[Particle]
            Set of all Particles in the system.
        box : Box
            The box object representing the system's boundary.

        Returns
        -------
        np.array
            The displacement vector from position2 to position1 taking into account periodic boundary conditions.
        """
        r_vec = np.zeros((len(particles), len(particles) - 1, 3))
        for i in range(len(particles)):
            for p in particles:
                if (p.position == particles[i].position).all():
                    pass
                else:
                    r_vec[i] = particles[i].position - p.position
                    r_vec[i] = r_vec[i] - np.rint(r_vec[i] / box.length) * box.length
        return r_vec


class NeighborList:
    """Abstract base class for neighbor list."""

    def build(self):
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError


class VerletList(NeighborList):

    def __init__(self, particles, box, boundary_condition, cutoff, skin_depth):
        self.particles = particles
        self.box = box
        self.boundary_condition = boundary_condition
        self.cutoff = cutoff
        self.skin_depth = skin_depth

    def build(self):
        """Build neighbor list of the particle."""
        self.neighbor_list = {}
        self.previous_positions = {p: np.copy(p.position) for p in self.particles}

        for p1 in self.particles:
            self.neighbor_list[p1] = {}
            for p2 in self.particles:
                displacement = self.boundary_condition.displacement(p1.position, p2.position, self.box)
                if p1 != p2 and np.linalg.norm(displacement) < (self.cutoff + self.skin_depth):
                    self.neighbor_list[p1].update({p2: displacement})

    def update(self):
        max_displacement = max(
            np.linalg.norm(self.boundary_condition.displacement(p.position, self.previous_positions[p], self.box)) for p
            in self.particles)
        if max_displacement > self.skin_depth / 2:
            self.build()


class Potential:
    """Abstract base class for potential."""
    def energy_calc(self, particles, nn_list, start_dist, end_dist):
        raise NotImplementedError

    def force_calc(self, nn_list, start_dist, end_dist):
        raise NotImplementedError


class LJPotential(Potential):

    def __init__(self, particles, epsilon, sigma, cutoff='soft'):
        self.particles = particles
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

        if cutoff not in ['hard', 'soft']:
            raise ValueError('Invalid type of cutoff.')

    def _switching_function(self, r, start_dist, end_dist):
        # Check the distance against the switching start and cutoff distance
        if r < start_dist:
            return 1.0
        elif r > end_dist:
            return 0.0

        # Compute the normalized distance within the switching interval
        # t = 1 when r = start_dist;  t = 0 when r = end_dist
        t = (end_dist - r) / (end_dist - start_dist)

        # Compute the switching function based on the chosen method

        return 0.5 * (1 - np.cos(np.pi * t))

    def _switching_derivative(self, r, start_dist, end_dist):
        if r < start_dist or r > end_dist:
            return 0.0
        else:
            return -0.5 * (np.pi / (end_dist - start_dist)) * np.sin(np.pi * (end_dist - r) / (end_dist - start_dist))

    def _energy_calc(self, r):

        if self.cutoff == 'hard':
            V = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
            return V

        if self.cutoff == 'soft':
            V = (self.switching_function(r, start_dist, end_dist) *
                 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6))
            return V

    def _force_vector(self, r_vec, start_dist, end_dist):
        net_force = np.zeros_like(r_vec, dtype=float)

        r_magnitude = np.linalg.norm(r_vec)

        if r_magnitude <= end_dist:
            # Compute force without considering cutoff
            force_magnitude = 24 * epsilon * (
                        (2 * (sigma / r_magnitude) ** 12) - (sigma / r_magnitude) ** 6) / r_magnitude

            if self.cutoff == 'soft':
                switching_value = self._switching_function(r_magnitude, start_dist, end_dist)
                switching_derivative = self._switching_derivative(r_magnitude, start_dist, end_dist)
                force_magnitude *= switching_value
                # Add extra force due to derivative of switching function
                force_magnitude -= switching_derivative * 4 * epsilon * (
                            (sigma / r_magnitude) ** 12 - (sigma / r_magnitude) ** 6)

            # Add force vector to total force
            net_force = force_magnitude * (r_vec / r_magnitude)

        return net_force

    def _force_calc(self, r_vec, start_dist, end_dist):
        r_magnitude = np.linalg.norm(r_vec)

        if r_magnitude <= cutoff_distance:
            # Compute force without considering cutoff
            force_magnitude = 24 * epsilon * (
                    (2 * (sigma / r_magnitude) ** 12) - (sigma / r_magnitude) ** 6) / r_magnitude

            if self.cutoff == 'soft':
                switching_value = self.switching_function(r_magnitude, start_dist, end_dist)
                switching_derivative = self.switching_derivative(r_magnitude, start_dist, end_dist)
                force_magnitude *= switching_value
                # Add extra force due to derivative of switching function
                force_magnitude -= switching_derivative * 4 * epsilon * (
                        (sigma / r_magnitude) ** 12 - (sigma / r_magnitude) ** 6)
        else:
            force_magnitude = 0.0

        return force_magnitude

    def energy_calc(self, particles, nn_list, start_dist, end_dist):
        raise NotImplementedError

    def force_calc(self, nn_list, start_dist, end_dist):
        self.force = {}
        self.force_vector = {}
        for p1 in nn_list.keys():
            net_force = np.zeros((1, 3))
            for p2 in nn_list[p1]:
                r_vec = nn_list[p1][p2]
                force_vector = self._force_vector(r_vec, start_dist, end_dist)
                net_force += force_vector

            self.force_vector[p1] = net_force
            self.force[p1] = np.linalg.norm(net_force)





def genRandomParticles(natoms, system_scale=10):
    # Generate random atom positions
    dim = 3  # 3 dimensional
    minDist = 0.8  # minimum required distance between atoms
    positions = np.zeros((natoms, dim))
    positions[0] = np.random.rand(dim)
    for i in range(1, natoms):
        iter, maxIter = 0, 1e5
        while True and iter < maxIter:
            iter += 1
            newpos = np.random.rand(dim) * system_scale
            dist = newpos - positions[0:i]
            if np.all(np.linalg.norm(dist, axis=1) > minDist):
                break
        assert (iter < maxIter)
        positions[i] = newpos

    particles = []
    for i in range(natoms):
        particles.append(Particle(positions[i]))

    return particles


def neighborlist_calc(particles, system_scale, cutoff_dist, skin_depth, i=None):
    box = Box(np.array([0.0, 0.0, 0.0]), np.array([system_scale, system_scale, system_scale]))

    pbc = PeriodicBoundaryCondition()
    pbc.apply(particles, box)

    neighbor = VerletList(particles, box, pbc, cutoff_dist, skin_depth)
    neighbor.build()
    if i is None:
        nn_list = neighbor.neighbor_list
    elif type(i) is int:
        nn_list = neighbor.neighbor_list[particles[i - 1]]
    else:
        raise TypeError("i should be None or an integer.")

    return nn_list


def visualize(particles: List[Particle], system_scale, cutoff_dist, skin_depth, i: int):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 获取目标粒子和其近邻列表中的粒子
    target_p = particles[i - 1]

    neighbor_list = neighborlist_calc(particles, system_scale, cutoff_dist, 0)
    verlet_list = neighborlist_calc(particles, system_scale, cutoff_dist, skin_depth)

    neighbor_particles = neighbor_list[target_p]
    verlet_particles = verlet_list[target_p]

    verlet_hull = np.array([p for p in verlet_particles if p not in neighbor_particles])
    other_particles = np.array([p for p in particles if p not in verlet_particles])
    neighbor_positions = np.array([p.position for p in neighbor_particles])
    verlet_hull_positions = np.array([p.position for p in verlet_hull])
    positions = np.array([p.position for p in other_particles if p is not target_p])

    ax.scatter(target_p.position[0], target_p.position[1], target_p.position[2], s=100, marker="*", label='Particles',
               color='r')

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=100, label='Particles', color='grey')

    ax.scatter(neighbor_positions[:, 0], neighbor_positions[:, 1], neighbor_positions[:, 2], label='Neighbours', s=100,
               color='r')

    ax.scatter(verlet_hull_positions[:, 0], verlet_hull_positions[:, 1], verlet_hull_positions[:, 2], label='Verlet',s=100,
               color='y')

    center = target_p.position
    r1 = cutoff_dist

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = center[0] + r1 * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r1 * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r1 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='orange', alpha=0.3)

    r2 = cutoff_dist + skin_depth

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = center[0] + r2 * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r2 * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r2 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='wheat', alpha=0.3)

    vertices = np.array([[0, 0, 0],
                         [system_scale, 0, 0],
                         [system_scale, system_scale, 0],
                         [0, system_scale, 0],
                         [0, 0, system_scale],
                         [system_scale, 0, system_scale],
                         [system_scale, system_scale, system_scale],
                         [0, system_scale, system_scale]])

    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    for edge in edges:
        ax.plot3D(*zip(vertices[edge[0]], vertices[edge[1]]), color="grey")

    ax.set_title('Particle Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.axis('off')

    plt.show()


if __name__ == "__main__":
    # Parameters
    natoms = 100
    system_scale = 20
    start_dist = 3
    cutoff_dist = 6
    skin_depth = 2
    epsilon = 1.0
    sigma = 1.0

    # Generate random particles
    particles = genRandomParticles(natoms, system_scale)

    # Instantiate
    box = Box(np.array([0.0, 0.0, 0.0]), np.array([system_scale, system_scale, system_scale]))
    pbc = PeriodicBoundaryCondition()
    lj_potential = LJPotential(particles, epsilon, sigma, cutoff='soft')

    # Apply periodic boundary condition
    pbc.apply(particles, box)
    nn_list = neighborlist_calc(particles, system_scale, cutoff_dist, skin_depth)
    lj_potential.force_calc(nn_list, start_dist, cutoff_dist)

    nn_list_p1 = neighborlist_calc(particles, system_scale, cutoff_dist, skin_depth, 1)
    nn_list_p1_position = {}
    for p in nn_list_p1.keys():
        nn_list_p1_position.update({str(p.position): nn_list_p1[p]})

    print("Neighbor index\t\t\t\t\tposition\t\t\t\t\tdistance")
    print("----------------------------------------------------------------------------------------------------")
    for i, neighb in enumerate(nn_list_p1_position.keys()):
        print("{}\t\t\t\t\t{}\t\t\t\t\t{}".format(i + 1, neighb, np.linalg.norm(nn_list_p1_position[neighb])))
    print("====================================================================================================")

    print("Particle Index\t\t\t\t\tForce Vector\t\t\t\t\tForce")
    print("----------------------------------------------------------------------------------------------------")
    for i, p in enumerate(particles):
        print("{}\t\t\t\t\t{}\t\t\t\t\t{}".format(i + 1, lj_potential.force_vector[p], lj_potential.force[p]))
    print("====================================================================================================")

    visualize(particles, system_scale, cutoff_dist, skin_depth, 7)
