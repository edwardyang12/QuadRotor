import numpy as np
from numpy.linalg import inv
from collections import namedtuple

DesiredState = namedtuple('DesiredState', 'pos vel acc yaw yawdot')

def get_helix_waypoints(t, n):
    """ The function generate n helix waypoints from the given time t
        output waypoints shape is [n, 3]
    """
    waypoints_t = np.linspace(t, t + 2*np.pi, n)
    x = 0.5*np.cos(waypoints_t)
    y = 0.5*np.sin(waypoints_t)
    z = waypoints_t

    return np.stack((x, y, z), axis=-1)

def get_MST_coefficients(waypoints):
    # generate MST coefficients for each segment, coeff is now 1D array [64,]
    coeff_x = MST(waypoints[:,0]).transpose()[0]
    coeff_y = MST(waypoints[:,1]).transpose()[0]
    coeff_z = MST(waypoints[:,2]).transpose()[0]
    return (coeff_x, coeff_y, coeff_z)

def get_poly_cc(n, k, t):
    """ This is a helper function to get the coeffitient of coefficient for n-th
        order polynomial with k-th derivative at time t.
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."

    cc = np.ones(n)
    D  = np.linspace(0, n-1, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0

    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])

    return cc

# Minimum Snap Trajectory
def MST(waypoints):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [8N,1] coeffitients matrix for the N+1 waypoints.
    1.The Problem
    Generate a full trajectory across N+1 waypoint made of N polynomial line segment.
    Each segment is defined as 7 order polynomial defined as follow:
    Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7
    Each polynomial has 8 unknown coefficients, thus we will have 8*N unknown to
    solve in total, so we need to come up with 8*N constraints.
    2.The constraints
    In general, the constraints is a set of condition which define the initial
    and final state, continuity between each piecewise function. This includes
    specifying continuity in higher derivatives of the trajectory at the
    intermediate waypoints.
    3.Matrix Design
    Since we have 8*N unknown coefficients to solve, and if we are given 8*N
    equations(constraints), then the problem becomes solving a linear equation.
    A * Coeff = B
    Let's look at B matrix first, B matrix is simple because it is just some constants
    on the right hand side of the equation. There are 8xN constraints,
    so B matrix will be [8N, 1].
    Now, how do we determine the dimension of Coeff matrix? Coeff is the final
    output matrix consists of 8*N elements. Since B matrix is only one column,
    thus Coeff matrix must be [8N, 1].
    Coeff.transpose = [a10 a11..a17...aN0 aN1..aN7]
    A matrix is tricky, we then can think of A matrix as a coeffient-coeffient matrix.
    We are no longer looking at a particular polynomial Pi, but rather P1, P2...PN
    as a whole. Since now our Coeff matrix is [8N, 1], and B is [8N, 8N], thus
    A matrix must have the form [8N, 8N].
    A = [A10 A12 ... A17 ... AN0 AN1 ...AN7
         ...
        ]
    Each element in a row represents the coefficient of coeffient aij under
    a certain constraint, where aij is the jth coeffient of Pi with i = 1...N, j = 0...7.
    """

    n = len(waypoints) - 1

    # initialize A, and B matrix
    A = np.zeros((8*n, 8*n))
    B = np.zeros((8*n, 1))

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # Constraint 1
    for i in range(n):
        A[i][8*i:8*(i+1)] = get_poly_cc(8, 0, 0)

    # Constraint 2
    for i in range(n):
        A[i+n][8*i:8*(i+1)] = get_poly_cc(8, 0, 1)

    # Constraint 3
    for k in range(1, 4):
        A[2*n+k-1][:8] = get_poly_cc(8, k, 0)

    # Constraint 4
    for k in range(1, 4):
        A[2*n+3+k-1][-8:] = get_poly_cc(8, k, 1)

    # Constraint 5
    for i in range(n-1):
        for k in range(1, 7):
            A[2*n+6 + i*6+k-1][i*8 : (i*8+16)] = np.concatenate((get_poly_cc(8, k, 1), -get_poly_cc(8, k, 0)))

    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff

if __name__ == '__main__':
    waypoints = get_helix_waypoints(0, 9)
    print(waypoints)
    (coeff_x, coeff_y, coeff_z) = get_MST_coefficients(waypoints)
    for _ in range(10):
        desired_state = generate_trajectory(time, 1.2, waypoints, coeff_x, coeff_y, coeff_z)
        time = time + 0.1
        print(desired_state)
