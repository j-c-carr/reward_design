from scipy.optimize import minimize
import numpy as np
np.random.seed(0)

from plotting import plot_histogram_of_trajectory_distribution
from dataclasses import dataclass
from abc import ABC
from typing import (Callable, Dict, Generic, Iterator, Iterable,
                    Mapping, Optional, Sequence, Tuple, TypeVar)

from complex_environments import LineWorld, WindyLineWorld

# Creating the transition probability tensor
from policies import compute_pi_fringe, make_random_good_policy_mapping, make_policies_consistent, make_another_good_policy_mapping


def unbounded_reward_design(env, t_max, pi_g, initial_policies):
    """Unbounded reward design algorithm"""

    assert t_max <= env.time_axis.max() + 1, f'environment stops at T={env.time_axis.max()} but preferences specified until T={t_max}'
    pi_fringe = compute_pi_fringe(pi_g, env)

    # Each :pi: induces a distribution over T-step action-observation trajectories
    H_T = env.H[np.where(env.time_axis == t_max)[0][0] + 1]  # first time is ()

    rho_g = np.zeros((len(pi_g), len(H_T)))
    rho_fringe = np.zeros((len(pi_fringe), len(H_T)))

    for i, pi in enumerate(pi_g):
        rho_g[i] = env.compute_exp_visit(pi, t_max)
        assert rho_g[i].sum().astype(np.float32) == 1.0, f'Distribution must sum to one but pi_(g, {i}) sums to {rho_g[i].sum()}'

    j = 0
    for i1, pi1 in enumerate(pi_fringe):
        rho_fringe[i1] = env.compute_exp_visit(pi1, t_max)

        # Check if fringe policies have the same distribution as any of the good policies
        for i2 in range(rho_g.shape[0]):
            if np.abs((rho_fringe[i1] - rho_g[i2])).max() == 0:
                j += 1
                print('not realizable')
        assert rho_fringe[i1].sum().astype(np.float32) == 1.0, f'Distribution must sum to one but pi_(b, {i1}) sums to {rho_fringe[i1].sum()}'
    print(f'num unrealizable policies: {j} out of {len(pi_fringe)}')

    # create linear program constraints
    constraints = []
    for k1 in range(1, rho_g.shape[0]):
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.dot(rho_g[0], x) - np.dot(rho_g[k1], x)
        })

    for k2 in range(rho_fringe.shape[0]):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: np.dot(rho_g[0], x) - np.dot(rho_fringe[k2], x)
        })

    # Set bounds and initial guess for reward function
    r_max = 10_000
    bounds = np.array([(-r_max, r_max) for _ in range(len(H_T))])
    x_init = np.zeros(len(H_T))

    objective = lambda x: -(np.dot(rho_g[0], x) - np.dot(rho_fringe[0], x))

    solution = minimize(objective, x_init, method='SLSQP', bounds=bounds, constraints=constraints)

    if np.abs(solution.fun) == 0.0:
        # plot_histogram_of_trajectory_distribution(H_T, rho_g[0])
        print('No reward function found')
        print('\n ------------------ Pi_init -------------------\n')
        for pi_init in initial_policies:
            print('\t --------- pi_init ---------------')
            print(pi_init)
        print('\n ------------------ Pi_G -------------------\n')
        for ix, pi__g in enumerate(pi_g):
            plot_histogram_of_trajectory_distribution(f'./figures/tests/rho_g{ix}', H_T, rho_g[ix])
            print('\t --------- pi_g ---------------')
            print(pi__g)
        print('\n ------------------ Pi_fringe -------------------\n')
        for jx, pi_f in enumerate(pi_fringe):
            plot_histogram_of_trajectory_distribution(f'./figures/tests/rho_fringe{jx}', H_T, rho_fringe[jx], color='r')
            print('\t --------- PI_F ---------------')
            print(pi_f)
        return None

    else:
        print('Reward function found!')
        print('\n ------------------ Pi_init -------------------\n')
        for pi_init in initial_policies:
            print('\t --------- pi_init ---------------')
            print(pi_init)
        print('\n ------------------ Pi_G -------------------\n')
        for ix, pi__g in enumerate(pi_g):
            #plot_histogram_of_trajectory_distribution(f'./figures/tests/rho_g{ix}', H_T, rho_g[ix])
            print('\t --------- pi_g ---------------')
            print(pi__g)
        print('\n ------------------ Pi_fringe -------------------\n')
        for jx, pi_f in enumerate(pi_fringe):
            #plot_histogram_of_trajectory_distribution(f'./figures/tests/rho_fringe{jx}', H_T, rho_fringe[jx], color='r')
            print('\t --------- PI_F ---------------')
            print(pi_f)
        print(solution.x)
        return solution.x


for _ in range(1):
    windy_line_world = WindyLineWorld(observations=[0, 1, 2], actions=[0, 1], time_axis=np.arange(0, 3), wind_strength=0.3)

    pi_g1 = make_random_good_policy_mapping(windy_line_world, num_random_actions=1)
    pi_g2 = make_random_good_policy_mapping(windy_line_world, num_random_actions=1)

    soap = make_policies_consistent([pi_g1, pi_g2], windy_line_world)

    unbounded_reward_design(windy_line_world, windy_line_world.time_axis.max(), soap, initial_policies=[pi_g1, pi_g2])

    # TOOD: make everything logger-based.