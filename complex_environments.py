import numpy as np
from dataclasses import dataclass
from abc import ABC
from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar)


O = TypeVar('O')    # Observation
A = TypeVar('A')    # Action
S = TypeVar('S')    # Internal representation

@dataclass
class ComplexEnvironment(ABC):
    observations: Tuple[O]
    actions: Tuple[A]
    time_axis: List[int]
    rho: Tuple[np.ndarray] = None

    def initialize_histories(self):
        """
        Store transition probabilities for each t in :time_axis:
        """
        pass

    def initialize_transition_probabilities(self):
        """Returns the probabilities of each action given a history and action.
        history must length 2(t-1)."""
        pass

    def compute_exp_visit(self, pi, t_max: None):
        """Computes the distribution over action-observation t-step action-observation trajectories in the environment"""

        timesteps = self.time_axis if t_max is None else np.arange(0, t_max + 1)

        assert len(pi) >= (t_max-1), f'policy stops at time T={len(pi)} but trajectories go until T={t_max}'

        H_T = self.H[np.where(self.time_axis == t_max)[0][0] + 1]   # first index of H_T is (), so need to add +1

        rho = np.zeros(len(H_T))

        for i, h_T in enumerate(H_T):
            prob_of_h = 1

            # Compute the probability of each trajectory by multiplying probabilities of intermediate trajectories
            for t in timesteps:

                if prob_of_h == 0:
                    break

                else:
                    # Check if action a_t = h_T[2*t] was chosen by policy pi
                    if pi[t][tuple(h_T[:2*t])] != h_T[2*t]:
                        prob_of_h = 0

                    else:
                        # prob = self.rho[t + 1][(tuple(h_T[:2 * t]), h_T[2 * t], h_T[2 * t + 1])]
                        # if prob > 0:
                            # print('pi  follows this trajectory')
                            # print((tuple(h_T[:2 * t]), h_T[2 * t], h_T[2 * t + 1]))
                            # print(f'with probability {prob}')
                        prob_of_h *= self.rho[t+1][(tuple(h_T[:2*t]), h_T[2*t], h_T[2*t+1])]

            # If non-negative probability, add to list
            if prob_of_h > 0:
                # print(f'prob of {h_T} is {prob_of_h}')
                rho[i] = prob_of_h

        return rho


@dataclass
class LineWorld(ComplexEnvironment):
    """Complex environment which inherets action space, observation space, time axis, and transition probabilities"""

    def __init__(self, observations, actions, time_axis):
        assert len(actions) == 2, f'Line world accepts only 2 actions but received {len(actions)} actions.'

        self.observations = observations
        self.actions = actions  # left or right
        self.time_axis = time_axis  # left or right

        self.H = self.initialize_histories()
        self.rho = self.initialize_transition_probabilities()

    def initialize_histories(self):
        """
        Compute all histories accessible in the environment
        """
        H = []

        H_prev = [()]

        # Create a list of all histories
        for t in self.time_axis:
            H.append(H_prev)
            H_curr = []

            for h in H_prev:

                h = list(h)
                for a in self.actions:

                    if a == self.actions[0]:
                        if t == 0:
                            next_h = h + [a, self.observations[0]]
                        else:
                            if h[-1] == self.observations[0]:
                                next_h = h + [a, self.observations[0]]  # at the boundary
                            else:
                                next_h = h + [a, h[-1] - 1]  # move left
                    else:
                        if t == 0:
                            next_h = h + [a, self.observations[-1]]
                        else:
                            if h[-1] == self.observations[-1]:
                                next_h = h + [a, self.observations[-1]]  # at the boundary
                            else:
                                next_h = h + [a, h[-1] + 1]  # move right

                    H_curr.append(tuple(next_h))

            # H.append(H_curr)
            H_prev = H_curr

        return H

    def initialize_transition_probabilities(self):
        """Returns the probabilities of each action given a history and action.
        history must length 2(t-1)."""

        # Since rho only starts at t=1, create empty list at t=0
        rho = [{}]

        # Enumerate all possible histories
        for t in self.time_axis:
            rho_t = {}

            for h in self.H[t]:

                # Initialize all transition probabilities to zero
                for a in self.actions:
                    for o in self.observations:
                        rho_t[(h, a, o)] = 0

                # At time 0, start at boundaries
                if t == 0:
                    rho_t[(h, self.actions[0], self.observations[0])] = 1
                    rho_t[(h, self.actions[1], self.observations[-1])] = 1

                else:
                    # If at boundaries, stay in place or move in
                    if (h[-1] == self.observations[0]):
                        rho_t[(h, self.actions[0], self.observations[0])] = 1
                        rho_t[(h, self.actions[1], self.observations[1])] = 1
                    elif(h[-1] == self.observations[-1]):
                        rho_t[(h, self.actions[1], self.observations[-1])] = 1
                        rho_t[(h, self.actions[0], self.observations[-2])] = 1
                    # Else, move one step according to action
                    else:
                        rho_t[(h, self.actions[0], h[-1]-1)] = 1
                        rho_t[(h, self.actions[1], h[-1]+1)] = 1

            rho.append(rho_t)

        return tuple(rho)



class WindyLineWorld(ComplexEnvironment):
    """Complex environment which inherets action space, observation space, time axis, and transition probabilities"""

    def __init__(self, observations, actions, time_axis, wind_strength: float = 0.1):
        assert len(actions) == 2, f'Windy line world accepts only 2 actions but received {len(actions)} actions.'
        assert wind_strength > 0 and wind_strength < 1, f'expected wind strength to be in (0,1) but got {wind_strength}'

        self.observations = observations
        self.actions = actions  # left or right
        self.time_axis = time_axis
        self.wind_strength = wind_strength

        self.H = self.initialize_histories()
        self.rho = self.initialize_transition_probabilities()

    def initialize_histories(self):
        """
        Compute all histories accessible in the environment. E
        """

        #H = [[()]]
        H = []
        H_prev = [()]
        H.append(H_prev)

        # Create a list of all histories
        for t in range(self.time_axis.max() + 1):
            H_curr = []

            for h in H_prev:

                h = list(h)
                for a in self.actions:

                    # Each action has two possible trajectories, depending on wind strength

                    if a == self.actions[0]:
                        if t == 0:
                            next_h_1 = h + [a, self.observations[0]]
                            next_h_2 = h + [a, self.observations[-1]]
                        else:
                            if h[-1] == self.observations[0]:
                                next_h_1 = h + [a, self.observations[0]]  # at the left boundary
                                next_h_2 = h + [a, self.observations[1]]
                            else:
                                next_h_1 = h + [a, h[-1] - 1]  # move left
                                next_h_2 = h + [a, min(h[-1] + 1, self.observations[-1])]  # windy right
                    else:
                        if t == 0:
                            next_h_1 = h + [a, self.observations[-1]]
                            next_h_2 = h + [a, self.observations[0]]
                        else:
                            if h[-1] == self.observations[-1]:
                                next_h_1 = h + [a, self.observations[-1]]  # at the right boundary
                                next_h_2 = h + [a, self.observations[-2]]
                            else:
                                next_h_1 = h + [a, h[-1] + 1]  # move right
                                next_h_2 = h + [a, max(h[-1] - 1, self.observations[0])]  # move left

                    H_curr.append(tuple(next_h_1))
                    H_curr.append(tuple(next_h_2))

            H.append(H_curr)
            H_prev = H_curr

        return H

    def initialize_transition_probabilities(self):
        """Returns the probabilities of each action given a history and action.
        history must length 2(t-1)."""

        # Since rho only starts at t=1, create empty list at t=0
        rho = [{}]

        # Enumerate all possible histories
        for t in self.time_axis:
            rho_t = {}

            for h in self.H[t]:

                # Initialize all transition probabilities to zero
                for a in self.actions:
                    for o in self.observations:
                        rho_t[(h, a, o)] = 0

                # At time 0, start at boundaries
                if t == 0:
                    # Move left or blown right
                    rho_t[(h, self.actions[0], self.observations[0])] += 1 - self.wind_strength
                    rho_t[(h, self.actions[0], self.observations[-1])] += self.wind_strength

                    # Move right or blown left
                    rho_t[(h, self.actions[1], self.observations[-1])] += 1 - self.wind_strength
                    rho_t[(h, self.actions[1], self.observations[0])] += self.wind_strength

                else:
                    # If at boundaries, stay in place or move in
                    if h[-1] == self.observations[0]:
                        rho_t[(h, self.actions[0], self.observations[0])] = 1 - self.wind_strength
                        rho_t[(h, self.actions[0], self.observations[1])] = self.wind_strength
                        rho_t[(h, self.actions[1], self.observations[1])] = 1 - self.wind_strength
                        rho_t[(h, self.actions[1], self.observations[0])] = self.wind_strength

                    elif h[-1] == self.observations[-1]:
                        rho_t[(h, self.actions[1], self.observations[-1])] = 1 - self.wind_strength
                        rho_t[(h, self.actions[1], self.observations[-2])] = self.wind_strength
                        rho_t[(h, self.actions[0], self.observations[-2])] = 1 - self.wind_strength
                        rho_t[(h, self.actions[0], self.observations[-1])] = self.wind_strength
                    # Else, move one step according to action and wind strength
                    else:
                        rho_t[(h, self.actions[0], h[-1]-1)] = 1 - self.wind_strength
                        rho_t[(h, self.actions[0], h[-1]+1)] = self.wind_strength
                        rho_t[(h, self.actions[1], h[-1]+1)] = 1 - self.wind_strength
                        rho_t[(h, self.actions[1], h[-1]-1)] = self.wind_strength

            for a in self.actions:
                total_prob = 0
                for o in self.observations:
                    if (h, a, o) in rho_t.keys():
                        total_prob += rho_t[(h, a, o)]

                assert total_prob == 1, f'total prob should be one but got {total_prob}'

            rho.append(rho_t)

        return tuple(rho)
