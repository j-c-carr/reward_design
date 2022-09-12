import random
import itertools
import numpy as np
from copy import deepcopy
from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar)


H = TypeVar('H')    # History
O = TypeVar('O')    # Observation
A = TypeVar('A')    # Action
S = TypeVar('S')    # Internal representation


def make_policies_consistent(initial_policies, env):
    """Given a list of policies, returna a list of policies with consisten history-action choices"""

    all_policies = []
    # 1. create all policies at time :t:, for each t
    # 2. create policies by choosing one policy for each t.

    all_pi_choices = []   # list of length :t:, giving all possible behaviours at time :t:

    for t in env.time_axis:

        # for each t, create a list of
        all_possible_pi_t = []  # all possible behaviours at time :t:

        # For each history in the environment, create a list of all possible actions taken in the :initial_policies:
        history_action_pairs = {}
        for h in env.H[t]:
            history_action_pairs[h] = set()
            # keep track of all actions selected by all policies in the group
            for pi in initial_policies:
                if h in pi[t].keys():
                    history_action_pairs[h].add(pi[t][h])

            # If no policy from the :initial_policies: follow that history, remove it when building the consistent policies.
            if len(history_action_pairs[h]) == 0:
                del history_action_pairs[h]

        for action_choices in itertools.product(*list(history_action_pairs.values())):
            pi_t = {}
            for i, h in enumerate(history_action_pairs.keys()):
                pi_t[h] = action_choices[i]

            all_possible_pi_t.append(pi_t)

        all_pi_choices.append(all_possible_pi_t)
        print(f't={t} done')

    # Create the consistent policies by collection policies from all acceptable actions.
    consistent_policies = []
    for pi in itertools.product(*all_pi_choices):
        consistent_policies.append(tuple(pi))

    return consistent_policies


def create_all_supported_policies(env):
    """
    Creates all policies for trajectories in the environment. IN PROGRESS
    """

    all_policies = []
    # 1. create all policies at time :t:, for each t
    # 2. create policies by choosing one policy for each t.
    all_pi_choices = []   # list of length :t:, giving all possible behaviours at time :t:

    for t in env.time_axis[:-1]:
        all_possible_pi_t = []      # all possible behaviours at time :t:
        # if t == 0:
        #     for a in env.actions:
        #         all_possible_pi_t.append({(): a})
        # else:
        # Create a list of histories for each list
        history_action_pairs = {}
        for h in env.H[t]:
            # ASSUME ALL ACTIONS ARE ALLOWED AT EACH TIME
            history_action_pairs[h] = env.actions

        # Enumerate all possible policy rules at time :t: by selecting every action for every history
        for action_choices in itertools.product(*list(history_action_pairs.values())):
            pi_t = {}
            for i, h in enumerate(env.H[t]):
                pi_t[h] = action_choices[i]
            all_possible_pi_t.append(pi_t)

        print(f't={t} done')
        all_pi_choices.append(all_possible_pi_t)

    print('Intermediate policies computed')

    for pi in itertools.product(all_pi_choices):
        print(pi)
        all_policies.append(tuple(pi))


def make_random_good_policy_mapping(env, num_random_actions=0) -> Tuple[Mapping[H, A]]:
    """Create a policy that goes left for :num_random_actions: time steps, and then goes left."""
    
    policy = []
    H_t = [()]      # All t-step histories occurring under policy pi

    for t in env.time_axis:
        prefs = {}

        if t == 0:
            a = random.choice(env.actions) if num_random_actions >= env.time_axis.shape[0] else env.actions[0]
            prefs[()] = a
            H_t = [a]

        else:
            next_H_t = []
            for h in H_t:
                for o in env.observations:
                    current_trajectory = [h] + [o] if t == 1 else [*h] + [o]

                    if tuple(current_trajectory) not in env.H[t]:
                        continue

                    if t > env.time_axis.max() - num_random_actions:
                        # print(f'Choosing random action at time {t}')
                        a = random.choice(env.actions)
                        prefs[tuple(current_trajectory)] = a
                        current_trajectory.append(a)
                    else:
                        prefs[tuple(current_trajectory)] = env.actions[0]
                        current_trajectory.append(env.actions[0])

                    next_H_t.append(tuple(current_trajectory))

            H_t = next_H_t

        policy.append(prefs)

    return tuple(policy)


def make_another_good_policy_mapping(env) -> Tuple[Mapping[H, A]]:
    policy = []
    H_t = [()]  # All t-step histories occuring under policy pi
    for t in env.time_axis:
        prefs = {}

        if t == 0:
            prefs[()] = 0
            H_t = [(0)]

        else:
            next_H_t = []
            for h in H_t:
                for o in env.observations:
                    current_trajectory = [h] + [o] if t == 1 else [*h] + [o]
                    # if t == 1:
                    #    current_trajectory = [h] + [o]
                    prefs[tuple(current_trajectory)] = env.actions[0]
                    current_trajectory.append(env.actions[0])

                    next_H_t.append(tuple(current_trajectory))

            H_t = next_H_t

        policy.append(prefs)

    return tuple(policy)


def make_bad_policy_mapping(observations, actions, time_axis) -> Tuple[Mapping[H, A]]:
    policy = []
    H_t = [()]  # All t-step histories occuring under policy pi
    for t in time_axis:
        prefs = {}

        if t == 0:
            prefs[()] = 0
            H_t = [(0)]

        else:
            next_H_t = []
            for h in H_t:
                for o in observations:
                    current_trajectory = [h] + [o] if t == 1 else [*h] + [o]
                    # if t == 1:
                    #    current_trajectory = [h] + [o]
                    prefs[tuple(current_trajectory)] = actions[0]
                    current_trajectory.append(actions[0])

                    next_H_t.append(tuple(current_trajectory))

            H_t = next_H_t

        policy.append(prefs)

    return tuple(policy)


def sampled_actions(policies, h, t):
    """returns a list of all actions chosen by a set of policies for a history :h:"""
    sampled_actions = set()
    for pi in policies:
        if h in pi[t].keys():
            sampled_actions.add(pi[t][h])
    return sampled_actions


def compute_pi_fringe(policies, env, no_steady_state=True):

    fringe_policies = []

    for pi in policies:
        for t in env.time_axis[:-1]:
            # pi[t] is a dictionary of (h, a) pairs
            for h, a in pi[t].items():
                fringe_actions = set(env.actions) - sampled_actions(policies, h, t) - sampled_actions(fringe_policies, h, t)

                # Create fringe that differs by exactly one action from pi
                # TODO: all proceeding trajectories that have the swapped action must also be swapped.
                for fringe_action in fringe_actions:
                    pi_fringe = deepcopy(pi)
                    pi_fringe[t][h] = fringe_action

                    # Check action differences are attainable in the environment
                    if no_steady_state:
                        in_supp = False
                        for o in env.observations:
                            if ((h, fringe_action, o) in env.rho[t+1].keys()) and (env.rho[t+1][(h, fringe_action, o)] > 0):
                                in_supp = True
                        if not in_supp:
                            break

                    # For each time for the rest of the policy, swap (h,a) with (h, a').
                    for t_dash in range(t+1, len(pi)):
                        for k in pi[t_dash].keys():
                            if k[:2*(t+1)-1] == (*h, a):
                                new_key = tuple(list(h) + [fringe_action] + list(k[2*(t+1)-1:]))
                                pi_fringe[t_dash][new_key] = pi_fringe[t_dash][k]
                                del pi_fringe[t_dash][k]

                    fringe_policies.append(pi_fringe)

    return tuple(fringe_policies)


