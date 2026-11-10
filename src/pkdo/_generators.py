"""Dataset and prior knowledge generators functions for PKDO testing"""

import numpy as np
import random
from typing import List

from pylfit import Dataset, Rule

def add_noise_to_dataset(dataset, noise):
    _data = dataset._data.copy()
    for i, state in _data.iterrows():
        _data.iloc[i] = add_noise_to_state(state, dataset.domains(), noise)

    return Dataset(_data, dataset.features, dataset.targets)


def add_noise_to_state(state, variables_domain, noise, features=None):
    """Add noise in a state"""
    if noise == 0:  # Avoid further process if no noise added
        return state.copy()

    state = state.copy()
    for var in state.index:
        if features:
            if var not in features:
                continue
        if np.random.binomial(1, noise) == 1:
            state.loc[var] = random.sample(sorted(variables_domain[var]), 1)[0]
    return state


def partial_knowledge(rules: List[Rule], knowledge_coverage=0.5, targets=None):
    """
    Return partial knowledge by removing rules and pruning them of atoms.
    """
    if targets:
        rules = [rule for rule in rules if rule.head[0] not in targets]
    random_rules = random.sample(rules, int(len(rules) * knowledge_coverage))

    cropped_knowledge = []
    for rule in random_rules:
        # Keep one atom rule as they are
        if len(rule.body) == 1:
            cropped_knowledge.append(rule)
            continue
        # Remove one atom from two atoms rules
        elif len(rule.body) == 2:
            body = {var: value for var, value in random.sample(list(rule.body.items()), 1)}
            cropped_knowledge.append(Rule(rule.head, body))
        # Remove two atoms from two or more atoms rules
        else:
            body = {var: value for var, value in random.sample(list(rule.body.items()), 2)}
            cropped_knowledge.append(Rule(rule.head, body))

    return cropped_knowledge


def split_data(dataset, split=0.7):
    """Split a dataset into train and test data."""
    data = dataset._data.copy()
    i_train = random.sample(range(0, data.shape[0]), int(split * data.shape[0]))
    i_test = [ind for ind in range(0, data.shape[0]) if ind not in i_train]

    train_dataset = Dataset(data.iloc[i_train], dataset.features, dataset.targets)
    test_dataset = Dataset(data.iloc[i_test], dataset.features, dataset.targets)

    return train_dataset, test_dataset