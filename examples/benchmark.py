import copy
import random
from pathlib import Path


import numpy as np
import pandas as pd

from pylfit import Dataset, Rule, RuleSetModel, WeightedRuleSetModel
from pkdo import PKDO
from pkdo import add_noise_to_dataset, add_noise_to_state, partial_knowledge, split_data

# from scripts.generators import 

ACCURACY_THRESHOLD = 0.95

PKN_COVERAGE = 0.80

TRAIN_SPLIT = 0.70

NOISE_LIST = [0, 0.01, 0.05, 0.10, 0.12, 0.15, 0.20]

HERE = Path(__file__).resolve().parent

# List of tuples of (dataset_path, dataset_rules_path)
DATASETS_LIST = [
    (HERE / "data" / "mammalian.csv", HERE / "data" / "rules" / "mammalian_real_rules")
]

SEPARATOR_PRINT = "------------------"

random.seed(50)



def rules_check(rules_a, rules_b):
    """Check the percentage of rules 'a' in rules 'b' set."""
    raise NotImplementedError


def test_rules_on_noisy_dataset(rules, dataset, noise_list):
    datasets = []

    for noise in noise_list:
        _dataset = copy.deepcopy(dataset)
        if noise == 0:
            datasets.append(_dataset)
        else:
            datasets.append(add_noise_to_dataset(_dataset, noise))

    results = {"noise": [], "accuracy": [], "rule_size": [], "rule": []}

    for i, dataset_i in enumerate(datasets):
        for rule in rules:
            positives = dataset_i.positives(rule.head[0], rule.head[1])
            negatives = dataset_i.negatives(rule.head[0], rule.head[1])

            results["noise"].append(noise_list[i])
            results["accuracy"].append(
                PKDO._accuracy(rule, positives=positives, negatives=negatives)
            )
            results["rule_size"].append(len(rule.body))
            results["rule"].append(rule)

    return results


def main() -> None:

    for d in DATASETS_LIST:
        DATASET_PATH = d[0]
        PKN_PATH = d[1]

        print(f"\n{SEPARATOR_PRINT * 3}")
        print("dataset import...")
        frame = pd.read_csv(DATASET_PATH).astype(str)
        features = [column for column in frame.columns if ("t_1" in column)]
        targets = [column for column in frame.columns if ("t_1" not in column)]

        print(
            f"{DATASET_PATH.name}: {len(frame)} observations, {len(features)} features"
        )

        full_dataset = Dataset(frame, features, targets)

        print(
            f"Split in {int(TRAIN_SPLIT * 100)}% train, and {int((1 - TRAIN_SPLIT) * 100)}% test"
        )
        train, test = split_data(dataset=full_dataset, split=TRAIN_SPLIT)

        # Get real rules
        real_rules = []
        with open(PKN_PATH, "r") as mammal_real:
            for line in mammal_real.readlines():
                real_rules.append(Rule.from_string(line.strip()))

        # Get real model
        real_model = RuleSetModel(full_dataset, real_rules)

        print(f"\n{SEPARATOR_PRINT} Generating cropped knowledge")
        print(f"Coverage : {PKN_COVERAGE * 100}%")
        cropped_knowledge = partial_knowledge(
            real_rules, knowledge_coverage=PKN_COVERAGE
        )

        print(f"\n{SEPARATOR_PRINT} Generating noisy datasets")
        # Generate noisy datasets
        noisy_train = []
        for noise in NOISE_LIST:
            _ndataset = add_noise_to_dataset(train, noise)
            noisy_train.append(_ndataset)

        noisy_tests = []
        for noise in NOISE_LIST:
            _ndataset = add_noise_to_dataset(test, noise)
            noisy_tests.append(_ndataset)

        print(f"\n{SEPARATOR_PRINT} Launching PKDO")

        # # Launch PKDO on prior knowledge ######################
        # print("\n-----------PKDO-------------")

        for i, n_d in enumerate(noisy_train):
            print(f"\nLearning on {NOISE_LIST[i]}% of noise.")

            # Learning phase
            PKDO_model = RuleSetModel(
                n_d,
                PKDO.fit(
                    n_d,
                    cropped_knowledge,
                    ACCURACY_THRESHOLD,
                    learn_all=True,
                    # pareto_limit=5,
                    verbose=False,
                ),
            )

            weighted_pkdo_model = WeightedRuleSetModel.from_rule_set_model(
                PKDO_model, n_d
            )

            # Get the model prediction over the test data
            t_pred = [
                weighted_pkdo_model.predict_proba(data[1])
                for data in test._data.iterrows()
            ]

            # Get the true model prediction over the test data
            t_true = real_model.predict_proba(data[1] for data in test._data.iterrows)

        print(PKDO_model.rules)

        # rules = sorted([rule.to_string() for rule in PKDO_model.rules])
        # with open("examples/tmp/learned.txt", "w+") as out_rules:
        #     for rule in rules:
        #         out_rules.write(f"{rule}\n")

        # print("---------test phase---------")


if __name__ == "__main__":
    main()
