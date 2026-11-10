import pandas as pd
import sys
from pathlib import Path

from pylfit import Dataset, RuleSetModel, Rule, WeightedRuleSetModel

from pkdo import PKDO
from pkdo import split_data, partial_knowledge, add_noise_to_dataset

DATASET_PATH = "/home/user/lfit/pylfit-pk/examples/data/mammalian.csv"
PKN_PATH = "/home/user/lfit/pylfit-pk/examples/data/rules/mammalian_real_rules"

TRAIN_SPLIT = 0.7
PKN_COVERAGE = 0.7

NOISE = 0.05

frame = pd.read_csv(DATASET_PATH).astype(str)
features = [column for column in frame.columns if ("t_1" in column)]
targets = [column for column in frame.columns if ("t_1" not in column)]

print(
    f"Mammalian: {len(frame)} observations, {len(features)} features"
)

full_dataset = Dataset(frame, features, targets)

# print(full_dataset.features)
# sys.exit()

train, test = split_data(dataset=full_dataset, split=TRAIN_SPLIT)

# Get real rules
real_rules = []
with open(PKN_PATH, "r") as mammal_real:
    for line in mammal_real.readlines():
        real_rules.append(Rule.from_string(line.strip()))

# Get real model
real_model = RuleSetModel(full_dataset, real_rules)
cropped_knowledge = partial_knowledge(
    real_rules, knowledge_coverage=PKN_COVERAGE
        )

noisy_train = add_noise_to_dataset(train, NOISE)
noisy_test = add_noise_to_dataset(test, NOISE)

PKDO_model = PKDO.fit(
    noisy_train,
    cropped_knowledge,
    specificity_threshold=0.95,
    learn_all=True,
    coverage_threshold=0.9,
    min_support=20,
    verbose=True
)


weighted_pkdo_model = WeightedRuleSetModel(train,PKDO_model)

# Get the model prediction over the test data
t_pred = [
    weighted_pkdo_model.predict_proba(data)
    for i, data in test._data.iterrows()
]

# Get the true model prediction over the test data
t_true = real_model.predict_proba(data[1] for i, data in test._data.iterrows())