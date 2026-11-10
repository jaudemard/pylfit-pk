"""A LFIT algorithm to learn prior knowledge and data optimal model.

Identify a set of rules that fit the prior knowledge and are adapted to the data.
"""

import sys

from pylfit import Dataset, Rule
from pylfit import RuleSetModel

from pylfit.algorithms._algorithm import Algorithm

class PKDO(Algorithm):
    """Rules learner based on prior knowledge specialization and data."""

    @staticmethod
    def fit(
        dataset: Dataset,
        prior_rules: list[Rule],
        specificity_threshold: float,
        learn_all: bool = False,
        coverage_threshold: float = None,
        pareto_limit: int = None,
        min_support: int = None,
        verbose: bool = False,
    ) -> RuleSetModel:
        """
        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of the system
            rules: list of pylfit.objects.Rule
        """
        if not isinstance(dataset, Dataset):
            raise ValueError(
                f"Dataset type not supported, PKDO expect {Dataset.__name__!s}"
            )

        # Use the accuracy threshold as the coverage threshold if none provided
        if not coverage_threshold:
            coverage_threshold = specificity_threshold

        sorted_rules = {}
        for rule in prior_rules:
            if rule.head not in sorted_rules:
                sorted_rules[rule.head] = []
            sorted_rules[rule.head].append(rule)

        final_rules = []

        # Add empty rule to start for every head without backg round knowledge
        if learn_all:
            for target in dataset.targets:
                values = [(target, val) for val in dataset.domain(target)]
                for val in values:
                    if val not in sorted_rules:
                        sorted_rules[val] = PKDO.create_single_atom_rules(
                            dataset.domains(),
                            val,
                            dataset.features)
                    else:
                        sorted_rules[val] += PKDO.create_single_atom_rules(
                            dataset.domains(),
                            val,
                            dataset.features
                            )

        for head in sorted_rules:
            if verbose:
                print(f"Start learning for {head} from prior knowledge and dataset.")

            learned_rules = PKDO._fit_target_value(
                dataset,
                head,
                sorted_rules[head],
                specificity_threshold=specificity_threshold,
                coverage_threshold=coverage_threshold,
                pareto_limit=pareto_limit,
                min_support=min_support,
                verbose=verbose,
            )
            final_rules += learned_rules

            if verbose:
                print(f"Found {len(learned_rules)} rules.")

        final_rules = PKDO._minimality(final_rules)

        return set(final_rules)

    @staticmethod
    def _rule_key(rule: Rule) -> frozenset:
        """Canonical, hashable key for a rule's body."""
        return frozenset((atom, rule.body[atom]) for atom in rule.body)

    @staticmethod
    def _fit_target_value(
        dataset: Dataset,
        head: tuple[str, str],
        influences: list[Rule],
        specificity_threshold: float,
        coverage_threshold: float,
        pareto_limit: int,
        min_support: int,
        verbose: bool,
    ) -> set[Rule]:
        """Learns the target minimal rules from prior knowledge specialization"""
        positives = dataset.positives(head[0], head[1])
        negatives = dataset.negatives(head[0], head[1])

        final_rules: tuple[Rule, tuple[float, int]] = set()

        seen = set()

        search_space = {influence: [influence] for influence in influences}

        # enhable learning as long as promosing rules can still be explored
        while search_space:
            pruned = []
            for prior in search_space:
                new_candidates = []
                for candidate in search_space[prior]:
                    # Avoid exploring candidate with minimal solution
                    subsumed = False
                    for f_rule in final_rules:
                        if f_rule.is_more_general_than(candidate):
                            subsumed = True
                            break
                    if subsumed:
                        continue

                    # Avoid re-exploring branch
                    key = PKDO._rule_key(candidate)
                    if key in seen:
                        continue
                    seen.add(key)

                    support = PKDO._support(candidate, positives)
                    if min_support:
                        if support < min_support:
                            # if the support is too low, specialization is useless
                            # the rule can be pruned
                            continue

                    specificity = PKDO._specificity(candidate, positives, negatives)
                    if specificity >= specificity_threshold:
                        if verbose:
                            print(
                                f"Accepted (specificity = {specificity:.3f},",
                                f"support = {support}/{len(positives)})",
                                f": {candidate}",
                            )
                        final_rules.add(candidate)
                        continue

                    # keep candidates on the side for specialization
                    new_candidates.append((candidate, (specificity, support)))
                if new_candidates:
                    search_space[prior] = new_candidates
                else:
                    pruned.append(prior)

            for prior in pruned:
                search_space.pop(prior, None)

            # Compute the final rules coverage of positive examples, stop if superior to threshold
            if len(final_rules) > 0:
                coverage = PKDO._compute_coverage(final_rules, positives)
                if verbose:
                    print(
                        f"Now matching {coverage * 100:.3f}% of the positive examples."
                    )
                if coverage >= coverage_threshold:
                    if verbose:
                        print("Reached satisfying coverage, stop learning.")
                    break

            # Get the pareto front of the left candidates to only keep promising ones
            frontier = {}
            for prior in search_space:
                frontier[prior] = []
                # Do not limit to the first frontier if allowed
                if pareto_limit is not None:
                    frontier[prior] = PKDO._pareto_front(search_space[prior])
                    new_candidates = search_space[prior]
                    while (len(frontier[prior]) < pareto_limit) and (
                        len(frontier[prior]) + len(new_candidates) >= pareto_limit
                    ):
                        new_candidates = [
                            c for c in new_candidates if c not in frontier[prior]
                        ]
                        frontier[prior] += PKDO._pareto_front(new_candidates)
                else:
                    # Use the Pareto front as the base
                    frontier[prior] = PKDO._pareto_front(search_space[prior])

            # Use specialization of the pareto front as the new search starter
            #
            next_starter = {}
            for prior in search_space:
                next_starter[prior] = set()
                for candidate in frontier[prior]:
                    next_starter[prior].update(
                        PKDO._least_specialization(candidate[0], dataset)
                    )
                search_space[prior] = list(next_starter[prior])

        # Ensure minimality
        minimal_final_rules = PKDO._minimality(final_rules)

        return minimal_final_rules

    @staticmethod
    def _dominates(score_a: tuple[float, int], score_b: tuple[float, int]) -> bool:
        """True if score_a Pareto-dominates score_b"""
        spe_a, sup_a = score_a
        spe_b, sup_b = score_b
        a_dominate = (spe_a > spe_b) and (sup_a > sup_b)
        return a_dominate

    @staticmethod
    def _pareto_front(scored_rules: list[tuple[Rule, tuple[float, int]]]) -> list[Rule]:
        """Keeps only the non-dominated rules from a list of (rule, score)."""
        scored_rules.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
        front = []
        for rule, score in scored_rules:
            is_dominated = False
            for other_rule, other_score in front:
                if PKDO._dominates(other_score, score):
                    is_dominated = True
                    break
            if not is_dominated:
                front = [(r, s) for r, s in front if not PKDO._dominates(score, s)]
                front.append((rule, score))
        return front

    # @staticmethod
    # def _compute_distance_to_prior(rule,prior_knowledge):
    #     raise NotImplementedError

    @staticmethod
    def _least_specialization(rule: Rule, dataset: Dataset) -> list[Rule]:
        """Generates all specialization of a rule."""
        specializations = []

        for feature in dataset.features:
            domain = dataset.domain(feature)

            if feature in rule.body:
                continue
            for val in domain:
                _body = dict(rule.body)
                _body[feature] = val
                new_rule = Rule(rule.head, _body)
                specializations.append(new_rule)
        return specializations

    @staticmethod
    def _specificity(
        rule: Rule, positives: list[tuple[str]], negatives: list[tuple[str]]
    ) -> float:
        """Computes a rule accuracy score.

        Count true positives and true negatives of a rule prediction
        on the observed initial states, to return the ratio of correct
        prediction over the total amount of unique initial states.

        Args:
            rule: a logical rule
            negatives: a list of negative initial states for the rule head
            positives: a list of positive initial states for the rule head

        Returns:
            accuracy score
        """
        if len(rule.body) == 0:
            return 1 - (len(negatives) / (len(positives) + len(negatives)))

        tn = 0

        for neg in negatives:
            if not rule.matches(neg):
                tn += 1

        return tn / len(negatives)

    @staticmethod
    def _support(rule, positives, get_states=False) -> int:
        """
        Extract the amount of states supporting a rule.

        Args:
        rule: A rule.
        positives: A list of states.
        get_states: If true, return the amount of matched states, and the matched states
            if

        """
        support = 0

        if get_states:
            matched_states = set()

        for pos in positives:
            if rule.matches(pos):
                support += 1
                if get_states:
                    matched_states.add(frozenset(pos.items()))

        if get_states:
            return support, matched_states

        return support

    @staticmethod
    def _compute_coverage(rules: tuple[Rule, float, int], positives) -> float:
        """
        Compute the coverage of a set of rules over positive examples.
        """
        matched_states = set()

        for r in rules:
            _, states = PKDO._support(r, positives, get_states=True)

            matched_states.update(states)

        return len(matched_states) / len(positives)

    @staticmethod
    def _minimality(rules):
        """Take a set of rules with the same head and make sur they're minimal."""
        minimal_rules = []

        if len(rules) == 1:
            return rules

        for rule in rules:
            if any(r.is_more_general_than(rule) for r in rules if r != rule):
                continue
            minimal_rules.append(rule)

        return minimal_rules

    @staticmethod
    def create_single_atom_rules(variables_domain, target, features):
        """Creates single atom rules for a given target."""
        single_atom_rules = []
        for feature in features:
            for val in variables_domain[feature]:
                single_atom_rules.append(Rule(target, {feature:val}))
        return single_atom_rules