"""Seeded synthetic layouts. Explicitly not samples of industrial task prevalence."""

from dataclasses import replace
from fractions import Fraction as F
import random

from .core import Model, Problem


FAMILIES = ("explicit", "equivalent", "fork", "observable", "pure_intent", "impossible")


def worked_example(**changes) -> Problem:
    p = Problem(
        task_id="worked_example", layout_id="worked", family="observable", split="diagnostic",
        edges=(("S", "P", 20), ("P", "R", 12), ("R", "A", 8), ("P", "B", 25)),
        home="S", start="P", budget=90, horizon=8, prior=(F(3, 5), F(2, 5)),
        viewpoints=(("A",), ("B",)), observation_rows=(("P", F(19, 20), F(1, 20)),),
    )
    return replace(p, **changes)


def generate_tasks(config: dict, split: str) -> list[Problem]:
    if split == "sensitivity":
        result = []
        for budget in (80, 90, 110, 150):
            for ask in (0, 6, 20):
                for rel in (F(1, 2), F(3, 4), F(19, 20)):
                    result.append(worked_example(
                        task_id=f"sensitivity_B{budget}_Q{ask}_R{str(rel).replace('/', '-')}",
                        split=split, budget=budget, ask_cost=ask,
                        observation_rows=(("P", rel, 1 - rel),), horizon=config["horizon"],
                    ))
        return result
    if split not in {"development", "evaluation"}:
        raise ValueError("Unknown split")
    rng = random.Random(config["seed"] + (0 if split == "development" else 100_000))
    result = []
    for index in range(config[f"{split}_layouts"]):
        cohort = "_" + config["cohort"] if config.get("cohort") else ""
        layout = f"{split}{cohort}_{index:03d}"
        edges = (("S", "P", rng.randint(12, 24)), ("P", "R", rng.randint(8, 16)),
                 ("R", "A", rng.randint(6, 12)), ("P", "B", rng.randint(18, 32)),
                 ("P", "O", rng.randint(3, 8)))
        prior = rng.choice((F(1, 2), F(3, 5), F(4, 5)))
        reliability = rng.choice((F(7, 10), F(17, 20), F(19, 20)))
        ask_cost = rng.choice((0, 6, 20))
        slack = rng.choice((1.05, 1.3, 2.0))
        base = Problem(
            task_id=layout, layout_id=layout, family="base", split=split,
            edges=edges, home="S", start="S", budget=500, horizon=config["horizon"],
            prior=(prior, 1 - prior), viewpoints=(("A",), ("B",)),
            observation_rows=(), ask_cost=ask_cost,
        )
        model = Model(base)
        completion = [model.completion_cost("S", z) for z in (0, 1)]
        budget = round(slack * max(completion))
        for family in FAMILIES:
            p = replace(base, task_id=f"{layout}_{family}", family=family, budget=budget)
            if family == "explicit":
                p = replace(p, prior=(F(1), F(0)))
            elif family == "equivalent":
                p = replace(p, viewpoints=(("A",), ("A",)))
            elif family == "fork":
                p = replace(p, budget=round(1.05 * max(completion)),
                            observation_rows=(("R", reliability, 1 - reliability),))
            elif family == "observable":
                # A physically correlated unknown tag, not hidden intent read from images.
                site = "P" if index % 2 == 0 else "O"
                p = replace(p, observation_rows=((site, reliability, 1 - reliability),))
            elif family == "pure_intent":
                p = replace(p, observation_rows=(("P", F(1, 2), F(1, 2)),))
            elif family == "impossible":
                p = replace(p, budget=int(0.65 * min(completion)))
            result.append(p)
    return result
