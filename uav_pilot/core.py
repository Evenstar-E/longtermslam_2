"""Public finite belief model and policies. No episode ground truth enters this module.

All paths are known collision-free. ``finish`` is a deterministic macro: shortest
path to ONE accepted viewpoint, one photograph, then shortest path home. Moving
and gathering information before committing are allowed. There is no feedback
after photography. Exactness is restricted to this action model and horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from heapq import heappop, heappush
from math import inf, isfinite
from time import perf_counter


@dataclass(frozen=True)
class State:
    node: str
    budget: int
    steps: int
    belief: tuple[Fraction, ...]
    seen: tuple[str, ...] = ()
    asked: bool = False


@dataclass(frozen=True)
class Action:
    kind: str
    target: str = ""

    def label(self) -> str:
        return self.kind + (":" + self.target if self.target else "")


@dataclass(frozen=True)
class Problem:
    task_id: str
    layout_id: str
    family: str
    split: str
    edges: tuple[tuple[str, str, int], ...]
    home: str
    start: str
    budget: int
    horizon: int
    prior: tuple[Fraction, ...]
    viewpoints: tuple[tuple[str, ...], ...]
    # Each row: node, P(observation=0 | z=0), P(observation=0 | z=1).
    observation_rows: tuple[tuple[str, Fraction, Fraction], ...]
    ask_cost: int = 6
    observe_cost: int = 3
    photo_cost: int = 5

    def to_dict(self) -> dict:
        return {
            **{k: getattr(self, k) for k in (
                "task_id", "layout_id", "family", "split", "edges", "home", "start",
                "budget", "horizon", "viewpoints", "ask_cost", "observe_cost", "photo_cost"
            )},
            "prior": [str(p) for p in self.prior],
            "observation_rows": [[n, str(a), str(b)] for n, a, b in self.observation_rows],
        }


@dataclass(frozen=True)
class Objective:
    failure_cost: float = 250.0
    energy_weight: float = 1.0
    question_penalty: float = 0.0

    def __post_init__(self):
        if not all(isfinite(x) and x >= 0 for x in (
            self.failure_cost, self.energy_weight, self.question_penalty
        )):
            raise ValueError("Objective coefficients must be finite and nonnegative")

    def value(self, metrics: Metrics) -> float:
        return (self.failure_cost * (1 - metrics.success)
                + self.energy_weight * metrics.energy
                + self.question_penalty * metrics.questions)


@dataclass(frozen=True)
class Metrics:
    success: float = 0.0
    energy: float = 0.0
    questions: float = 0.0
    observations: float = 0.0
    aborts: float = 0.0


class Model:
    def __init__(self, problem: Problem):
        self.problem = problem
        if len(problem.prior) != 2 or len(problem.viewpoints) != 2:
            raise ValueError("Pilot v1 supports exactly two symbolic hypotheses")
        if sum(problem.prior) != 1 or any(p < 0 for p in problem.prior):
            raise ValueError("Prior must be a probability distribution")
        if problem.horizon < 1 or problem.budget < 0:
            raise ValueError("Invalid horizon or budget")
        if any(c < 0 or not isinstance(c, int) for c in (
            problem.ask_cost, problem.observe_cost, problem.photo_cost
        )):
            raise ValueError("Resource costs must be nonnegative integers")
        self.adj: dict[str, list[tuple[str, int]]] = {}
        for a, b, c in problem.edges:
            if c <= 0 or not isinstance(c, int):
                raise ValueError("Movement costs must be positive integers")
            self.adj.setdefault(a, []).append((b, c))
            self.adj.setdefault(b, []).append((a, c))
        self.nodes = tuple(sorted(self.adj))
        self.views = tuple(sorted(set().union(*map(set, problem.viewpoints))))
        if not set(self.views + (problem.home, problem.start)).issubset(self.adj):
            raise ValueError("Unknown home/start/viewpoint")
        self.observations = {n: (p0, p1) for n, p0, p1 in problem.observation_rows}
        if len(self.observations) != len(problem.observation_rows):
            raise ValueError("Duplicate observation node")
        for node, probs in self.observations.items():
            if node not in self.adj or any(p < 0 or p > 1 for p in probs):
                raise ValueError("Invalid observation likelihood")
        self.distances = {}
        self.paths = {}
        for source in self.nodes:
            distances = {source: 0}
            paths = {source: (source,)}
            heap = [(0, source)]
            while heap:
                cost, node = heappop(heap)
                if cost != distances[node]:
                    continue
                for other, weight in sorted(self.adj[node]):
                    proposed = cost + weight
                    if proposed < distances.get(other, inf):
                        distances[other] = proposed
                        paths[other] = paths[node] + (other,)
                        heappush(heap, (proposed, other))
            if len(distances) != len(self.nodes):
                raise ValueError("Pilot graph must be connected")
            self.distances[source] = distances
            self.paths[source] = paths
        if self.distance(problem.start, problem.home) > problem.budget:
            raise ValueError("Initial state must permit safe return")

    def initial(self) -> State:
        p = self.problem
        return State(p.start, p.budget, p.horizon, p.prior)

    def distance(self, a: str, b: str) -> int:
        return self.distances[a][b]

    def finish_cost(self, node: str, view: str) -> int:
        return (self.distance(node, view) + self.problem.photo_cost
                + self.distance(view, self.problem.home))

    def completion_cost(self, node: str, z: int) -> int:
        return min(self.finish_cost(node, v) for v in self.problem.viewpoints[z])

    def margin(self, state: State, z: int) -> int:
        return state.budget - self.completion_cost(state.node, z)

    def actions(self, state: State) -> tuple[Action, ...]:
        abort = Action("abort")
        if state.steps <= 0:
            return (abort,)
        actions = [Action("finish", v) for v in self.views
                   if self.finish_cost(state.node, v) <= state.budget]
        uncertain = sum(p > 0 for p in state.belief) > 1
        reserve = self.distance(state.node, self.problem.home)
        if not state.asked and uncertain and self.problem.ask_cost + reserve <= state.budget:
            actions.append(Action("ask"))
        if (state.node in self.observations and state.node not in state.seen
                and self.problem.observe_cost + reserve <= state.budget):
            actions.append(Action("observe"))
        for node, cost in sorted(self.adj[state.node]):
            if cost + self.distance(node, self.problem.home) <= state.budget:
                actions.append(Action("move", node))
        return tuple(actions + [abort])

    def cost(self, state: State, action: Action) -> int:
        if action.kind == "finish":
            return self.finish_cost(state.node, action.target)
        if action.kind == "abort":
            return self.distance(state.node, self.problem.home)
        if action.kind == "move":
            return next(c for node, c in self.adj[state.node] if node == action.target)
        return self.problem.ask_cost if action.kind == "ask" else self.problem.observe_cost

    def branches(self, state: State, action: Action) -> tuple[tuple[Fraction, int | None, State], ...]:
        if action.kind in {"finish", "abort"}:
            return ()
        next_state = replace(state, budget=state.budget - self.cost(state, action),
                             steps=state.steps - 1)
        if action.kind == "move":
            return ((Fraction(1), None, replace(next_state, node=action.target)),)
        likelihood = (Fraction(1), Fraction(0)) if action.kind == "ask" else self.observations[state.node]
        result = []
        for outcome in (0, 1):
            probs = likelihood if outcome == 0 else tuple(1 - p for p in likelihood)
            weights = tuple(p * q for p, q in zip(state.belief, probs))
            probability = sum(weights)
            if probability == 0:
                continue
            posterior = tuple(w / probability for w in weights)
            child = replace(next_state, belief=posterior,
                            asked=state.asked or action.kind == "ask",
                            seen=tuple(sorted(state.seen + (state.node,)))
                            if action.kind == "observe" else state.seen)
            result.append((probability, outcome, child))
        return tuple(result)

    def terminal_metrics(self, state: State, action: Action) -> Metrics:
        success = sum(p for z, p in enumerate(state.belief)
                      if action.kind == "finish" and action.target in self.problem.viewpoints[z])
        return Metrics(float(success), self.cost(state, action), aborts=float(action.kind == "abort"))


class Policy:
    name = "base"

    def __init__(self, model: Model, objective: Objective):
        self.model, self.objective = model, objective

    def choose(self, state: State) -> Action:
        raise NotImplementedError


class MAPPolicy(Policy):
    name = "map"

    def choose(self, state: State) -> Action:
        z = max(range(2), key=lambda i: state.belief[i])
        candidates = [a for a in self.model.actions(state)
                      if a.kind == "finish" and a.target in self.model.problem.viewpoints[z]]
        return min(candidates, key=lambda a: (self.model.cost(state, a), a.target)) if candidates else Action("abort")


class AskFirstPolicy(MAPPolicy):
    name = "ask_first"

    def choose(self, state: State) -> Action:
        support = [z for z, p in enumerate(state.belief) if p > 0]
        shared = set.intersection(*(set(self.model.problem.viewpoints[z]) for z in support))
        actions = self.model.actions(state)
        if not any(a.kind == "finish" for a in actions):
            return Action("abort")
        common = [a for a in actions if a.kind == "finish" and a.target in shared]
        if common:
            return min(common, key=lambda a: self.model.cost(state, a))
        if Action("ask") in actions:
            return Action("ask")
        return super().choose(state)


def action_metrics(model: Model, state: State, action: Action, continuation) -> Metrics:
    if action.kind in {"finish", "abort"}:
        return model.terminal_metrics(state, action)
    children = [(float(p), continuation(child)) for p, _, child in model.branches(state, action)]
    return Metrics(
        success=sum(p * m.success for p, m in children),
        energy=model.cost(state, action) + sum(p * m.energy for p, m in children),
        questions=float(action.kind == "ask") + sum(p * m.questions for p, m in children),
        observations=float(action.kind == "observe") + sum(p * m.observations for p, m in children),
        aborts=sum(p * m.aborts for p, m in children),
    )


class MyopicVOIPolicy(Policy):
    """One legal action then MAP completion, with task utility and all resource costs."""
    name = "task_voi"

    def __init__(self, model, objective):
        super().__init__(model, objective)
        self.fallback = MAPPolicy(model, objective)

    def choose(self, state):
        def finish(child):
            return self.model.terminal_metrics(child, self.fallback.choose(child))
        return min(self.model.actions(state), key=lambda a: self.objective.value(
            action_metrics(self.model, state, a, finish)))


class ExactPolicy(Policy):
    """Enumerate all public actions/observations using finite-horizon dynamic programming.

    Rational beliefs avoid approximate belief-state merging. Utilities use floats.
    Cache is per instance, avoiding retention of every earlier task's model.
    """
    name = "exact"

    def __init__(self, model, objective, timeout_s=30.0):
        super().__init__(model, objective)
        self.deadline = perf_counter() + timeout_s
        self.expansions = 0
        self.solve = lru_cache(maxsize=None)(self._solve)

    def _solve(self, state):
        self.expansions += 1
        if self.expansions % 128 == 1 and perf_counter() > self.deadline:
            raise TimeoutError("Exact policy exceeded per-task solve deadline")
        best = None
        for action in self.model.actions(state):
            metrics = action_metrics(self.model, state, action, lambda s: self.solve(s)[1])
            value = self.objective.value(metrics)
            if best is None or value < best[0] - 1e-10:
                best = (value, metrics, action)
        return best

    def choose(self, state):
        return self.solve(state)[2]

    def close(self):
        self.solve.cache_clear()


class DepthLimitedPolicy(Policy):
    """Generic finite lookahead, MAP leaf value, and receding-horizon execution.

    Depth counts legal model actions, including information acquisition. This is
    a baseline, not a reference-specific algorithm. Predicted truncated values
    are NOT used as the reported executed-policy metrics.
    """
    name = "lookahead"
    depth = 2

    def __init__(self, model, objective, timeout_s=30.0, depth=None):
        super().__init__(model, objective)
        self.depth = self.depth if depth is None else depth
        if self.depth < 1:
            raise ValueError("Lookahead depth must be at least one")
        self.fallback = MAPPolicy(model, objective)
        self.deadline = perf_counter() + timeout_s
        self.expansions = 0
        self.solve = lru_cache(maxsize=None)(self._solve)

    def _solve(self, state, depth):
        self.expansions += 1
        if self.expansions % 128 == 1 and perf_counter() > self.deadline:
            raise TimeoutError("Lookahead policy exceeded per-task solve deadline")
        if depth == 0 or state.steps <= 0:
            action = self.fallback.choose(state)
            metrics = self.model.terminal_metrics(state, action)
            return self.objective.value(metrics), metrics, action
        best = None
        for action in self.model.actions(state):
            metrics = action_metrics(self.model, state, action,
                                     lambda child: self.solve(child, depth - 1)[1])
            value = self.objective.value(metrics)
            if best is None or value < best[0] - 1e-10:
                best = value, metrics, action
        return best

    def choose(self, state):
        return self.solve(state, min(self.depth, state.steps))[2]

    def close(self):
        self.solve.cache_clear()


class Lookahead2Policy(DepthLimitedPolicy):
    name = "lookahead_2"
    depth = 2


class Lookahead3Policy(DepthLimitedPolicy):
    name = "lookahead_3"
    depth = 3


def expected_metrics(policy: Policy, state: State | None = None) -> Metrics:
    """Enumerate a policy's outcomes, not Monte Carlo; no episode truth supplied."""
    @lru_cache(maxsize=None)
    def evaluate(current):
        action = policy.choose(current)
        if action not in policy.model.actions(current):
            raise ValueError("Policy returned an invalid action")
        return action_metrics(policy.model, current, action, evaluate)
    return evaluate(state or policy.model.initial())


SEARCH_POLICIES = (ExactPolicy, DepthLimitedPolicy)
POLICIES = {p.name: p for p in (
    MAPPolicy, AskFirstPolicy, MyopicVOIPolicy, Lookahead2Policy, Lookahead3Policy, ExactPolicy
)}
