"""Episode simulator and offline diagnostics; policy interface has no true_z field."""

from hashlib import blake2b
from time import perf_counter

from .core import Policy


def uniform(*parts) -> float:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") / 2**64


def rollout(policy: Policy, seed: int) -> tuple[dict, list[dict]]:
    model, problem = policy.model, policy.model.problem
    state = model.initial()
    true_z = 0 if uniform(problem.task_id, seed, "truth") < float(state.belief[0]) else 1
    initially_feasible = model.margin(state, true_z) >= 0
    lost = False
    energy = questions = observations = 0
    trace = []
    for step in range(problem.horizon + 1):
        if initially_feasible and model.margin(state, true_z) < 0:
            lost = True
        started = perf_counter()
        action = policy.choose(state)
        call_ms = (perf_counter() - started) * 1000
        if action not in model.actions(state):
            raise AssertionError("Illegal policy action")
        cost = model.cost(state, action)
        entry = {
            "step": step, "node": state.node, "budget": state.budget,
            "belief": [str(p) for p in state.belief], "steps_left": state.steps,
            "seen": state.seen, "asked": state.asked, "action": action.label(),
            "cost": cost, "policy_call_ms_cached": call_ms,
            "candidate_margins": [model.margin(state, z) for z in (0, 1)],
        }
        energy += cost
        questions += action.kind == "ask"
        observations += action.kind == "observe"
        if action.kind in {"finish", "abort"}:
            success = (action.kind == "finish" and action.target in problem.viewpoints[true_z])
            if action.kind == "finish":
                route = model.paths[state.node][action.target]
                budget = state.budget
                approach = []
                for a, b in zip(route, route[1:]):
                    budget -= next(c for n, c in model.adj[a] if n == b)
                    margin = budget - model.completion_cost(b, true_z)
                    lost |= initially_feasible and margin < 0
                    approach.append({"node": b, "budget": budget, "true_margin_offline": margin})
                entry["approach_diagnostic"] = approach
                entry["return_path"] = model.paths[action.target][problem.home]
            entry["budget_after"] = state.budget - cost
            entry["returned"] = True
            trace.append(entry)
            if energy > problem.budget:
                raise AssertionError("Episode exceeded resource budget")
            return {
                "seed": seed, "private_true_z": true_z, "success": int(success),
                "energy": energy, "questions": questions, "observations": observations,
                "abort": int(action.kind == "abort"), "returned": 1,
                "initial_true_feasible": int(initially_feasible),
                "pre_capture_lost_recoverability": int(lost) if initially_feasible else "",
                "steps": step + 1,
            }, trace
        branches = model.branches(state, action)
        outcome = None
        if action.kind == "ask":
            outcome = true_z  # Only this authorized information action reveals it.
        elif action.kind == "observe":
            probability_zero = float(model.observations[state.node][true_z])
            outcome = 0 if uniform(problem.task_id, seed, "observe", state.node) < probability_zero else 1
        state = next(child for _, result, child in branches if result == outcome)
        if state.budget < model.distance(state.node, problem.home):
            raise AssertionError("Return reserve violated")
        entry["observation"] = outcome
        entry["budget_after"] = state.budget
        trace.append(entry)
    raise AssertionError("Finite-horizon episode did not terminate")
