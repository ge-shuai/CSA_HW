"""E5 sensitivity check with random sets matched on observed in-degree."""

from collections import Counter
from pathlib import Path
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent.parent))

import csa_common as C

STRATA = ("0", "1", "2–3", "4–7", "8+")
N_RANDOM = 300
SEED = 20261124


def in_degree_stratum(degree):
    return STRATA[sum(degree >= cut for cut in (1, 2, 4, 8))]


def fixed_efficiency(adjacency, kept, denominator):
    distances = shortest_path(
        adjacency[kept][:, kept], method="D", directed=False, unweighted=True
    )
    valid = np.isfinite(distances) & (distances > 0)
    return float(np.sum(1.0 / distances[valid]) / denominator)


def main():
    data = C.build_all()
    C.assert_anchors(data)
    graph = C.induced(data["Gu"], data["sets"]["gcc"])
    nodes = sorted(graph)
    index = {node: i for i, node in enumerate(nodes)}
    directed = data["Gd"].subgraph(nodes)
    strata = [in_degree_stratum(directed.in_degree(node)) for node in nodes]
    pools = {
        s: [i for i, label in enumerate(strata) if label == s] for s in STRATA
    }
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=nodes, weight=None, dtype=np.int32, format="csr"
    )
    pair_set = data["ps"]
    prior = pd.read_csv(OUT / "scenario_metrics.csv")
    prior = prior.loc[prior["view"] == "as_observed"].set_index("scenario")
    rng = random.Random(SEED)
    summaries = []
    counts = []
    draws = []

    def measure(removed):
        kept = np.ones(len(nodes), dtype=bool)
        kept[removed] = False
        return fixed_efficiency(adjacency, kept, pair_set.denom)

    for scenario, key in (("dead162", "dead_all"), ("true129", "true_fail")):
        observed = sorted(index[node] for node in data["fail"][key] if node in index)
        n_removed = len(observed)
        assert n_removed == prior.loc[scenario, "k_removed"]
        observed_value = measure(observed)
        removed_nodes = {nodes[i] for i in observed}
        reference = C.eff_fixed(graph.subgraph(set(nodes) - removed_nodes), pair_set)
        assert np.isclose(observed_value, reference, atol=1e-10, rtol=0)
        assert np.isclose(observed_value, prior.loc[scenario, "E_fixed"], atol=1e-6, rtol=0)

        quotas = Counter(strata[i] for i in observed)
        counts.extend(
            {"scenario": scenario, "observed_in_degree": s,
             "pool_n": len(pools[s]), "removed_n": quotas[s]} for s in STRATA
        )
        matched = []
        for rep in range(N_RANDOM):
            removed = [i for s in STRATA for i in rng.sample(pools[s], quotas[s])]
            value = measure(removed)
            matched.append(value)
            draws.append({"scenario": scenario, "draw": rep + 1, "E_fixed": value})

        mean = float(np.mean(matched))
        lo, hi = np.quantile(matched, [0.025, 0.975])
        uniform = prior.loc[scenario, "rand_mean_E_fixed"]
        summaries.append({
            "scenario": scenario, "removed_n": n_removed, "matched_draws": N_RANDOM,
            "seed": SEED, "observed_E_fixed": observed_value,
            "uniform_mean_E_fixed": uniform, "matched_mean_E_fixed": mean,
            "matched_p025": lo, "matched_p975": hi,
            "observed_minus_uniform": observed_value - uniform,
            "observed_minus_matched": observed_value - mean,
            "matched_at_least_observed": sum(x >= observed_value - 1e-12 for x in matched),
        })

    for name, rows in (
        ("matched_indegree_summary", summaries),
        ("matched_indegree_strata", counts),
        ("matched_indegree_draws", draws),
    ):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.9f")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
