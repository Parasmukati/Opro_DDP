# OPRO Network Design Optimization (maximize reliability under budget)
import random
import math
import time
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import requests
from api_key import GROQ_API_KEY
from collections import deque

# ------------------------
# Groq API config (unchanged)
# ------------------------
GROQ_API_KEY = GROQ_API_KEY
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5, timeout=30):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You must return ONLY a single Python fenced code block that assigns a variable named `design` containing a list of (u,v) edges to build. No prints, no other variables."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.12,
        "max_tokens": 2000
    }
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                wait_time = backoff * (2 ** attempt)
                print(f"Rate limit hit. Sleeping {wait_time}s...")
                time.sleep(wait_time)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("Groq API error:", e)
            time.sleep(backoff)
    return None

# ------------------------
# Problem generator
# ------------------------
def generate_network(num_nodes=12, edge_prob=0.25, cost_min=1, cost_max=8, rel_min=0.7, rel_max=0.99, seed=42):
    random.seed(seed)
    nodes = list(range(num_nodes))
    edges = []
    cost = {}
    reliability = {}
    for i in nodes:
        for j in nodes:
            if i < j and random.random() < edge_prob:
                edges.append((i,j))
                cost[(i,j)] = random.randint(cost_min, cost_max)
                # reliability probability that edge survives (independent)
                reliability[(i,j)] = round(random.uniform(rel_min, rel_max), 3)
    return nodes, edges, cost, reliability

def pick_terminals(nodes, k=3, seed=999):
    random.seed(seed)
    return random.sample(nodes, k)

# ------------------------
# Utility: connectivity check (undirected) given built edges and failed edges
# ------------------------
def is_terminals_connected(nodes, built_edges_set, failed_edges_set, terminals):
    # build adjacency for surviving edges
    adj = {n: [] for n in nodes}
    for (u,v) in built_edges_set:
        if (u,v) in failed_edges_set or (v,u) in failed_edges_set:
            continue
        adj[u].append(v)
        adj[v].append(u)
    # BFS from first terminal and check reachability of all terminals
    start = terminals[0]
    seen = {start}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                dq.append(nb)
    return all(t in seen for t in terminals)

# ------------------------
# Monte-Carlo Reliability Estimation
# ------------------------
def estimate_reliability(build_edges, reliability_map, nodes, terminals, mc_samples=2000, seed=None):
    """
    Estimate probability that all terminals are mutually connected (in same component)
    given build_edges (list/iterable of (u,v)), reliability_map keyed by (u,v) or (v,u).
    """
    if seed is not None:
        random.seed(seed)
    built_set = set(tuple(e) if e[0] <= e[1] else (e[1], e[0]) for e in build_edges)
    succ = 0
    # pre-map reliability for undirected canonical ordering
    rel = {}
    for (u,v),p in reliability_map.items():
        key = (u,v) if u <= v else (v,u)
        rel[key] = float(p)
    for _ in range(mc_samples):
        failed = set()
        for e in built_set:
            p = rel.get(e, 0.0)
            # success with prob p
            if random.random() > p:
                failed.add(e)
        if is_terminals_connected(nodes, built_set, failed, terminals):
            succ += 1
    return float(succ) / mc_samples

# ------------------------
# Greedy baseline (heuristic)
# Greedy: add edges by marginal estimated reliability gain per unit cost until budget exhausted
# (uses Monte Carlo marginal evals; small samples for speed)
# ------------------------
def greedy_baseline(nodes, edges, cost_map, rel_map, terminals, budget, mc_samples=800):
    remaining_edges = set(edges)
    chosen = set()
    remaining_budget = budget
    # Evaluate marginal gain roughly for each candidate at each step
    while True:
        best_gain = 0.0
        best_edge = None
        best_est_rel = None
        curr_rel = estimate_reliability(chosen, rel_map, nodes, terminals, mc_samples=mc_samples)
        for e in list(remaining_edges):
            c = cost_map[e]
            if c > remaining_budget:
                continue
            cand = set(chosen)
            cand.add(e)
            est = estimate_reliability(cand, rel_map, nodes, terminals, mc_samples=mc_samples//2)
            gain = est - curr_rel
            score = gain / (c + 1e-9)
            if score > best_gain:
                best_gain = score
                best_edge = e
                best_est_rel = est
        if best_edge is None:
            break
        chosen.add(best_edge)
        remaining_edges.remove(best_edge)
        remaining_budget -= cost_map[best_edge]
        # small stopping if negligible gain
        if best_gain < 1e-4:
            break
    final_rel = estimate_reliability(chosen, rel_map, nodes, terminals, mc_samples=mc_samples)
    return list(chosen), final_rel

# ------------------------
# LLM prompt builder
# ------------------------
def build_nd_prompt(nodes, edges, cost_map, rel_map, terminals, budget, history):
    py_edges = [ (int(u), int(v), int(cost_map[(u,v)]), float(rel_map[(u,v)])) for (u,v) in edges ]
    prompt = []
    prompt.append("Design a network under a budget constraint. Return ONLY a single Python fenced code block.")
    prompt.append("You must assign a variable named `design` containing a list of edges to build.")
    prompt.append("Edges are undirected tuples (u,v) where u < v.")
    prompt.append("")
    prompt.append(f"nodes = {nodes}")
    prompt.append(f"edges = {py_edges}  # (u, v, cost, reliability)")
    prompt.append(f"terminals = {terminals}  # all terminals must remain connected after random failures")
    prompt.append(f"budget = {budget}")
    prompt.append("")
    prompt.append("Important requirements:")
    prompt.append("- `design` must be a list of distinct (u,v) tuples (u < v).")
    prompt.append("- Total cost of selected edges must be <= budget.")
    prompt.append("- We evaluate reliability by independent per-edge survival with the reliability values above.")
    prompt.append("- Aim to maximize probability that all terminals remain connected (we estimate with Monte Carlo).")
    prompt.append("- Do NOT print, do NOT import, do not define functions. Final line must be: design = [(u,v), ...]")
    if history:
        prompt.append("")
        prompt.append("Recent best designs (cost, estimated reliability):")
        for d, (c, r) in history[:3]:
            prompt.append(f"- cost={c}, rel={r:.3f}, edges={d[:6]}...")
    prompt.append("")
    prompt.append("Example final line:")
    prompt.append("```python")
    prompt.append("design = [(0,3),(3,5),(1,2)]")
    prompt.append("```")
    return "\n".join(prompt)

# ------------------------
# Execute LLM output and extract design
# ------------------------
def run_llm_and_get_design(llm_output, nodes, edges, cost_map, rel_map, budget):
    if not llm_output:
        return None, None, False
    try:
        if "```python" in llm_output:
            code = llm_output.split("```python",1)[1].split("```",1)[0]
        elif "```" in llm_output:
            code = llm_output.split("```",1)[1].split("```",1)[0]
        else:
            code = llm_output
        code = textwrap.dedent(code).strip()
        safe_builtins = {
            "range": range, "len": len, "sum": sum, "max": max, "min": min,
            "enumerate": enumerate, "list": list, "int": int, "zip": zip,
            "tuple": tuple, "abs": abs, "all": all, "any": any,
            "sorted": sorted, "map": map, "filter": filter,
            "bool": bool, "float": float, "str": str, "set": set, "dict": dict,
            "bin": bin, "hex": hex, "oct": oct, "divmod": divmod, "round": round,
            "random": random,"__import__": __import__, "print": print,"__name__": __name__

        }
        exec_env = {"__builtins__": safe_builtins, "nodes": nodes, "edges": edges, "budget": budget}
        try:
            exec(code, exec_env)
        except Exception as e:
            print("LLM code execution error:", e)
            print("LLM Code (truncated):\n", "\n".join(code.splitlines()[:200]))
            return None, code, False
        if "design" not in exec_env:
            print("LLM did not set variable `design`. Code was:\n", code[:1000])
            return None, code, False
        design = exec_env["design"]
        # normalize and validate shape
        if not isinstance(design, list):
            return None, code, False
        norm = []
        seen = set()
        total_cost = 0
        for item in design:
            if not (isinstance(item, (list,tuple)) and len(item)==2):
                return None, code, False
            u = int(item[0]); v = int(item[1])
            if u == v:
                return None, code, False
            a,b = (u,v) if u < v else (v,u)
            if (a,b) not in edges:
                return None, code, False
            if (a,b) in seen:
                continue
            seen.add((a,b))
            total_cost += cost_map[(a,b)]
            norm.append((a,b))
        if total_cost > budget:
            return norm, code, False
        return norm, code, True
    except Exception as e:
        print("Parse error:", e)
        return None, None, False

# ------------------------
# OPRO loop
# ------------------------
def opro_network_design(nodes, edges, cost_map, rel_map, terminals, budget, iterations=8, batch=3, mc_samples_eval=1500):
    history = []  # tuples (design_list, (cost, est_rel))
    successful_codes = []
    # seed: greedy baseline (cheap-first and greedy marginal)
    print("Computing greedy baseline...")
    greedy_design, greedy_rel = greedy_baseline(nodes, edges, cost_map, rel_map, terminals, budget, mc_samples=800)
    greedy_cost = sum(cost_map[e] for e in greedy_design)
    history.append((greedy_design, (greedy_cost, greedy_rel)))
    print(f" Greedy baseline: cost={greedy_cost}, rel≈{greedy_rel:.3f}, edges={len(greedy_design)}")

    # seed 2: cheapest-first until budget
    cheap_sorted = sorted(edges, key=lambda e: cost_map[e])
    cheap_design = []
    rem = budget
    for e in cheap_sorted:
        if cost_map[e] <= rem:
            cheap_design.append(e)
            rem -= cost_map[e]
    cheap_rel = estimate_reliability(cheap_design, rel_map, nodes, terminals, mc_samples=600)
    cheap_cost = sum(cost_map[e] for e in cheap_design)
    history.append((cheap_design, (cheap_cost, cheap_rel)))
    print(f" Cheap-first baseline: cost={cheap_cost}, rel≈{cheap_rel:.3f}, edges={len(cheap_design)}")

    best_progress = []
    for it in range(iterations):
        print(f"\n=== Iteration {it+1}/{iterations} ===")
        prompt = build_nd_prompt(nodes, edges, cost_map, rel_map, terminals, budget, history)
        for b in range(batch):
            print(f" Requesting LLM (batch {b+1}/{batch})...")
            out = groq_generate(prompt)
            design, code, ok = run_llm_and_get_design(out, nodes, edges, cost_map, rel_map, budget)
            if design is None:
                print("  ✗ LLM produced unparsable design.")
                continue
            if not ok:
                print("  ✗ LLM produced design that violates budget or edges invalid. Code preview:")
                print(code[:400])
                continue
            # evaluate reliability with MC
            est = estimate_reliability(design, rel_map, nodes, terminals, mc_samples=mc_samples_eval, seed=it*100 + b)
            total_cost = sum(cost_map[e] for e in design)
            history.append((design, (total_cost, est)))
            successful_codes.append({"design": design, "cost": total_cost, "rel": est, "code": code})
            print(f"  ✓ Feasible design: cost={total_cost}, est_rel≈{est:.4f}, edges={len(design)}")
        # prune keep top few by rel
        history = sorted(history, key=lambda x: -x[1][1])[:20]
        best_progress.append(history[0][1][1])
    best_design, (best_cost, best_rel) = history[0]
    return (best_design, best_cost, best_rel), successful_codes, best_progress, (greedy_design, greedy_cost, greedy_rel), (cheap_design, cheap_cost, cheap_rel)

# ------------------------
# Plotting utility
# ------------------------
def plot_progression(prog, baseline_rel):
    plt.figure(figsize=(8,4))
    iters = list(range(1, len(prog)+1))
    plt.plot(iters, prog, marker='o', label='OPRO LLM best rel')
    plt.axhline(y=baseline_rel, linestyle='--', label=f'Greedy baseline rel ≈ {baseline_rel:.3f}')
    plt.xlabel('Iteration')
    plt.ylabel('Best estimated reliability')
    plt.title('OPRO Network Design progression vs Greedy baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------
# MAIN: run instance
# ------------------------
if __name__ == "__main__":
    # parameters (tune for speed/quality)
    NUM_NODES = 12
    EDGE_PROB = 0.28
    COST_MIN, COST_MAX = 1, 10
    REL_MIN, REL_MAX = 0.6, 0.98
    NUM_TERMINALS = 3
    BUDGET = 25
    SEED = 42

    nodes, edges, cost_map, rel_map = generate_network(NUM_NODES, EDGE_PROB, COST_MIN, COST_MAX, REL_MIN, REL_MAX, seed=SEED)
    terminals = pick_terminals(nodes, k=NUM_TERMINALS, seed=SEED+1)

    print("Nodes:", nodes)
    print("Terminals:", terminals)
    print("Edges (u,v) count:", len(edges))
    print("Example edges (u,v,cost,rel):")
    for e in edges[:12]:
        print(" ", e, "cost=", cost_map[e], "rel=", rel_map[e])

    # run OPRO network design
    (best_design, best_cost, best_rel), successful_codes, progress, greedy_info, cheap_info = opro_network_design(
        nodes, edges, cost_map, rel_map, terminals, BUDGET,
        iterations=6, batch=3, mc_samples_eval=1200
    )

    print("\n=== RESULTS ===")
    print(f"Greedy baseline: cost={greedy_info[1]}, rel≈{greedy_info[2]:.4f}, edges={len(greedy_info[0])}")
    print(f"Cheap-first baseline: cost={cheap_info[1]}, rel≈{cheap_info[2]:.4f}, edges={len(cheap_info[0])}")
    print(f"Best LLM design: cost={best_cost}, rel≈{best_rel:.4f}, edges={len(best_design)}")
    print("Top LLM solutions (best 3):")
    for ex in sorted(successful_codes, key=lambda x: -x["rel"])[:3]:
        print(f" cost={ex['cost']}, rel≈{ex['rel']:.4f}, edges={len(ex['design'])}")
        print(" Code preview:")
        print("-"*40)
        print(ex["code"][:800])
        print("-"*40)

    plot_progression(progress, greedy_info[2])
