import random
import time
import textwrap
import math
import matplotlib.pyplot as plt
import numpy as np
import requests
from ortools.sat.python import cp_model
from api_key import GROQ_API_KEY_2

# -------------------------
# Groq API config (unchanged)
# -------------------------
GROQ_API_KEY = GROQ_API_KEY_2
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5, timeout=30):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You generate ONLY Python code that assigns a variable named `flow_solution` (dict) and nothing else. No prints."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.15,
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

# -------------------------
# Problem generator
# -------------------------
def generate_random_graph(num_nodes=8, edge_prob=0.25, cap_min=5, cap_max=30, seed=42):
    random.seed(seed)
    nodes = list(range(num_nodes))
    edges = []
    capacities = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if random.random() < edge_prob:
                cap = random.randint(cap_min, cap_max)
                edges.append((i, j))
                capacities[(i,j)] = cap
    return nodes, edges, capacities

def generate_commodities(nodes, K=3, demand_min=5, demand_max=30, seed=42):
    random.seed(seed+7)
    commodities = []
    all_nodes = nodes[:]
    for k in range(K):
        # pick source != sink
        s = random.choice(all_nodes)
        t = random.choice(all_nodes)
        while t == s:
            t = random.choice(all_nodes)
        demand = random.randint(demand_min, demand_max)
        commodities.append({"id": k, "source": s, "sink": t, "demand": demand})
    return commodities

# -------------------------
# Validation & evaluation
# -------------------------
def evaluate_flow_solution(flow_solution, nodes, edges, capacities, commodities):
    """
    flow_solution expected as:
      { k: [ (u,v,flow), ... ] , ... }
    returns: (is_valid, total_routed, reason_or_details, per_commodity_routed)
    """
    # normalize to dict edges->sum over commodities
    edge_used = {e: 0 for e in edges}
    per_comm_flow = {c['id']: 0 for c in commodities}

    # per-commodity flow conservation map
    for c in commodities:
        k = c['id']
        if k not in flow_solution:
            # treated as zero flow
            continue
        rec = flow_solution[k]
        # accumulate flows leaving/entering nodes for conservation
        if not isinstance(rec, list):
            return False, 0, f"commodity {k} flow is not a list", None
        node_balance = {n: 0 for n in nodes}  # outflow - inflow
        for trip in rec:
            if not (isinstance(trip, (list,tuple)) and len(trip)==3):
                return False, 0, f"commodity {k} item malformed: {trip}", None
            u,v,f = trip
            if (u,v) not in edge_used:
                return False, 0, f"commodity {k} uses non-existing edge {(u,v)}", None
            try:
                f_val = float(f)
            except Exception:
                return False, 0, f"commodity {k} has non-numeric flow {f}", None
            if f_val < -1e-9:
                return False, 0, f"commodity {k} has negative flow on {(u,v)}", None
            edge_used[(u,v)] += f_val
            node_balance[u] += f_val
            node_balance[v] -= f_val
            per_comm_flow[k] += f_val
        # check conservation: for commodity k, flows sum balance should be source -> +flow out - in = routed
        # ideal: net outflow at source == net inflow at sink == routed amount; others net zero
        s = c['source']; t = c['sink']; demand = c['demand']
        net_source = node_balance[s]
        net_sink = -node_balance[t]  # sink receives inflow, node_balance[t] = out-in -> negative if inflow>out
        # allow some tolerance
        if abs(net_source - net_sink) > 1e-6:
            return False, 0, f"commodity {k} conservation mismatch (source net {net_source}, sink net {net_sink})", None
        # routed for commodity is net_source (may be <= demand)
        routed = net_source
        if routed < -1e-9:
            return False, 0, f"commodity {k} net routed negative {routed}", None
        # clamp small negatives
        per_comm_flow[k] = max(0.0, routed)

    # capacity check
    for e, used in edge_used.items():
        cap = capacities.get(e, 0)
        if used - cap > 1e-6:
            return False, 0, f"edge {e} exceeded capacity: used {used} > cap {cap}", None

    total_routed = sum(per_comm_flow.values())
    # also ensure per-commodity routed <= demand
    for c in commodities:
        k = c['id']
        if per_comm_flow[k] - c['demand'] > 1e-6:
            return False, 0, f"commodity {k} routed {per_comm_flow[k]} > demand {c['demand']}", None

    return True, float(total_routed), "feasible", per_comm_flow

# -------------------------
# CP-SAT baseline (exact small instances)
# -------------------------
def cp_sat_mcf(nodes, edges, capacities, commodities):
    """
    Integer CP-SAT model:
      f_k_e integer >=0 for each commodity k and edge e
      sum_k f_k_e <= cap_e
      flow conservation for each commodity: sum_out - sum_in = supply (supply= routed amount at source, negative at sink)
    We maximize total routed flow (sum of net outflow at sources). To keep linear, we introduce routed_k variable = net outflow at source (integer) and constrain accordingly.
    """
    model = cp_model.CpModel()
    # bounds: flows integer in [0, max_cap]
    max_cap = max(capacities.values()) if capacities else 0
    # create variables
    f = {}  # (k, u, v) -> var
    routed = {}
    for c in commodities:
        k = c['id']
        routed[k] = model.NewIntVar(0, c['demand'], f"routed_{k}")
        for (u,v) in edges:
            f[(k,u,v)] = model.NewIntVar(0, int(math.ceil(capacities[(u,v)])), f"f_{k}_{u}_{v}")

    # capacity constraints
    for (u,v) in edges:
        model.Add(sum(f[(k,u,v)] for k in routed.keys()) <= int(capacities[(u,v)]))

    # flow conservation and routed link
    for c in commodities:
        k = c['id']
        s = c['source']; t = c['sink']
        demand = c['demand']
        # flow conservation for every node
        for n in nodes:
            out_sum = sum(f[(k,u,v)] for (u,v) in edges if u == n)
            in_sum  = sum(f[(k,u,v)] for (u,v) in edges if v == n)
            if n == s:
                # net outflow from source = routed[k]
                model.Add(out_sum - in_sum == routed[k])
            elif n == t:
                # net outflow at sink should be -routed[k]
                model.Add(out_sum - in_sum == -routed[k])
            else:
                model.Add(out_sum - in_sum == 0)

    # objective maximize total routed
    model.Maximize(sum(routed[k] for k in routed.keys()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        flow_sol = {}
        per_comm = {}
        for c in commodities:
            k = c['id']
            per_comm[k] = solver.Value(routed[k])
            rec = []
            for (u,v) in edges:
                val = solver.Value(f[(k,u,v)])
                if val > 0:
                    rec.append((u,v, float(val)))
            flow_sol[k] = rec
        total = sum(per_comm.values())
        return flow_sol, float(total), per_comm
    else:
        return None, 0.0, {}

# -------------------------
# Build LLM prompt
# -------------------------
def build_mcf_prompt(nodes, edges, capacities, commodities, history):
    py_edges = [(int(u), int(v), int(capacities[(u,v)])) for (u,v) in edges]
    prompt = []
    prompt.append("Solve the MULTI-COMMODITY FLOW problem. You must output only Python code inside a single ```python fenced block.")
    prompt.append("You MUST assign a variable named `flow_solution` and nothing else. `flow_solution` must be a dict mapping commodity id -> list of (u,v,flow) tuples.")
    prompt.append("")
    prompt.append(f"nodes = {nodes}")
    prompt.append(f"edges = {py_edges}  # (u,v,capacity)")
    prompt.append(f"commodities = {[ (c['id'], int(c['source']), int(c['sink']), int(c['demand'])) for c in commodities ]}  # (id,source,sink,demand)")
    prompt.append("")
    prompt.append("Requirements:")
    prompt.append("- Each tuple in flow_solution[k] must be (u, v, flow) with numeric flow >= 0.")
    prompt.append("- You must only use the provided nodes and edges.")
    prompt.append("- Total flow on any edge (sum over commodities) must not exceed its capacity.")
    prompt.append("- Do NOT import external libs like ortools or pulp.")
    prompt.append("- For each commodity, the flow must satisfy flow conservation: net outflow at source = net inflow at sink = routed_amount; intermediate nodes net zero.")
    prompt.append("- Routed for each commodity must be <= demand.")
    prompt.append("- Do NOT print; do not define functions. Return straight-line code. Final line must set flow_solution variable.")
    if history:
        prompt.append("")
        prompt.append("Recent feasible solutions (top few):")
        for sol, total in history[:3]:
            prompt.append(f"- total={total:.1f}")
    prompt.append("")
    prompt.append("Example final line format (exact variable name):")
    prompt.append("```python")
    prompt.append("flow_solution = {0: [(0,1,10.0), (1,2,5.0)], 1: [(0,2,3.0)]}")
    prompt.append("```")
    return "\n".join(prompt)

# -------------------------
# Execute LLM output and extract flows
# -------------------------
def run_llm_code_and_get_flows(llm_output, nodes, edges, capacities, commodities):
    if not llm_output:
        return None, None, None, False
    try:
        # extract code fence
        if "```python" in llm_output:
            code = llm_output.split("```python",1)[1].split("```",1)[0]
        elif "```" in llm_output:
            code = llm_output.split("```",1)[1].split("```",1)[0]
        else:
            code = llm_output
        code = textwrap.dedent(code).strip()
        # safe exec environment
        safe_builtins = {
            "range": range, "len": len, "sum": sum, "max": max, "min": min,
            "enumerate": enumerate, "list": list, "int": int, "zip": zip,
            "tuple": tuple, "abs": abs, "all": all, "any": any,
            "sorted": sorted, "map": map, "filter": filter,
            "bool": bool, "float": float, "str": str, "set": set, "dict": dict,
            "bin": bin, "hex": hex, "oct": oct, "divmod": divmod, "round": round,
            "random": random,"__import__": __import__, "print": print,"__name__": __name__

        }
        exec_env = {
            "__builtins__": safe_builtins,
            "nodes": nodes,
            "edges": edges,
            "capacities": capacities,
            "commodities": commodities,
            # helpful aliases
            "math": math
        }
        try:
            exec(code, exec_env)
        except Exception as e:
            # show code snippet for debugging
            print("LLM code execution error:", e)
            print("LLM Code (truncated):\n", "\n".join(code.splitlines()[:200]))
            return None, code, None, False
        if "flow_solution" not in exec_env:
            print("LLM did not set 'flow_solution'. Code was:\n", code[:1000])
            return None, code, None, False
        sol = exec_env["flow_solution"]
        # normalize: ensure keys are commodity ids ints and values list of (u,v,flow)
        if not isinstance(sol, dict):
            print("flow_solution not dict")
            return None, code, None, False
        # quick shape checks & convert tuples/lists
        normalized = {}
        for k, rec in sol.items():
            try:
                kid = int(k)
            except Exception:
                return None, code, None, False
            if not isinstance(rec, list):
                return None, code, None, False
            triples = []
            for item in rec:
                if not (isinstance(item, (list,tuple)) and len(item)==3):
                    return None, code, None, False
                u = int(item[0]); v = int(item[1])
                f = float(item[2])
                triples.append((u,v,f))
            normalized[kid] = triples
        # validate
        valid, total, details, per_comm = evaluate_flow_solution(normalized, nodes, edges, capacities, commodities)
        return normalized, code, (valid, total, details, per_comm), valid
    except Exception as e:
        print("Parse error:", e)
        return None, None, None, False

# -------------------------
# OPRO main loop
# -------------------------
def opro_mcf(nodes, edges, capacities, commodities, iterations=8, batch=3):
    history = []  # list of (flow_solution, total)
    successful_codes = []
    # seed with trivial zero-flow
    history.append(( {c['id']: [] for c in commodities}, 0.0))
    # optionally add greedy seeds: route along single shortest path until capacity/demand exhausted (simple)
    # build adjacency
    adj = {n: [] for n in nodes}
    for (u,v) in edges:
        adj[u].append(v)
    # simple seed: route each commodity along direct edges if available
    seed = {}
    for c in commodities:
        k = c['id']; s = c['source']; t = c['sink']; d = c['demand']
        rec = []
        # if direct edge exists
        if (s,t) in capacities:
            flow = min(d, capacities[(s,t)])
            if flow > 0:
                rec.append((s,t,float(flow)))
        seed[k] = rec
    valid, total, _, _ = evaluate_flow_solution(seed, nodes, edges, capacities, commodities)
    if valid:
        history.append((seed, total))
        successful_codes.append({"routes": seed, "total": total, "code": "# seed direct"})
    # OPRO iterations
    best_progress = []
    for it in range(iterations):
        print(f"\n=== Iteration {it+1}/{iterations} ===")
        prompt = build_mcf_prompt(nodes, edges, capacities, commodities, history)
        for b in range(batch):
            print(f" Requesting LLM (batch {b+1}/{batch})...")
            out = groq_generate(prompt)
            sol_norm, code, info, ok = run_llm_code_and_get_flows(out, nodes, edges, capacities, commodities)
            if ok:
                valid_flag, total_routed, details, per_comm = info
                history.append((sol_norm, total_routed))
                successful_codes.append({"solution": sol_norm, "total": total_routed, "code": code})
                print(f"  ✓ Feasible LLM solution: total_routed={total_routed:.2f}")
            else:
                print("  ✗ LLM produced invalid or infeasible solution.")
        # prune keep top few
        history = sorted(history, key=lambda x: -x[1])[:20]
        best_progress.append(history[0][1])
    best_sol, best_total = history[0]
    return (best_sol, best_total), successful_codes, best_progress

# -------------------------
# Visualization utils
# -------------------------
def plot_progression(progression, baseline_total):
    plt.figure(figsize=(8,4))
    iters = list(range(1, len(progression)+1))
    plt.plot(iters, progression, marker='o', label='OPRO LLM best')
    plt.axhline(y=baseline_total, linestyle='--', label=f'CP-SAT baseline = {baseline_total:.1f}')
    plt.xlabel('Iteration')
    plt.ylabel('Best total routed flow')
    plt.title('OPRO LLM progression vs Baseline (CP-SAT)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------
# MAIN: run instance
# -------------------------
if __name__ == "__main__":
    # parameters
    NUM_NODES = 9
    EDGE_PROB = 0.35
    CAP_MIN, CAP_MAX = 8, 40
    K = 3
    SEED = 1234

    nodes, edges, capacities = generate_random_graph(NUM_NODES, EDGE_PROB, CAP_MIN, CAP_MAX, seed=SEED)
    commodities = generate_commodities(nodes, K=K, demand_min=5, demand_max=35, seed=SEED)

    print("Nodes:", nodes)
    print("Edges (u,v,cap):")
    for e in edges:
        print(f"  {e} cap={capacities[e]}")
    print("Commodities (id, source, sink, demand):")
    for c in commodities:
        print(f"  {c}")

    # Baseline exact solve (CP-SAT)
    print("\nSolving CP-SAT baseline (small instance)...")
    cp_solution, cp_total, cp_per_comm = cp_sat_mcf(nodes, edges, capacities, commodities)
    if cp_solution:
        print("Baseline total routed (CP-SAT):", cp_total)
        for k,v in cp_per_comm.items():
            print(f"  commodity {k} routed {v}")
    else:
        print("CP-SAT failed to find feasible solution (maybe instance infeasible)")

    # OPRO LLM-driven search
    print("\nRunning OPRO LLM search...")
    (best_sol, best_total), successful_codes, progress = opro_mcf(nodes, edges, capacities, commodities, iterations=6, batch=3)

    print("\n=== RESULTS ===")
    print("Baseline (CP-SAT) total:", cp_total)
    print("Best LLM total:", best_total)
    print("\nTop LLM examples (best 3):")
    for example in sorted(successful_codes, key=lambda x: -x["total"])[:3]:
        print(f"\nTotal={example['total']:.2f}")
        # print solution summary (per commodity routed)
        valid, total, details, per_comm = evaluate_flow_solution(example['solution'], nodes, edges, capacities, commodities)
        print(" Per-commodity routed:", per_comm)
        print(" Code preview:")
        print("-"*40)
        print(example["code"][:])
        print("-"*40)

    # plot progression (OPRO) vs baseline
    baseline_val = cp_total if cp_solution else 0.0
    plot_progression(progress, baseline_val)

    # show final flows on graph (text summary)
    print("\nFinal LLM best solution summary (per commodity):")
    valid, total, detail, per_comm = evaluate_flow_solution(best_sol, nodes, edges, capacities, commodities)
    print(" Valid:", valid, "Total:", total)
    for k in per_comm:
        print(f"  Commodity {k} routed {per_comm[k]} (demand {next(c['demand'] for c in commodities if c['id']==k)})")
