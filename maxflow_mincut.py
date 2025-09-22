# ============================
# OPRO - Maximum Flow / Minimum Cut
# ============================
import numpy as np
import matplotlib.pyplot as plt
import requests, time, textwrap, random
from collections import deque
from api_key import GROQ_API_KEY_1

# -----------------
# Groq API Config (unchanged pattern)
# -----------------
GROQ_API_KEY = GROQ_API_KEY_1
GROQ_MODEL="openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5, timeout=30):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role":"system","content":"You generate ONLY Python code for graph flow problems. Return direct code only, no explanation."},
            {"role":"user","content":prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 900
    }
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                wait_time = backoff * (2**attempt)
                print(f"Rate limit hit. Sleeping {wait_time}s...")
                time.sleep(wait_time)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("Groq API error:", e)
            time.sleep(backoff)
    return None

# -----------------
# Problem instance generator (directed graph)
# -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_NODES = 8    # including source(0) and sink(n-1)
SOURCE = 0
SINK = NUM_NODES - 1

def generate_random_graph(n, density=0.25, cap_low=1, cap_high=20):
    # adjacency capacity matrix
    cap = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u==v: continue
            # no edges into source or out of sink typically, but allow random connectivity
            if random.random() < density:
                cap[u][v] = int(np.random.randint(cap_low, cap_high+1))
    # ensure at least some connectivity from source and to sink
    for _ in range(max(1, n//3)):
        a = random.randint(1, n-2)
        b = random.randint(1, n-2)
        if a != b:
            cap[SOURCE][a] = max(cap[SOURCE][a], random.randint(cap_low, cap_high))
            cap[b][SINK] = max(cap[b][SINK], random.randint(cap_low, cap_high))
    return cap

CAP_MATRIX = generate_random_graph(NUM_NODES, density=0.25)

print("Capacity matrix (rows = from, cols = to):")
for i,row in enumerate(CAP_MATRIX):
    print(i, row)

# -----------------
# Baseline: Edmonds-Karp (compute max-flow and min-cut)
# -----------------
def edmonds_karp(cap, source, sink):
    n = len(cap)
    flow = [[0]*n for _ in range(n)]
    total_flow = 0
    while True:
        # BFS to find augmenting path in residual graph
        parent = [-1]*n
        parent[source] = source
        q = deque([source])
        while q and parent[sink] == -1:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and cap[u][v] - flow[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        if parent[sink] == -1:
            break  # no augmenting path
        # find bottleneck
        v = sink
        bottleneck = float('inf')
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, cap[u][v] - flow[u][v])
            v = u
        # augment
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += bottleneck
            flow[v][u] -= bottleneck
            v = u
        total_flow += bottleneck
    # compute min-cut (reachable from source in residual)
    visited = [False]*n
    q = deque([source]); visited[source]=True
    while q:
        u = q.popleft()
        for v in range(n):
            if not visited[v] and cap[u][v] - flow[u][v] > 0:
                visited[v] = True
                q.append(v)
    cut_edges = []
    for u in range(n):
        for v in range(n):
            if visited[u] and not visited[v] and cap[u][v] > 0:
                cut_edges.append((u,v,cap[u][v]))
    return flow, int(total_flow), cut_edges

baseline_flow, baseline_maxflow, baseline_cut = edmonds_karp(CAP_MATRIX, SOURCE, SINK)
print("\nBaseline (Edmonds-Karp) max-flow:", baseline_maxflow)
print("Baseline min-cut edges:", baseline_cut)

# -----------------
# Validate LLM-returned flow
# -----------------
def normalize_flow_candidate(candidate, n):
    """
    Accept formats:
     - list of triples [(u,v,flow), ...]
     - dict {(u,v): flow, ...} with tuple keys or "u,v" keys
     - adjacency matrix (list of lists)
    Return normalized list of triples.
    """
    if candidate is None: return None
    # adjacency matrix
    if isinstance(candidate, list) and len(candidate)==n and all(isinstance(row, list) for row in candidate):
        triples = []
        for u in range(n):
            for v in range(n):
                f = candidate[u][v]
                try:
                    if float(f) != 0:
                        triples.append((int(u),int(v),int(f)))
                except Exception:
                    pass
        return triples
    # list of triples
    if isinstance(candidate, list):
        ok = True
        triples = []
        for item in candidate:
            if not (isinstance(item, (list,tuple)) and len(item)==3):
                ok = False; break
            u,v,f = item
            try:
                triples.append((int(u),int(v),int(f)))
            except Exception:
                ok = False; break
        if ok:
            return triples
    # dict form
    if isinstance(candidate, dict):
        triples = []
        for k,v in candidate.items():
            if isinstance(k, tuple) and len(k)==2:
                u,vk = k; f = candidate[k]; 
                try:
                    triples.append((int(u),int(vk),int(f)))
                except Exception:
                    return None
            elif isinstance(k, str) and ',' in k:
                a,b = k.split(',')
                try:
                    triples.append((int(a.strip()), int(b.strip()), int(candidate[k])))
                except:
                    return None
            else:
                return None
        return triples
    return None

def validate_flow_solution(triples, cap, source, sink):
    """
    triples: list of (u,v,flow)
    checks:
      - non-negative flows, flow <= capacity
      - flow conservation at intermediate nodes
      - compute total flow leaving source (and entering sink)
    Returns (is_valid, total_flow, message)
    """
    n = len(cap)
    # build flow matrix
    flow = [[0]*n for _ in range(n)]
    for u,v,f in triples:
        if not (0 <= u < n and 0 <= v < n):
            return False, None, f"edge ({u},{v}) out of bounds"
        if f < 0:
            return False, None, f"negative flow on ({u},{v})"
        if f > cap[u][v]:
            return False, None, f"flow > capacity on ({u},{v})"
        flow[u][v] += f  # allow multiple triples aggregated
    # conservation
    for node in range(n):
        if node==source or node==sink: continue
        inflow = sum(flow[u][node] for u in range(n))
        outflow = sum(flow[node][v] for v in range(n))
        if inflow != outflow:
            return False, None, f"flow conservation violated at node {node} (in {inflow} != out {outflow})"
    total_out = sum(flow[source][v] for v in range(n))
    total_in_sink = sum(flow[u][sink] for u in range(n))
    if total_out != total_in_sink:
        # allow equality check; if not equal it's invalid
        return False, None, f"source out {total_out} != sink in {total_in_sink}"
    return True, int(total_out), "ok"

# -----------------
# Prompt Builder (strict)
# -----------------
def build_flow_prompt(cap_matrix, source, sink, history):
    n = len(cap_matrix)
    lines = []
    lines.append("Solve this MAXIMUM FLOW problem (directed graph).")
    lines.append(f"n = {n}  # nodes labeled 0..{n-1}")
    lines.append(f"source = {source}; sink = {sink}")
    lines.append("capacity = " + str(cap_matrix))
    lines.append("")
    lines.append("STRICT requirements:")
    lines.append("- Return ONLY Python code (single python fenced block).")
    lines.append("- Do NOT import external libs. Do NOT print.")
    lines.append("- At the END define two variables exactly:")
    lines.append("    flow_solution = [(u,v,flow), ...]  # list of integer triples")
    lines.append("    max_flow = <integer>                 # integer total flow value")
    lines.append("- Flow must respect capacities and conservation (intermediate nodes in==out).")
    lines.append("- The solution must be exact (maximize total flow). Use brute-force, flow augmenting, DFS, Edmonds-Karp style reconstruction or exact search.")
    if history:
        lines.append("\nRecent attempts (flow values):")
        for _,v in history[:4]:
            lines.append(f"- {v}")
    lines.append("\nReturn only the fenced python block.")
    return "\n".join(lines)

# -----------------
# Execute LLM code safely and extract flow
# -----------------
def run_llm_code_and_get_flow(llm_output, cap_matrix, source, sink):
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
        # safe builtins
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
            "n": len(cap_matrix),
            "capacity": cap_matrix,
            "source": source,
            "sink": sink
        }
        # execute
        try:
            exec(code, exec_env)
        except Exception as e:
            print("LLM code execution error:", e)
            print("LLM Code (trunc):\n", "\n".join(code.splitlines()[:200]))
            return None, code, False
        # extract solution variables
        candidate = None
        candidate_flow_val = None
        if "flow_solution" in exec_env:
            candidate = exec_env["flow_solution"]
            candidate_flow_val = exec_env.get("max_flow", None)
        elif "flow" in exec_env:
            candidate = exec_env["flow"]
            candidate_flow_val = exec_env.get("max_flow", None)
        elif "solution" in exec_env:
            candidate = exec_env["solution"]
            candidate_flow_val = exec_env.get("max_flow", None)
        if candidate is None:
            print("LLM did not set flow_solution variable. Code was:\n", code[:800])
            return None, code, False
        triples = normalize_flow_candidate(candidate, len(cap_matrix))
        if triples is None:
            print("Could not normalize candidate flow. Candidate type:", type(candidate))
            return None, code, False
        valid, computed_flow, msg = validate_flow_solution(triples, cap_matrix, source, sink)
        if not valid:
            print("LLM solution invalid:", msg)
            return triples, code, False
        # If the LLM reported max_flow, ensure it matches computed
        if candidate_flow_val is not None and int(candidate_flow_val) != int(computed_flow):
            print("LLM reported max_flow mismatch:", candidate_flow_val, "computed:", computed_flow)
            # still accept computed value (mark as not perfect if mismatch)
            return triples, code, False
        return triples, code, int(computed_flow)
    except Exception as e:
        print("Parse error:", e)
        return None, None, False

# -----------------
# OPRO loop for max-flow
# -----------------
ITERATIONS = 6
LLM_BATCH_SIZE = 3

def opro_maxflow(cap_matrix, source, sink, iterations, batch):
    history = []   # (triples, flow_value)
    successes = []
    for it in range(iterations):
        print(f"\n=== Iteration {it+1}/{iterations} ===")
        prompt = build_flow_prompt(cap_matrix, source, sink, history)
        for b in range(batch):
            print(f" Request LLM (batch {b+1}/{batch})...")
            llm_out = groq_generate(prompt)
            triples, code, mk = run_llm_code_and_get_flow(llm_out, cap_matrix, source, sink)
            if mk and mk>0:
                history.append((triples, mk))
                successes.append({"triples":triples, "flow":mk, "code":code})
                print(f"  ✓ Valid LLM flow found: flow={mk}")
            else:
                print("  ✗ Invalid or no valid LLM flow")
        # keep top attempts
        history = sorted(history, key=lambda x: -x[1])[:20]
    best = max(history, key=lambda x:x[1]) if history else (None, None)
    return best, successes

# -----------------
# Run experiment
# -----------------
if __name__=="__main__":
    best_llm, successful_attempts = opro_maxflow(CAP_MATRIX, SOURCE, SINK, ITERATIONS, LLM_BATCH_SIZE)
    print("\n--- Baseline (Edmonds-Karp) ---")
    print("Max flow:", baseline_maxflow)
    print("Min-cut edges:", baseline_cut)

    if best_llm and best_llm[1] is not None:
        print("\n--- Best LLM (OPRO) ---")
        print("LLM reported flow value:", best_llm[1])
        print("LLM flow triples (sample up to 30):", best_llm[0][:30])
        for attempt in successful_attempts:
            if attempt["flow"] == best_llm[1]:
                print("Code that produced best flow:\n", attempt["code"])
                break
    else:
        print("\nNo feasible LLM solution was found by OPRO.")

    # Numerical comparison plot
    llm_val = best_llm[1] if best_llm and best_llm[1] is not None else 0
    base_val = baseline_maxflow
    plt.figure(figsize=(6,4))
    plt.bar(['Baseline (Edmonds-Karp)', 'OPRO LLM (best)'], [base_val, llm_val])
    plt.ylabel("Max Flow value")
    plt.title("Max Flow: Baseline vs OPRO LLM (best)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # Visualize baseline flows (arrowed lines) and min-cut (red X)
    # simple layout: nodes on circle
    n = len(CAP_MATRIX)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    coords = [(50 + 40*np.cos(a), 50 + 40*np.sin(a)) for a in angles]

    plt.figure(figsize=(7,7))
    for u in range(n):
        for v in range(n):
            if CAP_MATRIX[u][v] > 0:
                xu,yu = coords[u]
                xv,yv = coords[v]
                plt.arrow(xu, yu, (xv-xu)*0.85, (yv-yu)*0.85, head_width=1.4, length_includes_head=True, alpha=0.25)
                midx = (xu+xv)/2; midy=(yu+yv)/2
                plt.text(midx, midy, f"{CAP_MATRIX[u][v]}", fontsize=8, alpha=0.6)

    # draw nodes
    for i,(x,y) in enumerate(coords):
        if i==SOURCE:
            plt.scatter(x,y, s=220, marker='s', label=f"src {i}")
        elif i==SINK:
            plt.scatter(x,y, s=220, marker='D', label=f"sink {i}")
        else:
            plt.scatter(x,y, s=140, marker='o')
        plt.text(x+1.5, y+1.5, str(i), fontsize=10)
    # mark min-cut edges
    for (u,v,c) in baseline_cut:
        xu,yu = coords[u]; xv,yv = coords[v]
        plt.plot([xu,xv],[yu,yv], 'r--', linewidth=2.5)
        plt.scatter([(xu+xv)/2], [(yu+yv)/2], marker='x', color='r', s=100)
    plt.axis('off'); plt.title("Graph (capacities shown). Baseline min-cut edges marked in red.")
    plt.tight_layout(); plt.show()
