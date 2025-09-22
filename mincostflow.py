# ============================
# OPRO - Minimum Cost Flow (single cell)
# ============================
import numpy as np
import matplotlib.pyplot as plt
import requests, time, textwrap, random, heapq
from collections import defaultdict
from api_key import GROQ_API_KEY

# -----------------
# Groq API Config (same pattern)
# -----------------
GROQ_API_KEY = GROQ_API_KEY
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5, timeout=30):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL,
               "messages":[{"role":"system","content":"You generate ONLY Python code for optimization problems. Return direct code only, no explanation."},
                           {"role":"user","content":prompt}],
               "temperature":0.1,"max_tokens":3000}
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
# Problem generator: directed graph with capacity & cost
# -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_NODES = 8
SOURCE = 0
SINK = NUM_NODES - 1

def generate_min_cost_graph(n, density=0.25, cap_low=1, cap_high=10, cost_low=1, cost_high=20):
    cap = [[0]*n for _ in range(n)]
    cost = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u==v: continue
            if random.random() < density:
                c = int(np.random.randint(cap_low, cap_high+1))
                w = int(np.random.randint(cost_low, cost_high+1))
                cap[u][v] = c
                cost[u][v] = w
    # ensure connectivity from source and to sink
    for _ in range(max(1,n//3)):
        a = random.randint(1,n-2); b = random.randint(1,n-2)
        if a!=b:
            cap[SOURCE][a] = max(cap[SOURCE][a], random.randint(cap_low,cap_high))
            cost[SOURCE][a] = cost[SOURCE][a] or random.randint(cost_low,cost_high)
            cap[b][SINK] = max(cap[b][SINK], random.randint(cap_low,cap_high))
            cost[b][SINK] = cost[b][SINK] or random.randint(cost_low,cost_high)
    return cap, cost

CAP, COST = generate_min_cost_graph(NUM_NODES, density=0.28, cap_low=1, cap_high=8, cost_low=1, cost_high=15)

print("Capacity matrix:")
for i,row in enumerate(CAP):
    print(i, row)
print("\nCost matrix:")
for i,row in enumerate(COST):
    print(i, row)

# -----------------
# Baseline: Min-Cost Max-Flow via Successive Shortest Augmenting Path (Dijkstra with potentials)
# -----------------
def min_cost_max_flow(cap, cost, s, t, required_flow=None):
    n = len(cap)
    # residual capacities and costs
    res_cap = [row[:] for row in cap]
    res_cost = [[cost[u][v] if cap[u][v]>0 else 0 for v in range(n)] for u in range(n)]
    # residual reverse edges will use negative cost when flow sent
    flow = [[0]*n for _ in range(n)]
    total_flow = 0
    total_cost = 0
    # potentials for reduced costs (Johnson)
    potential = [0]*n

    def dijkstra():
        dist = [float('inf')]*n
        parent = [(-1,-1)]*n  # (prev_node, direction 1=forward 0=backward)
        dist[s] = 0
        pq = [(0,s)]
        while pq:
            d,u = heapq.heappop(pq)
            if d>dist[u]: continue
            for v in range(n):
                # forward edge
                if res_cap[u][v] > 0:
                    rcost = res_cost[u][v] + potential[u] - potential[v]
                    nd = d + rcost
                    if nd < dist[v]:
                        dist[v] = nd; parent[v] = (u,1); heapq.heappush(pq,(nd,v))
                # backward edge (if we have flow to cancel)
                if flow[v][u] > 0:
                    rcost = -res_cost[v][u] + potential[u] - potential[v]
                    nd = d + rcost
                    if nd < dist[v]:
                        dist[v] = nd; parent[v] = (v,0); heapq.heappush(pq,(nd,v))
        return dist, parent

    while True:
        dist, parent = dijkstra()
        if dist[t] == float('inf'):
            break
        # update potentials
        for i in range(n):
            if dist[i] < float('inf'):
                potential[i] += dist[i]
        # find bottleneck
        bottleneck = float('inf')
        v = t
        path = []
        while v != s:
            u,dirf = parent[v]
            if u==-1: break
            if dirf==1:
                bottleneck = min(bottleneck, res_cap[u][v])
                path.append((u,v,1))
                v = u
            else:
                # backward: capacity is flow[v][u] available to reduce
                bottleneck = min(bottleneck, flow[v][u])
                path.append((v,u,0))
                v = v
                v = parent[v][0]
        if bottleneck==float('inf') or parent[t][0]==-1:
            break
        # if required_flow specified, limit augment
        if required_flow is not None:
            bottleneck = min(bottleneck, required_flow - total_flow)
            if bottleneck <= 0: break
        # apply augmentation along path (reverse order)
        v = t
        while v != s:
            u,dirf = parent[v]
            if dirf==1:
                flow[u][v] += bottleneck
                res_cap[u][v] -= bottleneck
                # ensure reverse capacity exists logically (for cancel)
                # negative cost on reverse edge
                total_cost += bottleneck * res_cost[u][v]
                v = u
            else:
                # cancel backward flow
                flow[v][u] -= bottleneck
                res_cap[v][u] += bottleneck
                total_cost -= bottleneck * res_cost[v][u]
                # v already set in parent loop earlier
                v = u
        total_flow += bottleneck
        # stop if we achieved required_flow
        if required_flow is not None and total_flow >= required_flow:
            break
    return flow, int(total_flow), int(total_cost)

baseline_flow_mat, baseline_flow_val, baseline_cost = min_cost_max_flow(CAP, COST, SOURCE, SINK)
print("\nBaseline min-cost max-flow -> flow:", baseline_flow_val, "cost:", baseline_cost)

# produce baseline triples for later comparison
baseline_triples = []
n = len(CAP)
for u in range(n):
    for v in range(n):
        f = baseline_flow_mat[u][v]
        if f>0:
            baseline_triples.append((u,v,int(f)))

# -----------------
# Utilities: normalization & validation of LLM output (flow triples)
# -----------------
def normalize_flow_candidate(candidate, n):
    """Accept adjacency matrix, list of triples, dict forms."""
    if candidate is None: return None
    # adjacency matrix
    if isinstance(candidate, list) and len(candidate)==n and all(isinstance(row, list) for row in candidate):
        triples = []
        for u in range(n):
            for v in range(n):
                try:
                    f = candidate[u][v]
                except Exception:
                    return None
                if f and int(f)!=0:
                    triples.append((int(u),int(v),int(f)))
        return triples
    # list of triples
    if isinstance(candidate, list):
        ok = True; triples=[]
        for it in candidate:
            if not (isinstance(it,(list,tuple)) and len(it)==3):
                ok=False; break
            u,v,f = it
            try:
                triples.append((int(u),int(v),int(f)))
            except:
                ok=False; break
        if ok: return triples
    # dict with tuple keys or "u,v" keys
    if isinstance(candidate, dict):
        triples=[]
        for k,v in candidate.items():
            if isinstance(k, tuple) and len(k)==2:
                u,k2=k.split(',')
                try: triples.append((int(u),int(k2),int(v)))
                except: return None
            elif isinstance(k,str) and ',' in k:
                a,b = k.split(',')
                try: triples.append((int(a.strip()),int(b.strip()),int(v)))
                except: return None
            else:
                return None
        return triples
    return None

def validate_min_cost_flow(triples, cap, cost, s, t, required_flow=None):
    """Validate capacity, conservation, compute total flow and total cost. Returns (valid, flow_val, total_cost, message)."""
    n = len(cap)
    if triples is None: return False, None, None, "no triples"
    # build flow matrix
    flow = [[0]*n for _ in range(n)]
    for u,v,f in triples:
        if not (0<=u<n and 0<=v<n): return False,None,None,f"edge ({u},{v}) out of bounds"
        if f<0: return False,None,None,f"negative flow on ({u},{v})"
        if f>cap[u][v]: return False,None,None,f"flow > capacity on ({u},{v})"
        flow[u][v] += f
    # conservation check
    for node in range(n):
        if node==s or node==t: continue
        inflow = sum(flow[u][node] for u in range(n))
        outflow = sum(flow[node][v] for v in range(n))
        if inflow != outflow:
            return False,None,None,f"flow conservation violated at {node} (in {inflow} != out {outflow})"
    total_out = sum(flow[s][v] for v in range(n))
    total_in_t = sum(flow[u][t] for u in range(n))
    if total_out != total_in_t:
        return False,None,None,f"source out {total_out} != sink in {total_in_t}"
    total_cost = 0
    for u in range(n):
        for v in range(n):
            if flow[u][v]>0:
                total_cost += flow[u][v] * cost[u][v]
    # If required_flow specified, ensure it matches
    if required_flow is not None and total_out != required_flow:
        return False, None, None, f"flow {total_out} != required {required_flow}"
    return True, int(total_out), int(total_cost), "ok"

# -----------------
# Prompt builder for Min-Cost Flow
# -----------------
def build_mincost_prompt(cap, cost, s, t, history):
    n = len(cap)
    lines=[]
    lines.append("Solve this MINIMUM-COST flow problem on a directed graph.")
    lines.append(f"n = {n}; source = {s}; sink = {t}")
    lines.append("capacity = " + str(cap))
    lines.append("cost = " + str(cost))
    lines.append("")
    lines.append("STRICT requirements:")
    lines.append("- Return ONLY Python code in a single fenced block.")
    lines.append("- At the END assign EXACTLY these two variables:")
    lines.append("    flow_solution = [(u,v,flow), ...]  # integer triples")
    lines.append("    total_cost = <integer>")
    lines.append("- Optionally you may also return total_flow = <int>")
    lines.append("- Flow must respect capacities and conservation, and total_cost must be the sum of flow*edge_cost.")
    lines.append("- You must produce the MINIMUM possible cost for the maximum flow value (or if required_flow is specified, min-cost for that flow).")
    lines.append("- Do NOT use external libs like networkx, pulp, or ortools; pure Python is required.")
    if history:
        lines.append("\nRecent attempts (flow,cost):")
        for _, (f,c) in history[:4]:
            lines.append(f"- flow={f}, cost={c}")
    lines.append("\nReturn ONLY the fenced python block.")
    return "\n".join(lines)

# -----------------
# Execute LLM code safely & extract flow
# -----------------
def run_llm_code_and_get_flow(llm_output, cap, cost, s, t):
    if not llm_output: return None, None, False
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
        exec_env = {"__builtins__": safe_builtins, "n": len(cap), "capacity": cap, "cost": cost, "source": s, "sink": t}
        try:
            exec(code, exec_env)
            print(code)
        except Exception as e:
            print("LLM code execution error:", e)
            print("LLM Code (trunc):\n", "\n".join(code.splitlines()[:200]))
            return None, code, False
        candidate = None; reported_cost = None; reported_flow = None
        if "flow_solution" in exec_env:
            candidate = exec_env["flow_solution"]
            reported_cost = exec_env.get("total_cost", None)
            reported_flow = exec_env.get("total_flow", None)
        elif "solution" in exec_env:
            candidate = exec_env["solution"]
            reported_cost = exec_env.get("total_cost", None)
            reported_flow = exec_env.get("total_flow", None)
        if candidate is None:
            print("LLM did not set flow_solution. Code was:\n", code[:800])
            return None, code, False
        triples = normalize_flow_candidate(candidate, len(cap))
        if triples is None:
            print("Could not normalize candidate flow. Candidate type:", type(candidate))
            return None, code, False
        valid, fval, tcost, msg = validate_min_cost_flow(triples, cap, cost, s, t)
        if not valid:
            print("LLM solution invalid:", msg)
            return triples, code, False
        # if LLM reported cost or flow, check consistency
        if reported_cost is not None and int(reported_cost) != int(tcost):
            print("LLM reported total_cost mismatch:", reported_cost, "computed:", tcost)
            return triples, code, False
        if reported_flow is not None and int(reported_flow) != int(fval):
            print("LLM reported total_flow mismatch:", reported_flow, "computed:", fval)
            return triples, code, False
        return triples, code, (int(fval), int(tcost))
    except Exception as e:
        print("Parse error:", e)
        return None, None, False

# -----------------
# OPRO loop
# -----------------
ITERATIONS = 6
LLM_BATCH_SIZE = 3

def opro_mincost(cap, cost, s, t, iterations, batch):
    history = []   # (triples, (flow,cost))
    successes = []
    for it in range(iterations):
        print(f"\n=== Iteration {it+1}/{iterations} ===")
        prompt = build_mincost_prompt(cap, cost, s, t, history)
        for b in range(batch):
            print(f" Requesting LLM (batch {b+1}/{batch})...")
            llm_out = groq_generate(prompt)
            triples, code, result = run_llm_code_and_get_flow(llm_out, cap, cost, s, t)
            if result and result[0]>0:
                history.append((triples, result))
                successes.append({"triples":triples, "flow":result[0], "cost":result[1], "code":code})
                print(f"  ✓ Valid LLM flow found: flow={result[0]}, cost={result[1]}")
            else:
                print("  ✗ Invalid or no valid LLM flow")
        # keep best by (flow then min cost)
        history = sorted(history, key=lambda x:(-x[1][0], x[1][1]))[:20]
    best = max(history, key=lambda x:(x[1][0], -x[1][1])) if history else (None,None)
    return best, successes

# -----------------
# Run experiment
# -----------------
if __name__=="__main__":
    best_llm, successful_attempts = opro_mincost(CAP, COST, SOURCE, SINK, ITERATIONS, LLM_BATCH_SIZE)

    print("\n--- Baseline (SSAP) ---")
    print("Baseline flow:", baseline_flow_val, "baseline cost:", baseline_cost)
    print("Baseline triples (sample):", baseline_triples[:30])

    if best_llm and best_llm[1] is not None:
        print("\n--- Best LLM (OPRO) ---")
        print("LLM best flow, cost:", best_llm[1])
        print("LLM flow triples (sample):", best_llm[0][:30])
        print("Code that produced best flow:\n", successful_attempts[0]["code"])
    else:
        print("\nNo valid LLM solution found.")

    # Numeric comparison
    llm_cost = best_llm[1][1] if (best_llm and best_llm[1]) else None
    llm_flow = best_llm[1][0] if (best_llm and best_llm[1]) else 0
    base_cost = baseline_cost; base_flow = baseline_flow_val

    # Bar chart comparing flow then cost (if LLM found)
    labels = ["Baseline (flow)","OPRO LLM (flow)"]
    flows = [base_flow, llm_flow]
    plt.figure(figsize=(6,4)); plt.bar(labels, flows); plt.ylabel("Flow value"); plt.title("Flow: Baseline vs OPRO LLM"); plt.grid(axis='y', alpha=0.3); plt.show()

    if llm_cost is not None:
        plt.figure(figsize=(6,4))
        plt.bar(["Baseline (cost)","OPRO LLM (cost)"], [base_cost, llm_cost])
        plt.ylabel("Total cost"); plt.title("Total cost: Baseline vs OPRO LLM"); plt.grid(axis='y', alpha=0.3); plt.show()

    # Graph visualization (capacities and costs; baseline positive flows bold)
    n = len(CAP)
    angles = np.linspace(0,2*np.pi,n,endpoint=False)
    coords = [(50+40*np.cos(a), 50+40*np.sin(a)) for a in angles]
    plt.figure(figsize=(7,7))
    for u in range(n):
        for v in range(n):
            if CAP[u][v]>0:
                xu,yu = coords[u]; xv,yv = coords[v]
                plt.arrow(xu,yu,(xv-xu)*0.82,(yv-yu)*0.82, head_width=1.2, length_includes_head=True, alpha=0.25)
                midx,midy = (xu+xv)/2,(yu+yv)/2
                plt.text(midx, midy, f"{CAP[u][v]}/{COST[u][v]}", fontsize=8, alpha=0.6)
    # overlay baseline flows
    for (u,v,f) in baseline_triples:
        xu,yu = coords[u]; xv,yv = coords[v]
        plt.plot([xu,xv],[yu,yv], linewidth=3, alpha=0.8)
        plt.text((xu+xv)/2, (yu+yv)/2-3, f"f={f}", fontsize=9, color='blue')
    # nodes
    for i,(x,y) in enumerate(coords):
        if i==SOURCE: plt.scatter(x,y,s=220,marker='s',label=f"src {i}")
        elif i==SINK: plt.scatter(x,y,s=220,marker='D',label=f"sink {i}")
        else: plt.scatter(x,y,s=120,marker='o')
        plt.text(x+1.5,y+1.5,str(i))
    plt.title("Min-Cost Flow instance (cap/cost). Baseline flows bold. ")
    plt.axis('off'); plt.tight_layout(); plt.show()
