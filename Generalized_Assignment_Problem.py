# ================================
# OPRO - Generalized Assignment Problem (GAP)
# ================================
import numpy as np
import matplotlib.pyplot as plt
import requests, time, textwrap, random
from api_key import GROQ_API_KEY_1

# -----------------
# Groq API Config
# -----------------
GROQ_API_KEY = GROQ_API_KEY_1
GROQ_MODEL="openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5, timeout=30):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Generate ONLY Python code for generalized assignment problem (GAP). No explanation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1400
    }
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff * (2**attempt))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("Groq API error:", e)
            time.sleep(backoff)
    return None

# -----------------
# Problem Config
# -----------------
NUM_AGENTS = 4
NUM_TASKS = 8
ITERATIONS = 6
LLM_BATCH_SIZE = 2
np.random.seed(42)

def generate_gap_instance(m,n):
    cost = np.random.randint(5,30,(m,n)).tolist()
    usage = np.random.randint(1,10,(m,n)).tolist()
    capacity = np.random.randint(15,25,m).tolist()
    return cost,usage,capacity

cost_matrix, usage_matrix, capacity = generate_gap_instance(NUM_AGENTS, NUM_TASKS)

print("Cost Matrix (agents × tasks):")
for row in cost_matrix:
    print(row)
print("\nUsage Matrix (agents × tasks):")
for row in usage_matrix:
    print(row)
print("\nCapacities:", capacity)

# -----------------
# Evaluation
# -----------------
def evaluate_gap(assign, cost, usage, cap):
    """
    assign = list of (task,agent)
    cost/usage = [m][n]
    cap = [m]
    """
    m,n = len(cap), len(cost[0])
    if not isinstance(assign,list): return False,None
    seen = set()
    total_cost = 0
    load = [0]*m
    for pair in assign:
        if not (isinstance(pair,(list,tuple)) and len(pair)==2): return False,None
        t,a = pair
        if t in seen: return False,None
        if a<0 or a>=m or t<0 or t>=n: return False,None
        seen.add(t)
        total_cost += cost[a][t]
        load[a] += usage[a][t]
        if load[a] > cap[a]:
            return False,None
    if len(assign)!=n: return False,None
    return True,total_cost

# -----------------
# Prompt Builder
# -----------------
def build_gap_prompt(cost,usage,capacity,history):
    lines=[]
    lines.append("Solve this Generalized Assignment Problem (GAP).")
    lines.append(f"Cost matrix = {cost}  # rows=agents, cols=tasks")
    lines.append(f"Usage matrix = {usage}")
    lines.append(f"Capacities = {capacity}")
    lines.append("")
    lines.append("STRICT requirements:")
    lines.append("- Assign each task to exactly one agent.")
    lines.append("- Do not exceed capacity of any agent.")
    lines.append("- Minimize total cost.")
    lines.append("- Do NOT use pulp, ortools, or any external libraries.")
    lines.append("- You may use brute force, DFS/backtracking, or heuristics.")
    lines.append("- Return variable `assignment = [(task,agent), ...]` and `total_cost`.")
    if history:
        lines.append("\nRecent feasible solutions (costs):")
        for _,c in history[:2]:
            lines.append(f"- cost={c}")
    return "\n".join(lines)

# -----------------
# Runner
# -----------------
def run_llm_gap(llm_output,cost,usage,cap):
    if not llm_output: return None,None,False
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
        exec_env = {"__builtins__": safe_builtins,
                    "cost_matrix": cost, "usage_matrix": usage, "capacity": cap}
        exec(code,exec_env)
        if "assignment" not in exec_env: return None,code,False
        assign = exec_env["assignment"]
        valid,total = evaluate_gap(assign,cost,usage,cap)
        if not valid: return assign,code,False
        return assign,code,total
    except Exception as e:
        print("Exec error:",e)
        return None,None,False

# -----------------
# Simple Baseline (Greedy)
# -----------------
def greedy_gap(cost,usage,cap):
    m,n=len(cap),len(cost[0])
    load=[0]*m; assign=[]; total=0
    for t in range(n):
        best=None; best_c=1e9
        for a in range(m):
            if load[a]+usage[a][t]<=cap[a]:
                if cost[a][t]<best_c:
                    best_c=cost[a][t]; best=a
        if best is None: return None,None
        assign.append((t,best))
        load[best]+=usage[best][t]
        total+=cost[best][t]
    return assign,total

# -----------------
# OPRO Loop
# -----------------
def opro_gap(cost,usage,cap,iters,batch):
    history,successes=[],[]
    for it in range(iters):
        print(f"\nIteration {it+1}/{iters}")
        prompt=build_gap_prompt(cost,usage,cap,history)
        for b in range(batch):
            llm_out=groq_generate(prompt)
            assign,code,total=run_llm_gap(llm_out,cost,usage,cap)
            if total:
                history.append((assign,total))
                successes.append({"assign":assign,"cost":total,"code":code})
                print(f" ✓ Feasible LLM GAP: cost={total}")
            else:
                print(" ✗ Invalid LLM GAP")
        history=sorted(history,key=lambda x:x[1])[:10]
    best=min(history,key=lambda x:x[1]) if history else (None,None)
    return best,successes

# -----------------
# Run
# -----------------
if __name__=="__main__":
    best_llm,codes=opro_gap(cost_matrix,usage_matrix,capacity,ITERATIONS,LLM_BATCH_SIZE)
    if best_llm[0]:
        print("\nBest LLM GAP cost:",best_llm[1])
        print("Assignment:",best_llm[0])
        print("code:\n",codes[0]["code"])
    greedy_assign,greedy_cost=greedy_gap(cost_matrix,usage_matrix,capacity)
    print("\nGreedy baseline cost:",greedy_cost)

    # --- Compare plot ---
    labels=[f"T{t}" for t in range(NUM_TASKS)]
    fig,ax=plt.subplots(figsize=(8,5))
    x=np.arange(NUM_TASKS)
    if greedy_assign:
        ax.scatter(x,[a for t,a in greedy_assign],c="blue",label=f"Greedy cost={greedy_cost}")
    if best_llm[0]:
        ax.scatter(x,[a for t,a in best_llm[0]],c="red",marker="x",label=f"OPRO LLM cost={best_llm[1]}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Agents"); ax.set_xlabel("Tasks")
    ax.legend(); ax.set_title("GAP Assignments Comparison")
    plt.show()
