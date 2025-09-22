# ============================
# OPRO Bin Packing Problem (BPP)
# ============================
import numpy as np
import matplotlib.pyplot as plt
import requests, time, textwrap, random
from ortools.sat.python import cp_model
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
            {"role": "system", "content": "Generate ONLY Python code for bin packing problems. No explanation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1200
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
NUM_ITEMS = 12
BIN_CAPACITY = 20
ITERATIONS = 6
LLM_BATCH_SIZE = 2
np.random.seed(42)

def generate_bpp_instance(num_items, capacity):
    sizes = np.random.randint(2, 15, size=num_items)
    return sizes.tolist(), capacity

items, CAPACITY = generate_bpp_instance(NUM_ITEMS, BIN_CAPACITY)
print("Items:", items)
print("Bin capacity:", CAPACITY)

# -----------------
# Evaluation
# -----------------
def evaluate_bins(bins, items, capacity):
    """ bins = [[item indices], ...] """
    if not isinstance(bins, list): return False, None
    used = []
    for b in bins:
        if not isinstance(b, list): return False, None
        total = sum(items[i] for i in b)
        if total > capacity: return False, None
        used.extend(b)
    if sorted(used) != list(range(len(items))): 
        return False, None
    return True, len(bins)

# -----------------
# Prompt Builder
# -----------------
def build_bpp_prompt(items, capacity, history):
    lines = []
    lines.append("Solve this Bin Packing Problem (BPP).")
    lines.append(f"Items = {items}  # list of item sizes")
    lines.append(f"Bin capacity = {capacity}")
    lines.append("")
    lines.append("Requirements:")
    lines.append("- Place each item index exactly once into some bin.")
    lines.append("- Each bin is a list of item indices (0-based).")
    lines.append("- No bin total size may exceed capacity.")
    lines.append("- Minimize the number of bins used.")
    lines.append("- Do not use pulp or ortools or any external libraries.")
    lines.append("Return a variable named `bins = [[...], [...], ...]`")
    if history:
        lines.append("\nRecent solutions (#bins):")
        for _, nb in history[:2]:
            lines.append(f"- {nb} bins")
    return "\n".join(lines)

# -----------------
# Runner for LLM code
# -----------------
def run_llm_code_and_get_bins(llm_output, items, capacity):
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
            "random": random,"__import__": __import__, "print": print

        }
        exec_env = {"__builtins__": safe_builtins, "items": items, "capacity": capacity}
        exec(code, exec_env)
        if "bins" not in exec_env: return None, code, False
        bins = exec_env["bins"]
        valid, nbins = evaluate_bins(bins, items, capacity)
        if not valid: return bins, code, False
        return bins, code, nbins
    except Exception as e:
        print("Exec error:", e)
        return None,None,False

# -----------------
# OR-Tools Baseline
# -----------------
def ortools_bpp(items, capacity):
    num_items = len(items)
    num_bins = num_items
    model = cp_model.CpModel()
    x = {}
    for i in range(num_items):
        for b in range(num_bins):
            x[(i,b)] = model.NewBoolVar(f"x_{i}_{b}")
    y = [model.NewBoolVar(f"y_{b}") for b in range(num_bins)]
    for i in range(num_items):
        model.Add(sum(x[(i,b)] for b in range(num_bins)) == 1)
    for b in range(num_bins):
        model.Add(sum(x[(i,b)]*items[i] for i in range(num_items)) <= capacity * y[b])
    model.Minimize(sum(y[b] for b in range(num_bins)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bins = []
        for b in range(num_bins):
            if solver.Value(y[b]):
                bin_items = [i for i in range(num_items) if solver.Value(x[(i,b)])]
                bins.append(bin_items)
        return bins, len(bins)
    return None,None

# -----------------
# OPRO Loop
# -----------------
def opro_bpp(items, capacity, iters, batch):
    history,successes = [],[]
    for it in range(iters):
        print(f"\nIteration {it+1}/{iters}")
        prompt = build_bpp_prompt(items,capacity,history)
        for b in range(batch):
            llm_out = groq_generate(prompt)
            bins, code, nb = run_llm_code_and_get_bins(llm_out,items,capacity)
            if nb:
                history.append((bins,nb))
                successes.append({"bins":bins,"nb":nb,"code":code})
                print(f" ✓ Feasible LLM packing: {nb} bins")
            else:
                print(" ✗ Invalid LLM packing")
        history = sorted(history,key=lambda x:x[1])[:10]
    best = min(history,key=lambda x:x[1]) if history else (None,None)
    return best,successes

# -----------------
# Run
# -----------------
if __name__=="__main__":
    best_llm,codes = opro_bpp(items,CAPACITY,ITERATIONS,LLM_BATCH_SIZE)
    if best_llm[0]:
        print("\nBest LLM solution bins used:",best_llm[1])
        print("Bins:",best_llm[0])
        print("code:\n",codes[0]["code"])
    ort_bins, ort_n = ortools_bpp(items,CAPACITY)
    print("\nOR-Tools bins used:",ort_n)

    # --- Plot comparison ---
    fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
    def plot_bins(ax, bins, title):
        if not bins: 
            ax.set_title(title+" (no solution)")
            return
        for bi,bin_items in enumerate(bins):
            sizes = [items[i] for i in bin_items]
            left = 0
            for si,s in zip(bin_items,sizes):
                ax.barh(bi,s,left=left,label=f"item{si}")
                ax.text(left+s/2,bi,str(si),va="center",ha="center",color="white")
                left += s
        ax.set_xlim(0,CAPACITY)
        ax.set_ylim(-1,len(bins))
        ax.set_xlabel("Size"); ax.set_ylabel("Bin")
        ax.set_title(title)
    plot_bins(axes[0],ort_bins,f"OR-Tools ({ort_n} bins)")
    if best_llm[0]:
        plot_bins(axes[1],best_llm[0],f"OPRO LLM ({best_llm[1]} bins)")
    plt.tight_layout(); plt.show()
