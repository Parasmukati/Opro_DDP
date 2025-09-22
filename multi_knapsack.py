"""
Multi-Dimensional Knapsack Problem (MDKP) OPRO framework (single runnable cell).

Features:
- generates MDKP instances (multiple resource dimensions)
- solves optimally with OR-Tools CP-SAT (binary selection)
- prompts an LLM (Groq API) to produce code that returns a `solution` list
- executes and validates LLM-generated code in a restricted env
- OPRO loop: collects LLM proposals, evaluates & keeps the best
- rate-limit retries + exponential backoff for Groq calls
- plotting of progression vs optimal

Requirements:
- pip install ortools pandas matplotlib requests
- Put your Groq API key in api_key.GROQ_API_KEY (or set GROQ_API_KEY variable)
"""

import time
import textwrap
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
from api_key import GROQ_API_KEY  # expects your Groq key here

# --------------------------
# Groq API helpers
# --------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"   # change if needed
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=4, timeout=30):
    """Call Groq chat completion with retries on 429/network errors."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a careful code generator. Return ONLY a single Python code block "
                "that assigns a variable `solution` (list of 0/1 integers) of length n. "
                "Do not print or import anything."
            )},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 600
    }
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"Rate limit (429). Sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            wait = backoff * (2 ** attempt)
            print(f"Groq request error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None

# --------------------------
# Problem generation & baseline solver (CP-SAT)
# --------------------------
def generate_mdkp(num_items=8, dims=2, seed=42, weight_low=1, weight_high=20, value_low=10, value_high=120):
    """Generate an MDKP instance.
       Returns DataFrame with columns: item, weights (tuple), value
    """
    rnd = np.random.RandomState(seed)
    items = []
    weights = rnd.randint(weight_low, weight_high+1, size=(num_items, dims)).tolist()
    values = rnd.randint(value_low, value_high+1, size=num_items).tolist()
    for i in range(num_items):
        items.append({"item": f"Item{i+1}", "weights": tuple(int(w) for w in weights[i]), "value": int(values[i])})
    # capacities: set to ~40-60% of sum of weights per dimension to make it interesting
    caps = []
    for d in range(dims):
        total = sum(w[d] for w in weights)
        caps.append(int(total * rnd.uniform(0.35, 0.6)))
    df = pd.DataFrame(items)
    return df, caps

def cp_sat_mdkp_solver(items_df, capacities):
    """Solve binary MDKP optimally using CP-SAT. Returns (selection_list, total_value)."""
    model = cp_model.CpModel()
    n = len(items_df)
    dims = len(capacities)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    # capacity constraints across all dims
    for d in range(dims):
        model.Add(sum(int(items_df.iloc[i]["weights"][d]) * x[i] for i in range(n)) <= int(capacities[d]))
    model.Maximize(sum(int(items_df.iloc[i]["value"]) * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10  # reasonable limit
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel = [1 if solver.Value(x[i]) else 0 for i in range(n)]
        val = sum(int(items_df.iloc[i]["value"]) * sel[i] for i in range(n))
        return sel, val
    return [0]*n, 0

# --------------------------
# Evaluation & utils
# --------------------------
def evaluate_mdkp_solution(solution, items_df, capacities):
    """Return objective value if valid, else 0 for invalid/feasible check failure."""
    if not isinstance(solution, list):
        return 0
    n = len(items_df)
    if len(solution) != n:
        return 0
    if any(x not in (0,1) for x in solution):
        return 0
    dims = len(capacities)
    for d in range(dims):
        used = sum(solution[i] * int(items_df.iloc[i]["weights"][d]) for i in range(n))
        if used > capacities[d]:
            return 0
    total_value = sum(solution[i] * int(items_df.iloc[i]["value"]) for i in range(n))
    return total_value

# --------------------------
# LLM prompt builder (force literal data & n)
# --------------------------
def build_mdkp_code_prompt(items_df, capacities, history):
    py_items = [tuple(int(v) for v in row["weights"]) + (int(row["value"]),) for _, row in items_df.iterrows()]
    n = len(py_items)
    prompt_lines = [
        "Solve this MULTI-DIMENSIONAL 0/1 knapsack (MDKP) and return ONLY a python code block that assigns `solution`.",
        f"n = {n}",
        f"capacities = {list(int(c) for c in capacities)}",
        f"items_data = {py_items}  # each entry: (w1, w2, ..., value)",
        "",
        "Requirements:",
        "- Use the variables `n`, `capacities`, `items_data` exactly (do not assume other globals).",
        "- Choose 0/1 for each item (binary).",
        "- The final line MUST be: solution = [a list of length n with 0/1 integers].",
        "- Do NOT print or import. Provide only a single fenced python block.",
        "- The solution must be FEASIBLE (respect capacities) and OPTIMAL if possible.",
        "",
    ]
    if history:
        prompt_lines.append("Recent attempts (solution -> value):")
        for sol, val in history[:5]:
            prompt_lines.append(f"- {sol} -> {val}")
        prompt_lines.append("")
    prompt_lines.append("Return format example:")
    prompt_lines.append("```python")
    prompt_lines.append("# dynamic programming or direct algorithm using items_data and capacities")
    prompt_lines.append("# at the end:")
    prompt_lines.append("solution = [0]*n")
    prompt_lines.append("```")
    return "\n".join(prompt_lines)

# --------------------------
# Extract & execute LLM code safely
# --------------------------
def extract_code_block(llm_output):
    if not llm_output:
        return None
    if "```python" in llm_output:
        return llm_output.split("```python",1)[1].split("```",1)[0]
    if "```" in llm_output:
        return llm_output.split("```",1)[1].split("```",1)[0]
    # fallback: if LLM returns plain list, return that as code to assign
    if "[" in llm_output and "]" in llm_output:
        return f"solution = {llm_output.split('[',1)[1].split(']',1)[0]!s}"
    return None

def safe_exec_code(code, items_df, capacities):
    """
    Execute provided code in restricted environment. Return solution list or None.
    Allowed builtins kept minimal for DP logic.
    """
    try:
        code = textwrap.dedent(code)
        print(code)
        n = len(items_df)
        items_data = [tuple(int(x) for x in row["weights"]) + (int(row["value"]),) for _, row in items_df.iterrows()]
        safe_builtins = {
            "range": range, "len": len, "sum": sum, "max": max, "min": min,
            "enumerate": enumerate, "list": list, "int": int, "zip": zip,
            "tuple": tuple, "abs": abs, "all": all, "any": any,
            "sorted": sorted, "map": map, "filter": filter,
            "bool": bool, "float": float, "str": str, "set": set, "dict": dict,
            "bin": bin, "hex": hex, "oct": oct, "divmod": divmod, "round": round

        }
        env = {"__builtins__": safe_builtins, "items_data": items_data, "capacities": list(capacities), "n": n}
        exec(code, env, env)
        sol = env.get("solution", None)
        if isinstance(sol, list) and len(sol) == n and all(x in (0,1) for x in sol):
            return sol
        # sometimes LLM returns numpy array or other numeric types
        if hasattr(sol, "tolist"):
            cand = list(sol.tolist())
            if len(cand) == n and all(int(x) in (0,1) for x in cand):
                return [int(x) for x in cand]
        return None
    except Exception as e:
        # show brief message for debugging (user can inspect)
        print("Code execution error:", e)
        return None

# --------------------------
# OPRO loop for MDKP
# --------------------------
def opro_mdkp(items_df, capacities, iterations=8, batch_size=3):
    history = []
    n = len(items_df)
    # initial random seeds
    for _ in range(batch_size):
        sol = [random.choice([0,1]) for _ in range(n)]
        val = evaluate_mdkp_solution(sol, items_df, capacities)
        history.append((sol,val))
    history.sort(key=lambda x: -x[1])
    best_values = []
    successful_codes = []

    for it in range(iterations):
        print(f"\n--- Iteration {it+1} ---")
        prompt = build_mdkp_code_prompt(items_df, capacities, history[:6])
        for b in range(batch_size):
            llm_out = groq_generate(prompt)
            if not llm_out:
                print("No response from LLM.")
                continue
            code = extract_code_block(llm_out)
            if not code:
                print("Couldn't find a python code block in LLM output.")
                continue
            sol = safe_exec_code(code, items_df, capacities)
            if sol is None:
                print("LLM code produced invalid solution or failed. Preview (first 300 chars):")
                print(code[:300])
                continue
            val = evaluate_mdkp_solution(sol, items_df, capacities)
            history.append((sol,val))
            successful_codes.append({"iteration": it+1, "batch": b+1, "code": code, "solution": sol, "value": val})
            print(f"LLM proposal (batch {b+1}): value={val}, solution={sol}")
        history.sort(key=lambda x: -x[1])
        best_values.append(history[0][1])
        print(f"Best so far: value={history[0][1]}, solution={history[0][0]}")
    return history[0], best_values, successful_codes

# --------------------------
# Demo run (change params as desired)
# --------------------------
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    NUM_ITEMS = 8
    DIMS = 2               # number of resource dimensions
    ITERATIONS = 6
    BATCH = 3

    items_df, capacities = generate_mdkp(NUM_ITEMS, dims=DIMS, seed=42)
    print("Problem instance (MDKP):")
    for i,row in items_df.iterrows():
        print(f"  {row['item']}: weights={row['weights']}, value={row['value']}")
    print("Capacities:", capacities)

    print("\nSolving optimal baseline (CP-SAT)...")
    optimal_sol, optimal_val = cp_sat_mdkp_solver(items_df, capacities)
    print("CP-SAT optimal value:", optimal_val, "solution:", optimal_sol)

    print("\nRunning OPRO (LLM proposals)...")
    best, progression, codes = opro_mdkp(items_df, capacities, iterations=ITERATIONS, batch_size=BATCH)
    print("\nFinal best from OPRO:", best)

    # Show top successful codes if any
    if codes:
        top = sorted(codes, key=lambda x: x["value"], reverse=True)[:3]
        print("\nTop LLM-produced solutions:")
        for i, t in enumerate(top,1):
            print(f"\n#{i} - value {t['value']} (iter {t['iteration']}, batch {t['batch']})")
            # print code trimmed
            print(textwrap.indent(t['code'].strip()[:800], "  "))

    # Plot progression
    plt.figure(figsize=(8,4))
    plt.plot(range(1,len(progression)+1), progression, marker='o', label='OPRO best')
    plt.axhline(optimal_val, color='r', linestyle='--', label=f'Optimal ({optimal_val})')
    plt.xlabel("Iteration")
    plt.ylabel("Best value found")
    plt.title("MDKP: OPRO LLM vs CP-SAT Optimal")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
