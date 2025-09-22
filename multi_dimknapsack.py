import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import requests
import time
import textwrap
from ortools.sat.python import cp_model
from api_key import GROQ_API_KEY

# =======================
# Groq API Config
# =======================
GROQ_API_KEY = GROQ_API_KEY
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You generate ONLY direct Python code solving optimization problems with multiple constraints."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload)
            if r.status_code == 429:
                wait_time = backoff * (2 ** attempt)
                print(f"Rate limit hit. Sleeping {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("Groq API error:", e)
            time.sleep(backoff)
    return None

# =======================
# Multi-Dimensional Knapsack Config
# =======================
CAPACITY_WEIGHT = 50
CAPACITY_VOLUME = 60
NUM_ITEMS = 8
ITERATIONS = 10
LLM_BATCH_SIZE = 3

def generate_items(num_items):
    np.random.seed(42)
    return pd.DataFrame({
        "item": [f"Item{i+1}" for i in range(num_items)],
        "weight": np.random.randint(5, 20, size=num_items),
        "volume": np.random.randint(10, 30, size=num_items),
        "value": np.random.randint(30, 120, size=num_items)
    })

def cp_sat_knapsack_multi(weights, volumes, values, cap_weight, cap_volume):
    model = cp_model.CpModel()
    n = len(weights)
    x = [model.NewBoolVar(f'x_{i}') for i in range(n)]
    model.Add(sum(weights[i] * x[i] for i in range(n)) <= cap_weight)
    model.Add(sum(volumes[i] * x[i] for i in range(n)) <= cap_volume)
    model.Maximize(sum(values[i] * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selection = [1 if solver.Value(x[i]) else 0 for i in range(n)]
        total_value = sum(values[i] * selection[i] for i in range(n))
        return selection, total_value
    return [0]*n, 0

def evaluate_solution(solution, items, cap_weight, cap_volume):
    if not isinstance(solution, list) or len(solution) != len(items):
        return 0
    if not all(x in [0, 1] for x in solution):
        return 0
    total_weight = sum(w for w, take in zip(items["weight"], solution) if take)
    total_volume = sum(v for v, take in zip(items["volume"], solution) if take)
    if total_weight > cap_weight or total_volume > cap_volume:
        return 0
    return sum(val for val, take in zip(items["value"], solution) if take)

# =======================
# LLM Prompt
# =======================
def build_multi_prompt(items, cap_weight, cap_volume, history):
    py_items = [(int(w), int(vol), int(val)) for w, vol, val in zip(items["weight"], items["volume"], items["value"])]
    n = len(py_items)
    prompt = []
    prompt.append("Solve this MULTI-DIMENSIONAL 0/1 knapsack problem using dynamic programming EXACTLY.")
    prompt.append(f"capacity_weight = {cap_weight}")
    prompt.append(f"capacity_volume = {cap_volume}")
    prompt.append(f"items_data = {py_items}  # format: (weight, volume, value)")
    prompt.append(f"n = {n}  # number of items")
    prompt.append("")
    prompt.append("Requirements:")
    prompt.append("- Implement full 2D DP table (by weight and volume).")
    prompt.append("- Reconstruct chosen items as 'solution'.")
    prompt.append("- solution must be a binary list of length n (0/1).")
    prompt.append("- Do NOT print, import, or define functions.")
    prompt.append("- Return ONLY Python code enclosed in triple backticks.")
    prompt.append("")
    if history:
        prompt.append("Recent attempts and their values:")
        for sol, val in history[:3]:
            prompt.append(f"- {sol} → value {val}")
    prompt.append("")
    prompt.append("Example format:")
    prompt.append("```python")
    prompt.append("# dynamic programming multi-dimensional knapsack")
    prompt.append("solution = [0]*n")
    prompt.append("```")
    return "\n".join(prompt)

# =======================
# Execute LLM Code
# =======================
def execute_llm_code(llm_output, items, cap_weight, cap_volume):
    if not llm_output:
        return None, None, False
    try:
        if "```python" in llm_output:
            code = llm_output.split("```python")[1].split("```")[0]
        elif "```" in llm_output:
            code = llm_output.split("```")[1].split("```")[0]
        else:
            code = llm_output
        code = textwrap.dedent(code).strip()
        safe_builtins = {
            "range": range, "len": len, "sum": sum, "max": max, "min": min,
            "enumerate": enumerate, "list": list, "int": int, "zip": zip,
            "tuple": tuple, "abs": abs, "all": all, "any": any,
            "sorted": sorted, "map": map, "filter": filter,
            "bool": bool, "float": float, "str": str, "set": set, "dict": dict,
            "bin": bin, "hex": hex, "oct": oct, "divmod": divmod, "round": round

        }
        exec_env = {
            "__builtins__": safe_builtins,
            "items_data": [(int(w), int(v), int(val)) for w, v, val in zip(items["weight"], items["volume"], items["value"])],
            "capacity_weight": cap_weight,
            "capacity_volume": cap_volume,
            "n": len(items)
        }
        exec(code, exec_env)
        print("Executed Code:\n", code)
        if "solution" in exec_env:
            candidate = exec_env["solution"]
            if (isinstance(candidate, list) and len(candidate) == len(items) and all(x in (0,1) for x in candidate)):
                return candidate, code, True
    except Exception as e:
        print("Execution error:", e)
    return None, None, False

# =======================
# OPRO
# =======================
def opro_knapsack_multi(items, cap_weight, cap_volume, iterations, batch_size):
    history = []
    successful_codes = []
    for i in range(batch_size):
        sol = [random.choice([0, 1]) for _ in range(len(items))]
        val = evaluate_solution(sol, items, cap_weight, cap_volume)
        history.append((sol, val))
    history.sort(key=lambda x: -x[1])
    best_values = []
    for step in range(iterations):
        print(f"\n=== Iteration {step+1} ===")
        prompt = build_multi_prompt(items, cap_weight, cap_volume, history[:3])
        for batch_idx in range(batch_size):
            llm_output = groq_generate(prompt)
            sol, code, success = execute_llm_code(llm_output, items, cap_weight, cap_volume)
            if success:
                val = evaluate_solution(sol, items, cap_weight, cap_volume)
                history.append((sol, val))
                successful_codes.append((sol, val, code))
                print(f"Batch {batch_idx+1}: {sol} → value {val}")
        history.sort(key=lambda x: -x[1])
        best_values.append(history[0][1])
        print(f"Best so far: {history[0][1]}")
    return history[0], best_values, successful_codes

# =======================
# Run Experiment
# =======================
items_df = generate_items(NUM_ITEMS)
print("Items:\n", items_df)
print(f"Capacity Weight = {CAPACITY_WEIGHT}, Capacity Volume = {CAPACITY_VOLUME}")

best_sol, best_progression, successful_attempts = opro_knapsack_multi(
    items_df, CAPACITY_WEIGHT, CAPACITY_VOLUME, ITERATIONS, LLM_BATCH_SIZE
)

weights = items_df["weight"].tolist()
volumes = items_df["volume"].tolist()
values = items_df["value"].tolist()
dp_sol, dp_val = cp_sat_knapsack_multi(weights, volumes, values, CAPACITY_WEIGHT, CAPACITY_VOLUME)

print("\n=== FINAL RESULTS ===")
print(f"OPRO Best Solution: {best_sol[0]} → value {best_sol[1]}")
print(f"CP-SAT Optimal Solution: {dp_sol} → value {dp_val}")

plt.plot(best_progression, label="OPRO LLM")
plt.axhline(y=dp_val, color="r", linestyle="--", label="Optimal (CP-SAT)")
plt.xlabel("Iteration")
plt.ylabel("Best Value Found")
plt.title("Multi-Dimensional Knapsack (LLM vs Optimal)")
plt.legend()
plt.show()
