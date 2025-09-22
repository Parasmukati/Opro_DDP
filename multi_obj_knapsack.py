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
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You generate ONLY Python code for optimization problems. Return direct code, no explanations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 700
    }
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload)
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

# =======================
# Multi-Objective Knapsack Config
# =======================
CAPACITY = 50
NUM_ITEMS = 8
ITERATIONS = 8
LLM_BATCH_SIZE = 3

def generate_items(num_items):
    np.random.seed(42)
    return pd.DataFrame({
        "item": [f"Item{i+1}" for i in range(num_items)],
        "weight": np.random.randint(5, 20, size=num_items),
        "value": np.random.randint(30, 120, size=num_items),
        "risk": np.random.randint(10, 40, size=num_items)
    })

def evaluate_solution(solution, items, capacity):
    if not isinstance(solution, list) or len(solution) != len(items):
        return None
    if not all(x in [0, 1] for x in solution):
        return None
    total_weight = sum(w for w, take in zip(items["weight"], solution) if take)
    if total_weight > capacity:
        return None
    total_value = sum(v for v, take in zip(items["value"], solution) if take)
    total_risk = sum(r for r, take in zip(items["risk"], solution) if take)
    return (total_value, total_risk)

# Pareto frontier filter
def pareto_front(solutions):
    pareto = []
    for sol in solutions:
        dominated = False
        for other in solutions:
            if other != sol:
                if other[1][0] >= sol[1][0] and other[1][1] <= sol[1][1]:
                    if other[1][0] > sol[1][0] or other[1][1] < sol[1][1]:
                        dominated = True
                        break
        if not dominated:
            pareto.append(sol)
    return pareto

# =======================
# CP-SAT Baseline (scalarization)
# =======================
# def cp_sat_multi_obj(weights, values, risks, capacity, lambdas=[0.1, 0.5, 1, 2, 5]):
#     solutions = []
#     for lam in lambdas:
#         model = cp_model.CpModel()
#         n = len(weights)
#         x = [model.NewBoolVar(f'x_{i}') for i in range(n)]
#         model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
#         model.Maximize(sum(values[i] * x[i] for i in range(n)) - int(lam*100) * sum(risks[i] * x[i] for i in range(n)))
#         solver = cp_model.CpSolver()
#         status = solver.Solve(model)
#         if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#             sol = [1 if solver.Value(x[i]) else 0 for i in range(n)]
#             obj = evaluate_solution(sol, items_df, capacity)
#             if obj: solutions.append((sol, obj))
#     return pareto_front(solutions)

# =======================
# LLM Prompt
# =======================
def build_multiobj_prompt(items, capacity, history):
    py_items = [(int(w), int(val), int(risk)) for w, val, risk in zip(items["weight"], items["value"], items["risk"])]
    n = len(py_items)
    prompt = []
    prompt.append("Solve this MULTI-OBJECTIVE knapsack problem (maximize value, minimize risk).")
    prompt.append(f"capacity = {capacity}")
    prompt.append(f"items_data = {py_items}  # format: (weight, value, risk)")
    prompt.append(f"n = {n}")
    prompt.append("")
    prompt.append("Requirements:")
    prompt.append("- Use dynamic programming or brute force.")
    prompt.append("- Ensure weight <= capacity.")
    prompt.append("- Each solution is binary list of length n.")
    prompt.append("- Collect multiple non-dominated solutions (Pareto optimal).")
    prompt.append("- Assign final set of solutions to variable `pareto_solutions`.")
    prompt.append("- No prints, no functions, no imports.")
    if history:
        prompt.append("Recent good solutions:")
        for sol, obj in history[:3]:
            prompt.append(f"- {sol} → value {obj[0]}, risk {obj[1]}")
    prompt.append("")
    prompt.append("Return ONLY code in triple backticks.")
    return "\n".join(prompt)

# =======================
# Execute LLM Code
# =======================
def execute_llm_code(llm_output, items, capacity):
    if not llm_output: return None, None, False
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
            "items_data": [(int(w), int(v), int(r)) for w,v,r in zip(items["weight"], items["value"], items["risk"])],
            "capacity": capacity,
            "n": len(items)
        }
        exec(code, exec_env)
        print("Executed Code:\n", code)
        if "pareto_solutions" in exec_env:
            sols = exec_env["pareto_solutions"]
            if isinstance(sols, list):
                valid = []
                for sol in sols:
                    if isinstance(sol, list) and len(sol) == len(items):
                        obj = evaluate_solution(sol, items, capacity)
                        if obj: valid.append((sol, obj))
                return valid, code, True
    except Exception as e:
        print("Execution error:", e)
    return None, None, False

# =======================
# OPRO for Multi-Objective Knapsack
# =======================
def opro_multiobj_knapsack(items, capacity, iterations, batch_size):
    history = []
    successful_codes = []
    for step in range(iterations):
        print(f"\n=== Iteration {step+1} ===")
        prompt = build_multiobj_prompt(items, capacity, history)
        for b in range(batch_size):
            llm_output = groq_generate(prompt)
            sols, code, success = execute_llm_code(llm_output, items, capacity)
            if success and sols:
                for sol, obj in sols:
                    history.append((sol, obj))
                    successful_codes.append((sol, obj, code))
                    print(f"Batch {b+1}: {sol} → value {obj[0]}, risk {obj[1]}")
        history = pareto_front(history)
    return history, successful_codes

# =======================
# Run Experiment
# =======================
items_df = generate_items(NUM_ITEMS)
print("Items:\n", items_df)

def cp_sat_multi_obj(weights, values, risks, capacity):
    pareto = []
    max_risk = sum(risks)

    for eps in range(0, max_risk+1, max(1, max_risk//20)):  # sweep 20 steps
        model = cp_model.CpModel()
        n = len(weights)
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # constraints
        model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
        model.Add(sum(risks[i] * x[i] for i in range(n)) <= eps)

        # objective
        model.Maximize(sum(values[i] * x[i] for i in range(n)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol = [1 if solver.Value(x[i]) else 0 for i in range(n)]
            val = sum(values[i] * sol[i] for i in range(n))
            risk = sum(risks[i] * sol[i] for i in range(n))
            pareto.append((sol, (val, risk)))

    # Deduplicate by (val, risk)
    unique = {}
    for sol, (val, risk) in pareto:
        unique[(val, risk)] = sol
    pareto = [(s, (v, r)) for (v, r), s in unique.items()]

    # Sort by risk
    pareto.sort(key=lambda x: x[1][1])
    return pareto


opro_pareto, successful_attempts = opro_multiobj_knapsack(items_df, CAPACITY, ITERATIONS, LLM_BATCH_SIZE)
cp_sat_pareto = cp_sat_multi_obj(items_df["weight"], items_df["value"], items_df["risk"], CAPACITY)

print("\n=== FINAL RESULTS ===")
print("Pareto Front (OPRO):")
for sol, (val, risk) in opro_pareto:
    print(f"{sol} → value={val}, risk={risk}")
print("\nPareto Front (CP-SAT Scalarization):")
for sol, (val, risk) in cp_sat_pareto:
    print(f"{sol} → value={val}, risk={risk}")

# Plot
plt.figure(figsize=(8,6))
opro_vals = [obj[0] for _,obj in opro_pareto]
opro_risks = [obj[1] for _,obj in opro_pareto]
cp_vals = [obj[0] for _,obj in cp_sat_pareto]
cp_risks = [obj[1] for _,obj in cp_sat_pareto]
plt.scatter(opro_risks, opro_vals, c="blue", label="OPRO Pareto", s=80)
plt.scatter(cp_risks, cp_vals, c="red", marker="x", label="CP-SAT (scalarized)", s=100)
plt.xlabel("Risk")
plt.ylabel("Value")
plt.title("Multi-Objective Knapsack (Value vs Risk)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

