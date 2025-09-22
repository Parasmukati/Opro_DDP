import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import requests
import time
import textwrap
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
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
        "max_tokens": 800
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
# TSP Config
# =======================
NUM_CITIES = 8
ITERATIONS = 8
LLM_BATCH_SIZE = 3

def generate_cities(num_cities):
    np.random.seed(42)
    return pd.DataFrame({
        "city": [f"City{i}" for i in range(num_cities)],
        "x": np.random.randint(0, 100, size=num_cities),
        "y": np.random.randint(0, 100, size=num_cities)
    })

def compute_distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = cities.iloc[i]["x"] - cities.iloc[j]["x"]
            dy = cities.iloc[i]["y"] - cities.iloc[j]["y"]
            dist[i, j] = np.hypot(dx, dy)
    return dist

def evaluate_route(route, dist_matrix):
    if not isinstance(route, list) or len(route) != len(dist_matrix):
        return None
    if sorted(route) != list(range(len(dist_matrix))):
        return None
    total_dist = sum(dist_matrix[route[i]][route[(i+1)%len(route)]] for i in range(len(route)))
    return total_dist

# =======================
# LLM Prompt
# =======================
def build_tsp_prompt(cities, history):
    n = len(cities)
    coords = [(int(row["x"]), int(row["y"])) for _, row in cities.iterrows()]
    prompt = []
    prompt.append("Solve this Travelling Salesman Problem (TSP).")
    prompt.append(f"n = {n}")
    prompt.append(f"cities = {coords}  # format: (x, y)")
    prompt.append("")
    prompt.append("Requirements:")
    prompt.append("- Find a route visiting each city exactly once and returning to start.")
    prompt.append("- The route must be a permutation of range(n).")
    prompt.append("- Assign the final route to variable `route` as a list of length n.")
    prompt.append("- No prints, no functions, no imports.")
    if history:
        prompt.append("Recent good routes (lower distance is better):")
        for r, dist in history[:3]:
            prompt.append(f"- {r[:6]}... (len={len(r)}) → distance {dist:.2f}")
    prompt.append("")
    prompt.append("Return ONLY code in triple backticks.")
    return "\n".join(prompt)

# =======================
# Execute LLM Code
# =======================
def execute_llm_code(llm_output, cities, dist_matrix):
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
            "bin": bin, "hex": hex, "oct": oct, "divmod": divmod, "round": round,
            "random": random,"__import__": __import__, "print": print

        }
        exec_env = {
            "__builtins__": safe_builtins,
            "n": len(cities)
        }
        print("Executed Code:\n", code)
        exec(code, exec_env)
        # print("Executed Code:\n", code)
        if "route" in exec_env:
            route = exec_env["route"]
            dist = evaluate_route(route, dist_matrix)
            if dist:
                return route, code, True
    except Exception as e:
        print("Execution error:", e)
    return None, None, False

# =======================
# OPRO for TSP
# =======================
def opro_tsp(cities, dist_matrix, iterations, batch_size):
    history = []
    successful_codes = []
    for step in range(iterations):
        print(f"\n=== Iteration {step+1} ===")
        prompt = build_tsp_prompt(cities, history)
        for b in range(batch_size):
            llm_output = groq_generate(prompt)
            route, code, success = execute_llm_code(llm_output, cities, dist_matrix)
            if success and route:
                dist = evaluate_route(route, dist_matrix)
                history.append((route, dist))
                successful_codes.append((route, dist, code))
                print(f"Batch {b+1}: distance {dist:.2f}")
        history.sort(key=lambda x: x[1])
    return history[0], [h[1] for h in history], successful_codes

# =======================
# OR-Tools Baseline
# =======================
def ortools_tsp(dist_matrix):
    n = len(dist_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_params)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route, solution.ObjectiveValue()
    return None, None

# =======================
# Run Experiment
# =======================
cities_df = generate_cities(NUM_CITIES)
dist_matrix = compute_distance_matrix(cities_df)

print("Cities:\n", cities_df)

best_sol, progression, successful_attempts = opro_tsp(cities_df, dist_matrix, ITERATIONS, LLM_BATCH_SIZE)
ortools_route, ortools_dist = ortools_tsp(dist_matrix)

print("\n=== FINAL RESULTS ===")
print(f"OPRO Best Route: {best_sol[0]} → distance {best_sol[1]:.2f}")
print(f"OR-Tools Route: {ortools_route} → distance {ortools_dist:.2f}")

# Plot
plt.figure(figsize=(8,6))
# OPRO
opro_x = [cities_df.iloc[i]["x"] for i in best_sol[0] + [best_sol[0][0]]]
opro_y = [cities_df.iloc[i]["y"] for i in best_sol[0] + [best_sol[0][0]]]
plt.plot(opro_x, opro_y, 'b-o', label=f"OPRO LLM (dist={best_sol[1]:.2f})")
# OR-Tools
ortools_x = [cities_df.iloc[i]["x"] for i in ortools_route + [ortools_route[0]]]
ortools_y = [cities_df.iloc[i]["y"] for i in ortools_route + [ortools_route[0]]]
plt.plot(ortools_x, ortools_y, 'r--x', label=f"OR-Tools (dist={ortools_dist:.2f})")
plt.scatter(cities_df["x"], cities_df["y"], c="black", s=60, zorder=5)
for i, row in cities_df.iterrows():
    plt.text(row["x"]+1, row["y"]+1, row["city"], fontsize=8)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Travelling Salesman Problem (8 Cities)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
