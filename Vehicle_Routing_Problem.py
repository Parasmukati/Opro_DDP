# Single copyable cell: OPRO VRP (multiple vehicles, capacity constraints)
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import requests
import time
import textwrap
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from api_key import GROQ_API_KEY_1  # keep using your existing secret key import

# =======================
# Groq API Config (unchanged)
# =======================
GROQ_API_KEY = GROQ_API_KEY_1
# GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5, timeout=30):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You generate ONLY Python code for optimization problems. Return direct code only, no explanations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1500
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

# =======================
# Problem configuration
# =======================
NUM_CUSTOMERS = 10     # excluding depot; total nodes = NUM_CUSTOMERS + 1
NUM_VEHICLES = 4
VEHICLE_CAPACITY = 80
ITERATIONS = 8
LLM_BATCH_SIZE = 3
SEED = 42

def generate_vrp_instance(num_customers, seed=SEED):
    np.random.seed(seed)
    # depot at index 0
    coords = [(50,50)]  # depot centered
    # random customer coordinates
    xs = np.random.randint(0, 100, size=num_customers)
    ys = np.random.randint(0, 100, size=num_customers)
    demands = np.random.randint(5, 40, size=num_customers)  # demand per customer
    for x,y in zip(xs, ys):
        coords.append((int(x), int(y)))
    # demands array aligned: index 0 = depot demand 0
    demands_full = [0] + [int(d) for d in demands]
    return coords, demands_full

def compute_distance_matrix(coords):
    n = len(coords)
    mat = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            xj, yj = coords[j]
            mat[i][j] = float(np.hypot(xi-xj, yi-yj))
    return mat

def evaluate_routes(routes, dist_matrix, demands, capacity, num_vehicles_expected):
    """
    routes: list of vehicle routes, each route is list of node indices (0..n-1).
            convention: depot is index 0 but we expect routes without repeated depot entries:
            e.g., routes = [[1,3,5], [2,4], ...] meaning each vehicle starts at depot (0)->1->3->5->0
    Validate:
    - number of routes == num_vehicles_expected
    - each customer (1..n-1) visited exactly once or possibly not visited (we treat feasibility strictly: require all visited exactly once)
    - capacity per vehicle not exceeded
    - compute total distance
    Returns (is_valid, total_distance, vehicle_loads)
    """
    n = len(dist_matrix)
    customers = set(range(1, n))
    visited = []
    total_dist = 0.0
    vehicle_loads = []
    if not isinstance(routes, list):
        return False, None, None
    if len(routes) != num_vehicles_expected:
        # allow some vehicles empty but keep same count (user should return exactly NUM_VEHICLES lists)
        return False, None, None
    for route in routes:
        if not isinstance(route, list):
            return False, None, None
        load = 0
        prev = 0  # start at depot
        for node in route:
            if not isinstance(node, int) or node < 1 or node >= n:
                return False, None, None
            visited.append(node)
            total_dist += dist_matrix[prev][node]
            prev = node
            load += demands[node]
        # return to depot
        total_dist += dist_matrix[prev][0]
        if load > capacity:
            return False, None, None
        vehicle_loads.append(load)
    # check every customer visited exactly once
    if set(visited) != customers or len(visited) != len(customers):
        return False, None, None
    return True, total_dist, vehicle_loads

# =======================
# Prompt builder for VRP
# =======================
def build_vrp_prompt(coords, demands, num_vehicles, capacity, history):
    """
    coords: list of (x,y) with depot at index 0
    demands: list aligned to coords (index 0 demand 0)
    """
    n = len(coords)
    prompt_lines = []
    prompt_lines.append("Solve this Vehicle Routing Problem (VRP) with capacity constraints.")
    prompt_lines.append(f"Nodes (index, x, y): {[(i, int(x), int(y)) for i,(x,y) in enumerate(coords)]}")
    prompt_lines.append(f"Demands per node (index: demand): {list(enumerate(demands))}")
    prompt_lines.append(f"Vehicles: {num_vehicles}, Vehicle capacity: {capacity}")
    prompt_lines.append(f"Number of customers (excluding depot): {n-1}")
    prompt_lines.append("")
    prompt_lines.append("Requirements:")
    prompt_lines.append("- Return a variable named `routes` which is a list of length NUM_VEHICLES.")
    prompt_lines.append("  Each entry is a list of customer indices (integers) visited by that vehicle, in visiting order.")
    prompt_lines.append("  Do NOT include depot (0) inside each vehicle route list. The route implies depot->...->depot.")
    prompt_lines.append("- Each customer (1..n-1) MUST be visited exactly once across all vehicles.")
    prompt_lines.append("- For each vehicle, total demand of assigned customers must be <= vehicle capacity.")
    prompt_lines.append("- Minimize total distance implicitly (we'll evaluate), but you may use greedy/heuristic/dp approaches.")
    prompt_lines.append("- Do NOT print, do NOT import, do NOT define functions. Direct code only.")
    if history:
        prompt_lines.append("")
        prompt_lines.append("Recent feasible solutions (distance):")
        for r, dist in history[:3]:
            prompt_lines.append(f"- {r[:3]}... -> {dist:.2f}")
    prompt_lines.append("")
    prompt_lines.append("Return ONLY a single triple-fenced python code block. Final line must be:")
    prompt_lines.append("routes = [[...], [...], ...]   # exactly NUM_VEHICLES lists, each list of ints (customers)")
    return "\n".join(prompt_lines)

# =======================
# Execute LLM code (safe-ish) and extract `routes`
# =======================
def run_llm_code_and_get_routes(llm_output, coords, dist_matrix, demands, num_vehicles, capacity):
    if not llm_output:
        return None, None, None, False
    try:
        # extract fenced python
        if "```python" in llm_output:
            code = llm_output.split("```python",1)[1].split("```",1)[0]
        elif "```" in llm_output:
            code = llm_output.split("```",1)[1].split("```",1)[0]
        else:
            code = llm_output
        code = textwrap.dedent(code).strip()
        # Prepare restricted exec env
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
            # provide problem data in case LLM expects variables
            "coords": coords,
            "demands": demands,
            "NUM_VEHICLES": num_vehicles,
            "VEHICLE_CAPACITY": capacity,
            "n": len(coords)
        }
        try:
            exec(code, exec_env)
            print("Executed Code:\n", code)
        except Exception as e:
            # show code and error for inspection but fail gracefully
            print("Code execution error:", e)
            print("LLM Code was:\n", code)
            return None, code, None, False
        if "routes" not in exec_env:
            print("LLM did not create `routes` variable. Code was:\n", code)
            return None, code, None, False
        routes = exec_env["routes"]
        valid, total_dist, loads = evaluate_routes(routes, dist_matrix, demands, capacity, num_vehicles)
        return routes, code, (valid, total_dist, loads), valid
    except Exception as e:
        print("Parse/exec error:", e)
        return None, None, None, False

# =======================
# OR-Tools baseline VRP
# =======================
def ortools_vrp(dist_matrix, demands, num_vehicles, capacity):
    """
    Solve VRP with OR-Tools (minimize distance, respect vehicle capacity).
    Returns routes (list per vehicle) and total distance.
    """
    n = len(dist_matrix)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    # distance callback (rounded to int)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 1000)  # scale to preserve precision
    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    # capacity callback
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(demands[node])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,  # null capacity slack
        [int(capacity) for _ in range(num_vehicles)],
        True,
        "Capacity"
    )
    # search params
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = 10
    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None, None
    # extract routes
    routes = []
    total_dist = 0.0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        prev = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:  # skip depot in stored route
                route.append(node)
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            total_dist += dist_matrix[node][next_node]
            index = next_index
        routes.append(route)
    return routes, total_dist

# =======================
# OPRO loop for VRP (LLM code-gen + exec)
# =======================
def opro_vrp(coords, demands, num_vehicles, capacity, iterations, batch_size):
    dist_matrix = compute_distance_matrix(coords)
    history = []            # tuples (routes, total_dist)
    successful_codes = []   # list of dicts with code, routes, dist
    # initialize with simple heuristics
    print("Seeding with greedy splits...")
    # simple seed: greedy nearest/insertion or split by demand — provide a few feasible seeds
    customers = list(range(1, len(coords)))
    random.shuffle(customers)
    # seed 1: round-robin assignment
    seed_routes = [[] for _ in range(num_vehicles)]
    loads = [0]*num_vehicles
    idx = 0
    for c in customers:
        for attempt in range(num_vehicles):
            j = (idx + attempt) % num_vehicles
            if loads[j] + demands[c] <= capacity:
                seed_routes[j].append(c)
                loads[j] += demands[c]
                idx = j + 1
                break
    valid, td, vl = evaluate_routes(seed_routes, dist_matrix, demands, capacity, num_vehicles)
    if valid:
        history.append((seed_routes, td))
        successful_codes.append({"routes": seed_routes, "dist": td, "code": "# seed: round-robin"})
        print("Seed1 added, dist", td)
    # seed 2: greedy nearest for each vehicle (simple)
    remaining = set(customers)
    seed2 = [[] for _ in range(num_vehicles)]
    loads2 = [0]*num_vehicles
    for v in range(num_vehicles):
        current = 0
        # assign nearest until capacity
        while True:
            # find nearest remaining that fits
            cand = None
            cand_d = None
            for c in list(remaining):
                if loads2[v] + demands[c] <= capacity:
                    d = dist_matrix[current][c]
                    if cand is None or d < cand_d:
                        cand, cand_d = c, d
            if cand is None:
                break
            seed2[v].append(cand)
            loads2[v] += demands[cand]
            remaining.remove(cand)
            current = cand
    if remaining:
        # put leftovers to first vehicles that can take them
        for c in list(remaining):
            for v in range(num_vehicles):
                if loads2[v] + demands[c] <= capacity:
                    seed2[v].append(c)
                    loads2[v] += demands[c]
                    break
    valid, td2, vl2 = evaluate_routes(seed2, dist_matrix, demands, capacity, num_vehicles)
    if valid:
        history.append((seed2, td2))
        successful_codes.append({"routes": seed2, "dist": td2, "code": "# seed: nearest greedy"})
        print("Seed2 added, dist", td2)

    # OPRO iterations
    for it in range(iterations):
        print(f"\n=== Iteration {it+1} ===")
        prompt = build_vrp_prompt(coords, demands, num_vehicles, capacity, history)
        for b in range(batch_size):
            print(f" Requesting LLM (batch {b+1}/{batch_size})...")
            llm_out = groq_generate(prompt)
            routes, code, info, ok = run_llm_code_and_get_routes(llm_out, coords, compute_distance_matrix(coords), demands, num_vehicles, capacity)
            if ok:
                valid_flag, total_dist, loads = info
                history.append((routes, total_dist))
                successful_codes.append({"routes": routes, "dist": total_dist, "loads": loads, "code": code})
                print(f"  ✓ Feasible routes from LLM: dist={total_dist:.2f}, loads={loads}")
            else:
                print("  ✗ LLM produced invalid or infeasible routes.")
        # keep best few in history (prune)
        history = sorted(history, key=lambda x: x[1])[:20]  # keep top 20 best
    # best solution
    best = sorted(history, key=lambda x: x[1])[0] if history else (None, None)
    return best, successful_codes

# =======================
# Run everything
# =======================
if __name__ == "__main__":
    # build instance
    coords, demands = generate_vrp_instance(NUM_CUSTOMERS)
    dist_matrix = compute_distance_matrix(coords)
    print("Generated nodes (index, x, y):")
    for i,(x,y) in enumerate(coords):
        print(f"{i}: ({x},{y}) demand={demands[i]}")
    print(f"\nVehicles: {NUM_VEHICLES}, capacity: {VEHICLE_CAPACITY}\n")

    # Run OPRO LLM-driven VRP
    best_llm, successful_codes = opro_vrp(coords, demands, NUM_VEHICLES, VEHICLE_CAPACITY, ITERATIONS, LLM_BATCH_SIZE)
    if best_llm and best_llm[0] is not None:
        llm_routes, llm_dist = best_llm
        print("\nBest LLM solution distance:", llm_dist)
        print("Routes (per vehicle):", llm_routes)
    else:
        print("\nNo feasible LLM solution found.")

    # OR-Tools baseline
    ort_routes, ort_dist = ortools_vrp(dist_matrix, demands, NUM_VEHICLES, VEHICLE_CAPACITY)
    if ort_routes:
        print("\nOR-Tools solution distance:", ort_dist)
        print("OR-Tools routes:", ort_routes)
    else:
        print("\nOR-Tools failed to find a solution in time limit.")

    # Show top LLM codes (if any)
    if successful_codes:
        print("\nTop LLM code examples (best 3):")
        best_codes = sorted(successful_codes, key=lambda x: x["dist"])[:3]
        for i, c in enumerate(best_codes, 1):
            print(f"\nExample {i}: distance={c['dist']}, loads={c.get('loads')}")
            code_snippet = c.get("code", "")
            print("-" * 40)
            print(code_snippet[:1000])  # print first 1000 chars
            print("-" * 40)

    # Plot best LLM (if found) and OR-Tools
    plt.figure(figsize=(10,8))
    if best_llm and best_llm[0] is not None:
        routes_plot = best_llm[0]
        # convert to closed route for plotting
        for vi, route in enumerate(routes_plot):
            seq = [0] + route + [0]
            xs = [coords[i][0] for i in seq]
            ys = [coords[i][1] for i in seq]
            plt.plot(xs, ys, '-o', label=f"LLM Vehicle {vi+1}")
    if ort_routes:
        for vi, route in enumerate(ort_routes):
            seq = [0] + route + [0]
            xs = [coords[i][0] for i in seq]
            ys = [coords[i][1] for i in seq]
            plt.plot(xs, ys, '--x', label=f"OR-Tools Vehicle {vi+1}", linewidth=2)
    # draw nodes
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    plt.scatter(xs[1:], ys[1:], s=60, c='black', zorder=5)
    plt.scatter([coords[0][0]], [coords[0][1]], s=120, c='red', marker='s', label='Depot')
    for idx,(x,y) in enumerate(coords):
        plt.text(x+1, y+1, str(idx), fontsize=9)
    plt.title("VRP (LLM vs OR-Tools)")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
