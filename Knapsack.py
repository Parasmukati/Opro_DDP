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
# GROQ_MODEL = "llama3-8b-8192"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5):
    """Send prompt to Groq API and return generated text with retry on 429."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates Python code for optimization problems. Generate ONLY the core algorithm code without function definitions or print statements."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 600
    }
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload)
            if r.status_code == 429:  # Rate limit
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
# Knapsack Config
# =======================
CAPACITY = 50
NUM_ITEMS = 8
ITERATIONS = 10
LLM_BATCH_SIZE = 3

def generate_items(num_items):
    np.random.seed(42)
    return pd.DataFrame({
        "item": [f"Item{i+1}" for i in range(num_items)],
        "weight": np.random.randint(5, 20, size=num_items),
        "value": np.random.randint(30, 120, size=num_items)
    })

def cp_sat_knapsack(weights, values, capacity):
    model = cp_model.CpModel()
    n = len(weights)
    x = [model.NewBoolVar(f'x_{i}') for i in range(n)]
    model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
    model.Maximize(sum(values[i] * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selection = [1 if solver.Value(x[i]) else 0 for i in range(n)]
        total_value = sum(values[i] * selection[i] for i in range(n))
        return selection, total_value
    else:
        return [0]*n, 0

def evaluate_solution(solution, items, capacity):
    """Evaluate a knapsack solution and return its value."""
    if not isinstance(solution, list) or len(solution) != len(items):
        return 0
    if not all(x in [0, 1] for x in solution):
        return 0
    
    total_weight = sum(w for w, take in zip(items["weight"], solution) if take)
    total_value = sum(v for v, take in zip(items["value"], solution) if take)
    
    if total_weight > capacity:
        return 0
    return total_value

def build_simple_code_prompt(items, capacity, history):
    py_items = [(int(w), int(v)) for w, v in zip(items["weight"], items["value"])]
    n = len(py_items)
    prompt = []
    prompt.append("Solve this 0/1 knapsack problem using dynamic programming EXACTLY.")
    prompt.append(f"capacity = {capacity}")
    prompt.append(f"items_data = {py_items}  # format: list of (weight, value)")
    prompt.append(f"n = {n}  # number of items")
    prompt.append("")
    prompt.append("Requirements:")
    prompt.append("- Implement full DP table computation.")
    prompt.append("- Reconstruct the chosen items (solution) from the DP table properly.")
    prompt.append("- Define ONLY a variable named 'solution' which is a list of length n with 0/1.")
    prompt.append("- Do NOT print, import, or define functions. Write straight-line code.")
    prompt.append("- Your code must produce the OPTIMAL solution exactly.")
    prompt.append("")
    if history:
        prompt.append("Previous attempts and their objective values:")
        for sol, val in history[:5]:
            prompt.append(f"- {sol} → value {val}")
    prompt.append("")
    prompt.append("Return ONLY python code enclosed in triple backticks like this:")
    prompt.append("```")
    prompt.append("# dynamic programming knapsack solution")
    prompt.append("solution = *n  # replace this with your actual code")
    prompt.append("```")
    return "\n".join(prompt)

def execute_direct_code(llm_output, items, capacity, max_retries=3):
    """
    Execute LLM code safely with auto-fix retries.
    Returns (solution, code, success).
    """
    if not llm_output:
        return None, None, False
    
    # Extract code section
    if "```python" in llm_output:
        code_section = llm_output.split("```python")[1].split("```")[0]
    elif "```" in llm_output:
        code_section = llm_output.split("```")[1].split("```")[0]
    else:
        code_section = llm_output
    
    code = textwrap.dedent(code_section).strip()
    
    # Prepare environment
    items_data = [(int(row['weight']), int(row['value'])) for _, row in items.iterrows()]
    n = len(items)
    safe_builtins = {
        "range": range, "len": len, "max": max, "min": min, "enumerate": enumerate,
        "sum": sum, "list": list, "int": int, "float": float, "abs": abs,
        "all": all, "any": any, "sorted": sorted, "zip": zip, "bool": bool
    }
    
    exec_env = {
        "__builtins__": safe_builtins,
        "items_data": items_data,
        "capacity": int(capacity),
        "n": n
    }
    
    # Retry loop with fixes
    for attempt in range(max_retries):
        try:
            exec(code, exec_env)
            print(code)
            if "solution" in exec_env:
                sol = exec_env["solution"]
                if isinstance(sol, list) and len(sol) == len(items) and all(x in [0,1] for x in sol):
                    return sol, code, True
                else:
                    print(f"⚠️ Invalid solution format: {sol}")
            else:
                print("⚠️ No 'solution' found, attempting fix...")
                raise RuntimeError("solution missing")
        
        except Exception as e:
            print(f"❌ Execution error (attempt {attempt+1}): {e}")
            
            # --- Auto Fix Rules ---
            fixed = code
            if "print(" in fixed:
                fixed = "\n".join(line for line in fixed.splitlines() if "print(" not in line)
                print("🔧 Removed print statements")
            
            if "import " in fixed:
                fixed = "\n".join(line for line in fixed.splitlines() if not line.strip().startswith("import "))
                print("🔧 Removed import statements")
            
            if "def " in fixed:
                # Try to auto-call function at end
                func_name = None
                for line in fixed.splitlines():
                    if line.strip().startswith("def "):
                        func_name = line.strip().split("def ")[1].split("(")[0]
                        break
                if func_name:
                    fixed += f"\nsolution = {func_name}(items_data, capacity)"
                    print(f"🔧 Added call to {func_name}()")
            
            if "solution" not in fixed:
                fixed += f"\nsolution = [0]*n"
                print("🔧 Added default solution assignment")
            
            code = fixed  # replace with fixed version
            continue  # retry
    
    return None, code, False


# =======================
# OPRO with Bulletproof Execution
# =======================
def opro_knapsack(items, capacity, iterations, batch_size):
    """
    Bulletproof OPRO implementation that actually executes LLM code correctly.
    """
    history = []
    successful_codes = []
    
    # Initialize with random solutions
    print("🎯 Initializing with random solutions...")
    for i in range(batch_size):
        sol = [random.choice([0, 1]) for _ in range(len(items))]
        val = evaluate_solution(sol, items, capacity)
        history.append((sol, val))
        print(f"   Random {i+1}: {sol} → value {val}")
    
    history.sort(key=lambda x: -x[1])
    best_values = []
    
    for step in range(iterations):
        print(f"\n{'🔥 ITERATION ' + str(step+1):=^60}")
        
        # Build focused prompt
        prompt = build_simple_code_prompt(items, capacity, history[:3])
        
        iteration_solutions = []
        
        for batch_idx in range(batch_size):
            print(f"\n💡 Batch {batch_idx+1}/{batch_size}")
            
            # Get LLM response
            llm_output = groq_generate(prompt)
            
            if llm_output:
                # Execute the code
                solution, code, success = execute_direct_code(llm_output, items, capacity)
                
                if success and solution:
                    value = evaluate_solution(solution, items, capacity)
                    iteration_solutions.append((solution, value))
                    successful_codes.append({
                        'iteration': step + 1,
                        'batch': batch_idx + 1,
                        'code': code,
                        'solution': solution,
                        'value': value
                    })
                    
                    print(f"   ✅ SUCCESS: {solution} → value {value}")
                    
                    # Show selected items for interesting solutions
                    if value > 300:
                        selected = [f"Item{i+1}" for i, x in enumerate(solution) if x]
                        print(f"   📦 Selected: {selected}")
                else:
                    print(f"   ❌ FAILED to generate valid solution")
                    if code:
                        print(f"   📝 Code preview: {code[:100]}...")
            else:
                print(f"   ❌ No LLM response received")
        
        # Update history
        history.extend(iteration_solutions)
        history.sort(key=lambda x: -x[1])
        
        current_best = history[0][1]
        best_values.append(current_best)
        
        print(f"\n📊 Best after iteration {step+1}: {current_best}")
        print(f"🎯 Current best solution: {history[0][0]}")
    
    return history[0], best_values, successful_codes

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    print("🚀 OPRO KNAPSACK - BULLETPROOF VERSION")
    print("="*70)
    
    # Generate problem
    items_df = generate_items(NUM_ITEMS)
    print("📋 Problem Instance:")
    for i, row in items_df.iterrows():
        ratio = row['value'] / row['weight']
        print(f"   {row['item']}: weight={row['weight']:2d}, value={row['value']:3d}, ratio={ratio:.2f}")
    print(f"\n🎒 Capacity: {CAPACITY} kg")
    
    # Run OPRO
    print(f"\n🔥 Running OPRO: {ITERATIONS} iterations × {LLM_BATCH_SIZE} attempts each")
    best_solution, progression, successful_attempts = opro_knapsack(
        items_df, CAPACITY, ITERATIONS, LLM_BATCH_SIZE
    )
    
    # Get optimal solution for comparison
    weights = items_df['weight'].tolist()
    values = items_df['value'].tolist()
    optimal_sol, optimal_val = cp_sat_knapsack(weights, values, CAPACITY)
    
    # Final results
    print(f"\n{'🎉 FINAL RESULTS':=^70}")
    print(f"🥇 OPRO Best Solution: {best_solution[0]}")
    print(f"💰 OPRO Best Value: {best_solution[1]}")
    print(f"🎯 Optimal Solution: {optimal_sol}")
    print(f"⭐ Optimal Value: {optimal_val}")
    
    gap = optimal_val - best_solution[1]
    gap_pct = (gap / optimal_val * 100) if optimal_val > 0 else 0
    print(f"📈 Gap: {gap} ({gap_pct:.1f}%)")
    
    print(f"📊 Value Progression: {progression}")
    
    # Show successful code examples
    if successful_attempts:
        print(f"\n{'💻 SUCCESSFUL CODE EXAMPLES':=^70}")
        
        # Show top 3 performing codes
        top_codes = sorted(successful_attempts, key=lambda x: x['value'], reverse=True)[:3]
        
        for i, attempt in enumerate(top_codes, 1):
            print(f"\n🏆 Example {i} (Value: {attempt['value']})")
            print(f"   📍 Iteration {attempt['iteration']}, Batch {attempt['batch']}")
            print("   💻 Code:")
            print("   " + "-" * 50)
            for line in attempt['code'].split('\n'):
                print(f"   {line}")
            print("   " + "-" * 50)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(progression) + 1), progression, 
             'b-o', label="OPRO LLM", markersize=8, linewidth=3)
    plt.axhline(y=optimal_val, color='r', linestyle='--', 
                label=f'Optimal ({optimal_val})', linewidth=2)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Best Value Found", fontsize=14)
    plt.title("OPRO LLM Performance vs Optimal Solution", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    success_rate = len(successful_attempts) / (ITERATIONS * LLM_BATCH_SIZE) * 100
    print(f"\n📈 Success Rate: {len(successful_attempts)}/{ITERATIONS * LLM_BATCH_SIZE} ({success_rate:.1f}%)")
    print("🎯 Experiment Complete!")


