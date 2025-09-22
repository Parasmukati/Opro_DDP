# ============================
# OPRO Assignment Problem (AP)
# ============================
import numpy as np
import matplotlib.pyplot as plt
import requests, time, textwrap, random
from scipy.optimize import linear_sum_assignment  # baseline Hungarian
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
            {"role": "system", "content": "Generate ONLY Python code for assignment problems. No explanation."},
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
NUM_WORKERS = 6
NUM_JOBS = 6
ITERATIONS = 6
LLM_BATCH_SIZE = 2
np.random.seed(42)

def generate_assignment_instance(n,m):
    return np.random.randint(5,50,(n,m)).tolist()

cost_matrix = generate_assignment_instance(NUM_WORKERS, NUM_JOBS)
print("Cost matrix:")
for row in cost_matrix:
    print(row)

# -----------------
# Evaluation
# -----------------
def evaluate_assignment(assign, cost_matrix):
    """ assign = list of tuples (worker,job) """
    n,m = len(cost_matrix), len(cost_matrix[0])
    if not isinstance(assign, list): return False,None
    workers = set(); jobs = set(); total = 0
    for a in assign:
        if not (isinstance(a,(list,tuple)) and len(a)==2): return False,None
        w,j = a
        if w<0 or w>=n or j<0 or j>=m: return False,None
        if w in workers or j in jobs: return False,None
        workers.add(w); jobs.add(j)
        total += cost_matrix[w][j]
    if len(assign)!=n: return False,None
    return True,total

# -----------------
# Prompt Builder
# -----------------
def build_assignment_prompt(cost_matrix, history):
    lines = []
    lines.append("Solve this Assignment Problem (minimize total cost).")
    lines.append(f"Cost matrix = {cost_matrix}  # rows=workers, cols=jobs")
    lines.append("")
    lines.append("STRICT requirements:")
    lines.append("- Assign each worker to exactly one distinct job.")
    lines.append("- Each job must be assigned to exactly one worker.")
    lines.append("- Compute total cost and keep the best solution found.")
    lines.append("- Do NOT use pulp, ortools, or any external libraries.")
    lines.append("- You may use brute force, DFS, backtracking, or Hungarian algorithm (implemented manually).")
    lines.append("- Return a variable named `assignment = [(worker,job), ...]` with all pairs.")
    if history:
        lines.append("\nRecent feasible solutions (costs):")
        for _,cost in history[:2]:
            lines.append(f"- cost={cost}")
    return "\n".join(lines)

# -----------------
# Runner for LLM code
# -----------------
def run_llm_code_and_get_assignment(llm_output, cost_matrix):
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
        exec_env = {"__builtins__": safe_builtins, "cost_matrix": cost_matrix}
        exec(code, exec_env)
        if "assignment" not in exec_env: return None,code,False
        assign = exec_env["assignment"]
        valid,total = evaluate_assignment(assign,cost_matrix)
        if not valid: return assign,code,False
        return assign,code,total
    except Exception as e:
        print("Exec error:", e)
        return None,None,False

# -----------------
# OR-Tools Baseline (Hungarian via SciPy)
# -----------------
def scipy_assignment(cost_matrix):
    cm = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cm)
    assign = [(int(r),int(c)) for r,c in zip(row_ind,col_ind)]
    total = cm[row_ind,col_ind].sum()
    return assign, int(total)

# -----------------
# OPRO Loop
# -----------------
def opro_assignment(cost_matrix,iters,batch):
    history,successes = [],[]
    for it in range(iters):
        print(f"\nIteration {it+1}/{iters}")
        prompt = build_assignment_prompt(cost_matrix,history)
        for b in range(batch):
            llm_out = groq_generate(prompt)
            assign, code, total = run_llm_code_and_get_assignment(llm_out,cost_matrix)
            if total:
                history.append((assign,total))
                successes.append({"assign":assign,"cost":total,"code":code})
                print(f" ✓ Feasible LLM assignment: cost={total}")
            else:
                print(" ✗ Invalid LLM assignment")
        history = sorted(history,key=lambda x:x[1])[:10]
    best = min(history,key=lambda x:x[1]) if history else (None,None)
    return best,successes

# -----------------
# Run
# -----------------
if __name__=="__main__":
    best_llm,codes = opro_assignment(cost_matrix,ITERATIONS,LLM_BATCH_SIZE)
    if best_llm[0]:
        print("\nBest LLM solution cost:",best_llm[1])
        print("Assignment:",best_llm[0])
        print("code:\n",codes[0]["code"])
    ort_assign, ort_cost = scipy_assignment(cost_matrix)
    print("\nHungarian (SciPy) cost:",ort_cost)

    # --- Plot comparison ---
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    cm = np.array(cost_matrix)
    def plot_assign(ax, assign, cost, title):
        ax.imshow(cm,cmap="Blues")
        for (w,j) in assign:
            ax.scatter(j,w,c="red",marker="x",s=100)
        ax.set_title(f"{title}\nCost={cost}")
        ax.set_xlabel("Jobs"); ax.set_ylabel("Workers")
    if ort_assign:
        plot_assign(axes[0], ort_assign, ort_cost, "Hungarian (Optimal)")
    if best_llm[0]:
        plot_assign(axes[1], best_llm[0], best_llm[1], "OPRO LLM")
    plt.tight_layout(); plt.show()
