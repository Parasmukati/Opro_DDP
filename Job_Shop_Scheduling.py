import numpy as np
import matplotlib.pyplot as plt
import requests, time, textwrap
import random
import itertools
import copy, math
from ortools.sat.python import cp_model
from api_key import GROQ_API_KEY_1

# -----------------
# Groq API Config
# -----------------
GROQ_API_KEY = GROQ_API_KEY_1
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_generate(prompt, retries=3, backoff=5, timeout=30):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Generate ONLY Python code for scheduling problems. No explanation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1500
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
NUM_JOBS = 3
NUM_MACHINES = 3
ITERATIONS = 6
LLM_BATCH_SIZE = 2
np.random.seed(42)

# Generate random job-shop instance
def generate_jssp_instance(num_jobs, num_machines):
    jobs = []
    for j in range(num_jobs):
        ops = []
        machines = np.random.permutation(num_machines)
        for m in machines:
            duration = np.random.randint(2, 10)
            ops.append((int(m), int(duration)))
        jobs.append(ops)
    return jobs

jobs_data = generate_jssp_instance(NUM_JOBS, NUM_MACHINES)

print("Jobs data (job -> [(machine,duration), ...]):")
for j, ops in enumerate(jobs_data):
    print(f"Job {j}:", ops)

# -----------------
# Evaluate a schedule
# -----------------
def evaluate_schedule(schedule, jobs):
    """ schedule = { (job,op): start_time } """
    makespan = 0
    machine_usage = {m: [] for m in range(NUM_MACHINES)}
    try:
        for j, ops in enumerate(jobs):
            for o, (m,d) in enumerate(ops):
                st = schedule.get((j,o))
                if st is None:
                    return None
                if not isinstance(st, (int, float)):
                    return None
                et = st + d
                # precedence
                if o>0 and schedule[(j,o-1)] + jobs[j][o-1][1] > st:
                    return None
                # overlap check
                for s,e in machine_usage[m]:
                    if not (et <= s or st >= e):
                        return None
                machine_usage[m].append((st,et))
                makespan = max(makespan, et)
    except Exception:
        return None
    return int(makespan)

# -----------------
# Prompt Builder
# -----------------
def build_jssp_prompt(jobs, history):
    lines = []
    lines.append("Solve this Job-Shop Scheduling Problem (JSSP).")
    lines.append(f"Jobs data = {jobs}  # format: job -> [(machine, duration), ...]")
    lines.append("")
    lines.append("STRICT requirements:")
    lines.append("- Use brute force or DFS/backtracking to explore operation orderings (systematic search).")
    lines.append("- Respect precedence (op k must start after op k-1 of the same job).")
    lines.append("- Respect machine capacity (one operation per machine at a time).")
    lines.append("- Compute makespan and keep the best schedule found.")
    lines.append("- At the END assign BOTH variables:")
    lines.append("    best_schedule = { (job, op): start_time, ... }")
    lines.append("    schedule = best_schedule   # (alias so the evaluator finds it)")
    lines.append("    best_makespan = <int>")
    lines.append("- The code must be complete and runnable (no unfinished functions).")
    lines.append("- Do NOT print; return ONLY a single fenced python code block.")
    lines.append("- You may import itertools if needed, but avoid other imports (the runner strips imports).")
    if history:
        lines.append("\nRecent good makespans seen:")
        for _, mk in history[:2]:
            lines.append(f"- makespan={mk}")
    return "\n".join(lines)

# -----------------
# Normalizer for LLM schedule outputs
# -----------------
def normalize_schedule_candidate(candidate, jobs):
    """
    Accept schedule in several LLM-returned shapes and convert to {(j,o):start} dict.
    Handles:
      - dict with tuple keys (ideal)
      - dict with string keys like '(0,1)' or '0,1'
      - dict with string keys like 'job0_op1' or '0_1'
      - list of triples [(j,o,start), ...]
    Returns normalized dict or None.
    """
    if candidate is None:
        return None
    # if already correct
    if isinstance(candidate, dict):
        # tuple-keys quick path
        tuple_keys = all((isinstance(k, tuple) and len(k)==2) or isinstance(k, (str, int)) or isinstance(k, list) for k in candidate.keys())
        if tuple_keys:
            parsed = {}
            for k,v in candidate.items():
                nk = None
                if isinstance(k, tuple) and len(k)==2:
                    nk = (int(k[0]), int(k[1]))
                elif isinstance(k, list) and len(k)==2:
                    nk = (int(k[0]), int(k[1]))
                elif isinstance(k, int):
                    # ambiguous single int key not supported
                    return None
                elif isinstance(k, str):
                    s = k.strip()
                    if s.startswith("(") and s.endswith(")"):
                        s2 = s[1:-1]
                    else:
                        s2 = s
                    if "," in s2:
                        parts = [p.strip() for p in s2.split(",")]
                        if len(parts)==2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                            nk = (int(parts[0]), int(parts[1]))
                    if nk is None:
                        if "_" in s2:
                            parts = s2.split("_")
                            # try patterns: "0_1" or "job0_op1"
                            if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit():
                                nk = (int(parts[0]), int(parts[1]))
                            else:
                                # try job0_op1
                                try:
                                    if "job" in s2 and "op" in s2:
                                        a = s2.split("job")[-1]
                                        j = int(a.split("_")[0])
                                        o = int(a.split("op")[-1])
                                        nk = (j,o)
                                except Exception:
                                    nk = None
                    if nk is None:
                        return None
                else:
                    return None
                # parse value
                if isinstance(v, (int, float)):
                    parsed[nk] = int(v)
                elif isinstance(v, str):
                    try:
                        parsed[nk] = int(float(v))
                    except Exception:
                        return None
                else:
                    return None
            return parsed if parsed else None
    # list of triples
    if isinstance(candidate, list):
        parsed = {}
        for item in candidate:
            if isinstance(item, (list,tuple)) and len(item)==3:
                j,o,s = item
                try:
                    parsed[(int(j),int(o))] = int(s)
                except Exception:
                    return None
            else:
                return None
        return parsed if parsed else None
    return None

# -----------------
# Helper: strip top-level imports (avoid __import__ calls from LLMs)
# -----------------
def strip_import_lines(code):
    lines = code.splitlines()
    kept = []
    for ln in lines:
        stripped = ln.strip()
        # skip top-level import/from lines
        if stripped.startswith("import ") or stripped.startswith("from "):
            # we silently remove imports; commonly used modules are injected into exec_env
            continue
        kept.append(ln)
    return "\n".join(kept)

# -----------------
# Secure/extraction runner for LLM code -> schedule + makespan
# -----------------
def run_llm_code_and_get_schedule(llm_output, jobs):
    """
    Execute LLM code in a restricted env, accept variables:
      - best_schedule & best_makespan
      - schedule (legacy)
      - result/sol (dict)
    Normalize formats and validate with evaluate_schedule.
    Returns: (normalized_schedule_dict, code_str, makespan_or_False)
    """
    if not llm_output:
        return None, None, False

    # prefer fenced python
    if "```python" in llm_output:
        code = llm_output.split("```python",1)[1].split("```",1)[0]
    elif "```" in llm_output:
        code = llm_output.split("```",1)[1].split("```",1)[0]
    else:
        code = llm_output
    code = textwrap.dedent(code).strip()

    # remove top-level imports to avoid __import__ errors (we inject safe modules)
    safe_code = strip_import_lines(code)

    # Minimal safe builtins
    safe_builtins = {
        'range': range, 'len': len, 'int': int, 'float': float, 'max': max, 'min': min,
        'sum': sum, 'enumerate': enumerate, 'list': list, 'tuple': tuple, 'abs': abs, 'bool': bool,
        'dict': dict, 'set': set
    }

    # Pre-injected trusted helper objects (LLM may reference them)
    exec_env = {
        '__builtins__': safe_builtins,
        'jobs': jobs,
        'itertools': itertools,
        'copy': copy,
        'deepcopy': copy.deepcopy,
        'math': math
    }

    try:
        exec(safe_code, exec_env)
    except Exception as e:
        print("Exec error:", e)
        print("LLM Code (truncated):\n", ("\n".join(code.splitlines()[:200])))
        return None, code, False

    # Candidate schedule detection
    candidate = None
    cand_mk = None

    if 'best_schedule' in exec_env and 'best_makespan' in exec_env:
        candidate = exec_env['best_schedule']
        cand_mk = exec_env['best_makespan']
    elif 'schedule' in exec_env:
        candidate = exec_env['schedule']
        cand_mk = exec_env.get('makespan', None) or exec_env.get('best_makespan', None)
    elif 'result' in exec_env and isinstance(exec_env['result'], dict):
        candidate = exec_env['result']
    elif 'sol' in exec_env and isinstance(exec_env['sol'], dict):
        candidate = exec_env['sol']

    if candidate is None:
        print("LLM did not set expected variables (best_schedule/schedule). Code was (first 1200 chars):\n", code[:1200])
        return None, code, False

    normalized = normalize_schedule_candidate(candidate, jobs)
    if normalized is None:
        print("Could not normalize LLM schedule. Candidate type:", type(candidate))
        # Show small preview to help debugging
        try:
            import json
            preview = str(list(candidate.items())[:10])
            print("Candidate preview:", preview)
        except Exception:
            pass
        return None, code, False

    # validate schedule
    mk = evaluate_schedule(normalized, jobs)
    if mk is None:
        # if LLM provided makespan and schedule format looked parseable, print mismatch for debug
        if isinstance(cand_mk, (int,float)):
            print("LLM schedule failed validation (overlaps/precedence). Candidate makespan variable:", cand_mk)
        else:
            print("LLM schedule failed validation (overlaps/precedence).")
        return normalized, code, False

    return normalized, code, int(mk)

# -----------------
# OR-Tools Baseline
# -----------------
def ortools_jssp(jobs):
    model = cp_model.CpModel()
    horizon = sum(d for job in jobs for _,d in job)
    all_tasks = {}
    all_machines = {}
    for j, job in enumerate(jobs):
        for o, (m,d) in enumerate(job):
            all_tasks[(j,o)] = model.NewIntVar(0,horizon,f"start_{j}_{o}")
            end = model.NewIntVar(0,horizon,f"end_{j}_{o}")
            interval = model.NewIntervalVar(all_tasks[(j,o)], d, end, f"int_{j}_{o}")
            all_machines.setdefault(m,[]).append(interval)
            if o>0:
                model.Add(all_tasks[(j,o)] >= all_tasks[(j,o-1)] + jobs[j][o-1][1])
    for m,intervals in all_machines.items():
        model.AddNoOverlap(intervals)
    obj = model.NewIntVar(0,horizon,"makespan")
    for j,job in enumerate(jobs):
        model.Add(obj >= all_tasks[(j,len(job)-1)] + job[-1][1])
    model.Minimize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        schedule = {(j,o):solver.Value(all_tasks[(j,o)]) for j,job in enumerate(jobs) for o in range(len(job))}
        return schedule, solver.ObjectiveValue()
    return None,None

# -----------------
# OPRO Loop
# -----------------
def opro_jssp(jobs,iters,batch):
    history,successes = [],[]
    for it in range(iters):
        print(f"\nIteration {it+1}/{iters}")
        prompt = build_jssp_prompt(jobs,history)
        for b in range(batch):
            llm_out = groq_generate(prompt)
            if not llm_out:
                print(" ✗ No LLM response")
                continue
            sch, code, mk = run_llm_code_and_get_schedule(llm_out,jobs)
            if mk and mk>0:
                history.append((sch,mk))
                successes.append({"schedule":sch,"mk":mk,"code":code})
                print(f" ✓ Feasible LLM schedule mk={mk}")
            else:
                print(" ✗ Invalid LLM schedule")
        history = sorted(history,key=lambda x:x[1])[:10]
    best = min(history,key=lambda x:x[1]) if history else (None,None)
    return best,successes

# -----------------
# Run (main)
# -----------------
if __name__=="__main__":
    best_llm,codes = opro_jssp(jobs_data,ITERATIONS,LLM_BATCH_SIZE)
    if best_llm[0]:
        print("\nBest LLM makespan:",best_llm[1])
        print("Best LLM schedule:", best_llm[0])
        print("Best LLM code (truncated):\n", ("\n".join(codes[0]['code'].splitlines()[:20])))
    ort_sch,ort_mk = ortools_jssp(jobs_data)
    print("\nOR-Tools makespan:",ort_mk)

    fig, axes = plt.subplots(1,2,figsize=(14,5),sharey=True)
    colors = ["tab:blue","tab:orange","tab:green","tab:red"]

    def plot_schedule(ax, schedule, title):
        if not schedule: 
            ax.set_title(title+" (no schedule)")
            return
        for j,job in enumerate(jobs_data):
            for o,(m,d) in enumerate(job):
                st = schedule.get((j,o))
                if st is None: continue
                ax.barh(m,d,left=st,color=colors[j%len(colors)],edgecolor="black")
                ax.text(st+d/2,m,str(j),va="center",ha="center",color="white")
        ax.set_yticks(range(NUM_MACHINES))
        ax.set_yticklabels([f"M{m}" for m in range(NUM_MACHINES)])
        ax.set_xlabel("Time")
        ax.set_title(title)

    plot_schedule(axes[0], ort_sch, f"OR-Tools (mk={ort_mk})")
    plot_schedule(axes[1], best_llm[0], f"OPRO LLM (mk={best_llm[1]})" if best_llm[1] else "OPRO LLM")

    plt.tight_layout()
    plt.show()
