Classic 0/1 Knapsack and multi-constraint knapsack: llama-3.3-70b-versatile

Selected items with maximum total value under weight and additional constraints, validating LLM-generated solutions against OR-Tools DP solver.

  
  
Bounded Knapsack (limited copies of each item): llama-3.3-70b-versatile

Solved item selection where each item has a limited number of copies, comparing LLM output with dynamic programming results.


Multi-dimensional Knapsack (MDKP) (multiple constraints like weight, volume, cost): llama-3.3-70b-versatile

Optimized item selection across multiple resource constraints (e.g., weight, cost, volume) using LLM-guided heuristics and baseline solvers.


Multi-objective Knapsack (maximize value, minimize risk simultaneously): llama-3.3-70b-versatile

Balanced trade-off between maximizing value and minimizing risk simultaneously, with Pareto-front style evaluation of LLM solutions.


Travelling Salesman Problem (TSP) (shortest route visiting all cities): llama-3.3-70b-versatile

Generated optimal/near-optimal tours visiting all cities once and returning to start, validating LLM designs against OR-Tools solver.


Vehicle Routing Problem (VRP) (multiple vehicles, capacity constraints):

Designed routes for multiple vehicles with capacity constraints, exploring extensions like VRPTW and VRPPD via LLM-generated strategies.

Llama-3.3-70b-versatile

openai/gpt-oss-120b


Job-Shop Scheduling (machines and tasks with time constraints): openai/gpt-oss-120b

Scheduled tasks across multiple machines with precedence constraints, comparing LLM DFS/backtracking schedules with OR-Tools CP-SAT baselines.


Bin Packing Problem (minimize number of bins for items): openai/gpt-oss-120b

Packed items of varying sizes into the minimum number of bins, benchmarking LLM-produced heuristics against known greedy approximations.

Result without import PuLP for solving by LLM code

Result with import PuLP for solving by LLM code


Assignment Problem (assign workers to jobs with cost minimization): openai/gpt-oss-120b

Assigned workers to jobs with minimum total cost using Hungarian-like LLM algorithms, restricted from using external solvers like PuLP.


Generalized Assignment Problem (GAP) (resources assigned to tasks with capacities): openai/gpt-oss-120b

Allocated resources to tasks under capacity limits, tested LLM code on respecting feasibility while minimizing cost.


Maximum Flow / Minimum Cut: openai/gpt-oss-120b

Computed maximum feasible flow between source and sink in a network, verifying LLM augmenting-path solutions against classical algorithms.


Minimum Cost Flow: openai/gpt-oss-120b

Extended flow problem to minimize cost of sending flow through network, prompting LLM to generate Bellman-Ford/SPFA style solvers.


Multi-commodity Flow: openai/gpt-oss-120b

Handled simultaneous flows for multiple source-sink pairs, ensuring capacity sharing across commodities using LLM-guided heuristics.


Network Design Optimization (maximize reliability under budget): openai/gpt-oss-120b

Designed cost-constrained networks maximizing reliability (connectivity under random edge failures), combining greedy baselines with OPRO + LLM improvements.

