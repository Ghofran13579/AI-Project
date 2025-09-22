import pandas as pd
from pathlib import Path
import os
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_FOLDER = r"C:\Users\Massaoudi\Desktop\output_equitable"
OUTPUT_FOLDER = r"C:\Users\Massaoudi\Desktop\initial_populations_v_Q-NDSA_v3"
NUM_SOLUTIONS = 100
NURSES = 4
PATIENTS = 13

# Example Parameters (replace with your real data!)
# Travel cost c[n][i][j], Travel time T[n][i][j], Service time s[i], Overtime cost d[n], Max working time L[n]
c = np.random.randint(1, 20, size=(NURSES, PATIENTS+2, PATIENTS+2))
T = np.random.randint(5, 30, size=(NURSES, PATIENTS+2, PATIENTS+2))
s = np.random.randint(10, 60, size=(PATIENTS+2,))
d = np.random.randint(10, 50, size=(NURSES,))
L = np.full(NURSES, 480)  # max 8 hours in minutes
q = np.random.randint(1, 6, size=(NURSES, PATIENTS+1))
S = np.random.randint(1, 6, size=(NURSES, PATIENTS+1))

# Ensure output folder exists
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# =========================
# FUNCTION: Generate initial population
# =========================
def generate_population(J, N, num_solutions=100):
    population = []
    delimiters = [-i for i in range(1, N+1)]
    for _ in range(num_solutions):
        jobs = list(range(1, J+1))
        np.random.shuffle(jobs)
        cuts = sorted(np.random.choice(range(1, J), N-1, replace=False))
        groups = []
        last = 0
        for c in cuts:
            groups.append(jobs[last:c])
            last = c
        groups.append(jobs[last:])
        solution = []
        for d, g in zip(delimiters, groups):
            solution.append(d)
            solution.extend(g)
        population.append(solution)
    return population

# =========================
# FUNCTION: Evaluate population according to HHC model
# =========================
def evaluate_population(pop):
    values = []
    for sol in pop:
        F1 = 0  # Travel + Overtime cost
        F2 = 0  # Total workload (travel + service)
        F3 = 0  # Satisfaction

        nurse_indices = [i for i, val in enumerate(sol) if val < 0]
        nurse_indices.append(len(sol))  # Add sentinel for last nurse

        for idx in range(len(nurse_indices)-1):
            n = -sol[nurse_indices[idx]] - 1  # nurse index 0..N-1
            jobs = sol[nurse_indices[idx]+1:nurse_indices[idx+1]]
            
            time_worked = 0
            prev_node = 0  # start from depot 0
            for job in jobs:
                # Travel cost and time
                F1 += c[n][prev_node][job]
                time_worked += T[n][prev_node][job] + s[job]
                prev_node = job
                # Satisfaction
                F3 += q[n][job] - S[n][job]
            # Return to depot
            F1 += c[n][prev_node][PATIENTS+1]
            time_worked += T[n][prev_node][PATIENTS+1]

            # Overtime
            if time_worked > L[n]:
                F1 += d[n]*(time_worked - L[n])
            F2 += time_worked

        values.append((F1, F2, -F3))  # maximize satisfaction -> minimize -F3
    return values

# =========================
# Q-NDSA Helpers
# =========================
def dominates(a, b):
    if a[0] < b[0]: return True
    if a[0] > b[0]: return False
    if a[1] < b[1]: return True
    if a[1] > b[1]: return False
    return a[2] > b[2]

def fast_non_dominated_sort(values):
    S = [[] for _ in range(len(values))]
    n = [0]*len(values)
    rank = [0]*len(values)
    fronts = [[]]
    for p in range(len(values)):
        for q in range(len(values)):
            if dominates(values[p], values[q]):
                S[p].append(q)
            elif dominates(values[q], values[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 1
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 2
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return rank, fronts[:-1]

def crowding_distance(front, values):
    distance = [0.0]*len(values)
    if not front: return distance
    num_objectives = len(values[0])
    for m in range(num_objectives):
        front_sorted = sorted(front, key=lambda idx: values[idx][m])
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        min_val = values[front_sorted[0]][m]
        max_val = values[front_sorted[-1]][m]
        if max_val == min_val: continue
        for i in range(1, len(front_sorted)-1):
            prev_val = values[front_sorted[i-1]][m]
            next_val = values[front_sorted[i+1]][m]
            distance[front_sorted[i]] += (next_val - prev_val) / (max_val - min_val)
    return distance

# =========================
# MAIN: Process all instances
# =========================
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".csv"):
        inst_name = Path(file).stem
        print(f"\n=== Processing instance: {inst_name} ===")
        pop = generate_population(PATIENTS, NURSES, NUM_SOLUTIONS)
        df_raw = pd.DataFrame(pop)
        df_raw.to_csv(os.path.join(OUTPUT_FOLDER, f"population_raw_{inst_name}.csv"),
                      index=False, sep=';', header=[f"pos{i+1}" for i in range(PATIENTS+NURSES)])
        print(f"✅ Raw population saved for {inst_name}")

        values = evaluate_population(pop)
        ranks, fronts = fast_non_dominated_sort(values)

        distances = [0.0]*len(values)
        for front in fronts:
            d = crowding_distance(front, values)
            for idx, dist in zip(front, d):
                distances[idx] = dist

        df_eval = df_raw.copy()
        df_eval["F1"] = [v[0] for v in values]
        df_eval["F2"] = [v[1] for v in values]
        df_eval["F3"] = [-v[2] for v in values]  # convert back to positive satisfaction
        df_eval["Rank"] = ranks
        df_eval["CrowdingDistance"] = distances

        headers = [f"pos{i+1}" for i in range(PATIENTS+NURSES)] + ["F1","F2","F3","Rank","CrowdingDistance"]
        df_eval.to_csv(os.path.join(OUTPUT_FOLDER, f"population_QNDSA_{inst_name}.csv"),
                       index=False, sep=';', float_format="%.2f", header=headers)
        print(f"✅ Q-NDSA ranked population saved for {inst_name}")
