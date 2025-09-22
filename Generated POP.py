import pandas as pd
import random
from pathlib import Path
import os
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_FOLDER = r"C:\Users\Massaoudi\Desktop\output_equitable"  # folder with CSV instances
OUTPUT_FOLDER = r"C:\Users\Massaoudi\Desktop\initial_populations_v_Q-NDSA_v2"
NUM_SOLUTIONS = 100
NURSES = 4
PATIENTS = 13

# Ensure output folder exists
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# =========================
# FUNCTION: Generate initial population
# =========================
def generate_population(J, N, num_solutions=100):
    population = []
    delimiters = [-i for i in range(1, N+1)]  # nurse delimiters (-1, -2, ...)

    for _ in range(num_solutions):
        jobs = list(range(1, J+1))
        random.shuffle(jobs)

        cuts = sorted(random.sample(range(1, J), N-1))
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
# FUNCTION: Evaluate population
# =========================
def evaluate_population(pop):
    values = []
    for sol in pop:
        f1 = round(random.uniform(0, 100), 2)
        f2 = round(random.uniform(0, 100), 2)
        f3 = round(random.uniform(0, 100), 2)
        values.append((f1, f2, f3))
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

        # Step 1: Generate random solutions
        pop = generate_population(PATIENTS, NURSES, NUM_SOLUTIONS)
        df_raw = pd.DataFrame(pop)
        # Save raw population
        df_raw.to_csv(
            os.path.join(OUTPUT_FOLDER, f"population_raw_{inst_name}.csv"),
            index=False,
            header=[f"pos{i+1}" for i in range(PATIENTS+NURSES)],
            sep=';'  # ensures proper column separation
        )
        print(f"✅ Raw population saved for {inst_name}")

        # Step 2: Evaluate solutions
        values = evaluate_population(pop)

        # Step 3: Apply Q-NDSA ranking
        ranks, fronts = fast_non_dominated_sort(values)

        # Step 4: Compute crowding distance
        distances = [0.0]*len(values)
        for front in fronts:
            d = crowding_distance(front, values)
            for idx, dist in zip(front, d):
                distances[idx] = dist

        # Step 5: Save evaluated and ranked population
        df_eval = df_raw.copy()
        df_eval["F1"] = [v[0] for v in values]
        df_eval["F2"] = [v[1] for v in values]
        df_eval["F3"] = [v[2] for v in values]
        df_eval["Rank"] = ranks
        df_eval["CrowdingDistance"] = distances

        # Correct headers to include all columns
        headers = [f"pos{i+1}" for i in range(PATIENTS+NURSES)] + ["F1","F2","F3","Rank","CrowdingDistance"]

        df_eval.to_csv(
            os.path.join(OUTPUT_FOLDER, f"population_QNDSA_{inst_name}.csv"),
            index=False,
            sep=';',  # use semicolon for Excel compatibility
            float_format="%.2f",
            header=headers
        )
        print(f"✅ Q-NDSA ranked population saved for {inst_name}")
