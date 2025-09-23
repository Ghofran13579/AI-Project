import pandas as pd
from pathlib import Path
import os
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_FOLDER = r"C:\Users\Massaoudi\Desktop\output_equitable"
OUTPUT_FOLDER = r"C:\Users\Massaoudi\Desktop\initial_populations_v_Q-NDSA_corrected"
NUM_SOLUTIONS = 100
NURSES = 4
PATIENTS = 13

# Example Parameters (replace with real instance data when available!)
c = np.random.randint(1, 20, size=(NURSES, PATIENTS+2, PATIENTS+2))
T = np.random.randint(5, 30, size=(NURSES, PATIENTS+2, PATIENTS+2))
s = np.random.randint(10, 60, size=(PATIENTS+2,))
d = np.random.randint(10, 50, size=(NURSES,))
L = np.full(NURSES, 480)  # max 8 hours = 480 minutes
q = np.random.randint(1, 6, size=(NURSES, PATIENTS+1))
S = np.random.randint(1, 6, size=(NURSES, PATIENTS+1))

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
        for cut in cuts:
            groups.append(jobs[last:cut])
            last = cut
        groups.append(jobs[last:])
        solution = []
        for d, g in zip(delimiters, groups):
            solution.append(d)
            solution.extend(g)
        population.append(solution)
    return population

# =========================
# FUNCTION: Evaluate population (consistent with model)
# =========================
def evaluate_population(pop):
    values = []

    for sol in pop:
        F1, F2, F3 = 0, 0, 0

        nurse_indices = [i for i, val in enumerate(sol) if val < 0]
        nurse_indices.append(len(sol))

        for idx in range(len(nurse_indices)-1):
            n = -sol[nurse_indices[idx]] - 1
            jobs = sol[nurse_indices[idx]+1:nurse_indices[idx+1]]

            prev_node = 0
            time_worked = 0

            for j_idx, job in enumerate(jobs):
                travel_time = T[n][prev_node][job]

                if j_idx == 0:
                    start_time = travel_time
                else:
                    start_time = time_worked + travel_time

                end_time = start_time + s[job]

                # Objectives
                F1 += c[n][prev_node][job]            # travel cost
                F2 += travel_time + s[job]            # workload
                F3 += (q[n][job] - S[n][job])         # satisfaction

                time_worked = end_time
                prev_node = job

            if prev_node > 0:
                F1 += c[n][prev_node][PATIENTS+1]
                return_time = T[n][prev_node][PATIENTS+1]
                F2 += return_time
                time_worked += return_time

            if time_worked > L[n]:
                overtime = time_worked - L[n]
                F1 += d[n] * overtime

        values.append((F1, F2, -F3))  # negate F3 for minimization

    return values

# =========================
# Q-NDSA (Quick Non-Dominated Sorting Algorithm)
# =========================
def bi_objective_qndsa(indices, objectives):
    if not indices:
        return []
    front = []
    witness = float('inf')
    for i in indices:
        f2_value = objectives[i][1]
        if f2_value < witness:
            front.append(i)
            witness = f2_value
    return front

def is_non_dominated_tri_objective(candidate_idx, front_sorted, objectives):
    candidate_obj = objectives[candidate_idx]
    for front_idx in front_sorted:
        front_obj = objectives[front_idx]
        dominates = (front_obj[0] <= candidate_obj[0] and
                     front_obj[1] <= candidate_obj[1] and
                     front_obj[2] <= candidate_obj[2] and
                     (front_obj[0] < candidate_obj[0] or
                      front_obj[1] < candidate_obj[1] or
                      front_obj[2] < candidate_obj[2]))
        if dominates:
            return False
    return True

def rank_correction_tri_objective(front_E, remaining_population, objectives):
    if not front_E or not remaining_population:
        return front_E
    front_E_sorted = sorted(front_E, key=lambda i: objectives[i][2])
    extended_front = front_E_sorted.copy()
    remaining = [i for i in remaining_population if i not in front_E]
    for sol_idx in remaining:
        if is_non_dominated_tri_objective(sol_idx, front_E_sorted, objectives):
            insert_pos = 0
            while insert_pos < len(extended_front) and objectives[extended_front[insert_pos]][2] < objectives[sol_idx][2]:
                insert_pos += 1
            extended_front.insert(insert_pos, sol_idx)
    return extended_front

def Q_NDSA_tri_objective(population_objectives):
    if not population_objectives:
        return [], []
    N = len(population_objectives)
    ranks = [0] * N
    all_fronts = []
    remaining_indices = list(range(N))
    current_rank = 1
    while remaining_indices:
        remaining_indices.sort(key=lambda i: population_objectives[i][0])
        current_front = bi_objective_qndsa(remaining_indices, population_objectives)
        if len(population_objectives[0]) > 2:
            current_front = rank_correction_tri_objective(current_front, remaining_indices, population_objectives)
        for idx in current_front:
            ranks[idx] = current_rank
        all_fronts.append(current_front)
        remaining_indices = [i for i in remaining_indices if i not in current_front]
        current_rank += 1
    return ranks, all_fronts

def crowding_distance_3D(front_indices, objectives):
    distances = [0.0] * len(front_indices)
    if len(front_indices) <= 2:
        return [float('inf')] * len(front_indices)
    num_objectives = len(objectives[0])
    for obj_idx in range(num_objectives):
        sorted_indices = sorted(range(len(front_indices)),
                                key=lambda i: objectives[front_indices[i]][obj_idx])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        obj_values = [objectives[front_indices[i]][obj_idx] for i in sorted_indices]
        obj_range = obj_values[-1] - obj_values[0]
        if obj_range == 0:
            continue
        for i in range(1, len(sorted_indices)-1):
            distance_contrib = (obj_values[i+1] - obj_values[i-1]) / obj_range
            distances[sorted_indices[i]] += distance_contrib
    return distances

def apply_Q_NDSA_to_HHC(population, objectives_data):
    ranks, fronts = Q_NDSA_tri_objective(objectives_data)
    all_distances = [0.0] * len(population)
    for front in fronts:
        if len(front) > 0:
            distances = crowding_distance_3D(front, objectives_data)
            for i, sol_idx in enumerate(front):
                all_distances[sol_idx] = distances[i]
    return ranks, all_distances, fronts

# =========================
# MAIN
# =========================
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".csv"):
        inst_name = Path(file).stem
        print(f"\n=== Processing instance: {inst_name} ===")

        pop = generate_population(PATIENTS, NURSES, NUM_SOLUTIONS)
        values = evaluate_population(pop)
        ranks, distances, fronts = apply_Q_NDSA_to_HHC(pop, values)

        df_eval = pd.DataFrame(pop, columns=[f"pos{i+1}" for i in range(PATIENTS+NURSES)])
        df_eval["F1"] = [v[0] for v in values]
        df_eval["F2"] = [v[1] for v in values]
        df_eval["F3"] = [-v[2] for v in values]  # report positive satisfaction
        df_eval["Rank"] = ranks
        df_eval["CrowdingDistance"] = distances

        headers = ([f"pos{i+1}" for i in range(PATIENTS+NURSES)] +
                   ["F1", "F2", "F3", "Rank", "CrowdingDistance"])
        df_eval.to_csv(os.path.join(OUTPUT_FOLDER, f"population_QNDSA_{inst_name}.csv"),
                       index=False, sep=';', float_format="%.4f", header=headers)

        print(f"✅ Q-NDSA ranked population saved for {inst_name}")
        print(f"   - Number of fronts: {len(fronts)}")
        print(f"   - Best front size: {len(fronts[0]) if fronts else 0}")