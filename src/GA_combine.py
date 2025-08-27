import random
import numpy as np
import math
import time
from typing import List, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
SEED = 42  
random.seed(SEED)
np.random.seed(SEED)

# --- Tính khoảng cách một route ---
def calculate_route_distance(route, dist_matrix):
    return sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + dist_matrix[route[-1]][0]

# --- Hàm gốc tsp_split_dp dùng DP (chính xác nhưng chậm) ---
def tsp_split_dp_exact(route, m, dist_matrix):
    n = len(route)
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    path = [[-1] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for k in range(1, m + 1):
            for j in range(i):
                sub_route = [0] + route[j:i] + [0]
                cost = calculate_route_distance(sub_route, dist_matrix)
                if max(dp[j][k - 1], cost) < dp[i][k]:
                    dp[i][k] = max(dp[j][k - 1], cost)
                    path[i][k] = j

    routes = []
    i, k = n, m
    while k > 0:
        j = path[i][k]
        routes.append([0] + route[j:i] + [0])
        i, k = j, k - 1

    return routes[::-1]

# --- Greedy split nhanh để thay thế trong quá trình tiến hoá ---
def tsp_split_dp(route, m, dist_matrix):
    avg_len = len(route) // m
    routes = []
    for i in range(m):
        start = i * avg_len
        end = (i + 1) * avg_len if i < m - 1 else len(route)
        routes.append([0] + route[start:end] + [0])
    return routes

# --- Tính fitness (min-max) ---
def fitness_func(routes, dist_matrix):
    return max(calculate_route_distance(route, dist_matrix) for route in routes)

# --- Tạo cá thể TSP ngẫu nhiên ---
def generate_random_individual(n_cities):
    cities = list(range(1, n_cities))
    random.shuffle(cities)
    return cities

# --- Lai ghép ---
def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    segment = parent1[a:b]
    child = [gene for gene in parent2 if gene not in segment]
    return child[:a] + segment + child[a:]

# --- Tối ưu cục bộ ---
def local_search(route, dist_matrix, m):
    best = route[:]
    best_routes = tsp_split_dp(best, m, dist_matrix)
    best_fitness = fitness_func(best_routes, dist_matrix)

    for _ in range(5): 
        i, j = random.sample(range(len(route)), 2)
        if i == j:
            continue
        new = best[:]
        new[i], new[j] = new[j], new[i]
        new_routes = tsp_split_dp(new, m, dist_matrix)
        new_fitness = fitness_func(new_routes, dist_matrix)
        if new_fitness < best_fitness:
            best = new
            best_fitness = new_fitness

    return best

# --- Hàm chính giải bài toán m-TSP với Adaptive Mutation Rate ---
def solve(dist_matrix_input, m, population_size=100, generations=300,
          initial_mutation_rate=0.2, min_mutation=0.05, max_mutation=0.5,
          elite_size=2):
    global dist_matrix
    dist_matrix = dist_matrix_input  

    n_cities = len(dist_matrix)
    population = [generate_random_individual(n_cities) for _ in range(population_size)]

    best_solution = None
    best_fitness = float('inf')
    fitness_per_generation = []

    mutation_rate = initial_mutation_rate
    previous_best_fitness = float('inf')

    for gen in tqdm(range(generations), desc=f"Chạy GA (m = {m})", ncols=80):
        # Tính fitness song song
        evaluated = Parallel(n_jobs=-1)(
            delayed(lambda ind: (
                fitness_func(tsp_split_dp(ind, m, dist_matrix), dist_matrix),
                ind
            ))(individual)
            for individual in population
        )

        # Sắp xếp theo fitness tăng dần
        evaluated.sort()
        elites = [ind for _, ind in evaluated[:elite_size]]

        # Cập nhật best
        if evaluated[0][0] < best_fitness:
            best_fitness = evaluated[0][0]
            best_solution = evaluated[0][1]

        fitness_per_generation.append(best_fitness)

        # --- Adaptive Mutation Rate ---
        improvement = previous_best_fitness - best_fitness
        if improvement < 1e-2:
            mutation_rate = min(max_mutation, mutation_rate * 1.1)
        else:
            mutation_rate = max(min_mutation, mutation_rate * 0.9)
        previous_best_fitness = best_fitness

        if gen % 20 == 0 or gen == generations - 1:
            print(f"[Gen {gen:3d}] Best fitness: {best_fitness:.2f} | Mutation rate: {mutation_rate:.2f}")

        # Chọn nửa tốt nhất để sinh sản
        parents = [ind for _, ind in evaluated[:population_size // 2]]

        # Sinh cá thể mới (trừ elite)
        new_population = elites[:]
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)

            # Xác suất đột biến theo adaptive mutation rate
            if random.random() < mutation_rate:
                child = local_search(child, dist_matrix, m)

            new_population.append(child)

        population = new_population

    # Sau khi tối ưu xong → chia lại bằng tsp_split_dp_exact để có kết quả chính xác
    final_routes = tsp_split_dp_exact(best_solution, m, dist_matrix)
    total_distance = sum(calculate_route_distance(route, dist_matrix) for route in final_routes)

    return total_distance, final_routes, best_fitness, fitness_per_generation

# --- Hàm phát hiện hội tụ ---
def detect_convergence(generation_fitness, tolerance=1e-3, window=5):
    """Detect convergence in fitness evolution"""
    if len(generation_fitness) < window:
        return len(generation_fitness)
    for i in range(len(generation_fitness) - window):
        window_values = generation_fitness[i:i+window]
        if max(window_values) - min(window_values) < tolerance:
            return i + window
    return len(generation_fitness)


