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

# --- Hàm chính giải bài toán m-TSP ---
def solve(dist_matrix, m, population_size=100, generations=300):
    n_cities = len(dist_matrix)
    population = [generate_random_individual(n_cities) for _ in range(population_size)]

    best_solution = None
    best_fitness = float('inf')
    fitness_per_generation = []

    for gen in tqdm(range(generations), desc=f"Chạy GA (m = {m})", ncols=80):
        # Tính fitness song song
        evaluated = Parallel(n_jobs=-1)(
            delayed(lambda ind: (
                fitness_func(tsp_split_dp(ind, m, dist_matrix), dist_matrix),
                ind
            ))(individual)
            for individual in population
        )

        # Cập nhật lời giải tốt nhất
        for fitness, individual in evaluated:
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = individual

        fitness_per_generation.append(best_fitness)

        if gen % 20 == 0 or gen == generations - 1:
            print(f"[Gen {gen:3d}] Best fitness: {best_fitness:.2f}")

        # Chọn nửa tốt nhất
        evaluated.sort(key=lambda x: x[0])  # sắp xếp theo fitness
        parents = [ind for _, ind in evaluated[:population_size // 2]]

        # Tạo thế hệ mới
        new_population = []
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = local_search(child, dist_matrix, m)
            new_population.append(child)

        population = new_population

    # Sau khi tối ưu xong → chia lại bằng DP chính xác
    final_routes = tsp_split_dp_exact(best_solution, m, dist_matrix)
    total_distance = sum(calculate_route_distance(route, dist_matrix) for route in final_routes)

    return total_distance, final_routes, best_fitness, fitness_per_generation

def split_route(route, m):
    chunk_size = len(route) // m
    return [route[i*chunk_size:(i+1)*chunk_size] for i in range(m-1)] + [route[(m-1)*chunk_size:]]

def evaluate_fitness(individual, m, matrix):
    routes = split_route(individual, m)
    route_lengths = [calculate_route_distance([0] + r + [0], matrix) for r in routes]
    longest = max(route_lengths)
    balance = np.std(route_lengths)
    return longest, balance, routes

def initialize_population(pop_size, num_cities):
    return [random.sample(range(1, num_cities), num_cities - 1) for _ in range(pop_size)]

def tournament_selection(population, fitnesses):
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitnesses[i][0] < fitnesses[j][0] else population[j]

def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child_p1 = parent1[a:b]
    child = [gene for gene in parent2 if gene not in child_p1]
    return child[:a] + child_p1 + child[a:]

def mutate(individual, mutation_rate=0.1):
    for _ in range(int(len(individual) * mutation_rate)):
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
    return individual
