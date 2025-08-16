import random
import numpy as np
import math
import time
from typing import List, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed

def cached_route_distance(route_tuple):
    return sum(dist_matrix[route_tuple[i]][route_tuple[i + 1]] for i in range(len(route_tuple) - 1)) + dist_matrix[route_tuple[-1]][0]

# --- Tính khoảng cách một route ---
def calculate_route_distance(route, dist_matrix):
    return cached_route_distance(tuple(route))

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
def solve(dist_matrix_input, m, population_size=100, generations=300):
    global dist_matrix
    dist_matrix = dist_matrix_input  

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

        for fitness, individual in evaluated:
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = individual

        fitness_per_generation.append(best_fitness)

        if gen % 20 == 0 or gen == generations - 1:
            print(f"[Gen {gen:3d}] Best fitness: {best_fitness:.2f}")

        # Chọn nửa tốt nhất
        evaluated.sort()
        parents = [ind for _, ind in evaluated[:population_size // 2]]

        # Tạo thế hệ mới
        new_population = []
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = local_search(child, dist_matrix, m)
            new_population.append(child)

        population = new_population

    # Sau khi tối ưu xong → chia lại bằng tsp_split_dp_exact để có kết quả chính xác
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

def solve_nsga2(matrix, m, pop_size=100, generations=300):
    num_cities = len(matrix)
    population = initialize_population(pop_size, num_cities)
    best_generation = 0
    fitness_per_generation = []
    best_fitness_overall = float('inf')

    
    no_improve_count = 0
    best_so_far = float('inf')
    gen_best_found = 0  # Thế hệ mà best_so_far được ghi nhận


    start_time = time.time()

    for gen in range(generations):
        fitnesses = [evaluate_fitness(ind, m, matrix) for ind in population]
        fitness_total = [
            sum(calculate_route_distance([0] + r + [0], matrix) for r in split_route(ind, m))
            for ind in population
        ]
        current_best = min(f[0] for f in fitnesses)
        fitness_per_generation.append(current_best)


                # Early stopping logic
        if current_best < best_so_far - 1e-3:
            best_fitness_overall = current_best
            best_generation = gen
            no_improve_count = 0
        else:
            no_improve_count += 1

        # if gen % 10 == 0:
        #     print(f"[m={m}] Thế hệ {gen}, fitness tốt nhất: {current_best:.2f}")

        if no_improve_count >= 100:
            print(f"[m={m}] Dừng sớm tại thế hệ {gen}")
            break

        new_population = []
        for _ in range(pop_size):
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # Lấy lời giải tốt nhất
    best_index = np.argmin([f[0] for f in fitnesses])
    best_fitness, best_balance, best_routes = fitnesses[best_index]
    end_time = time.time()
    
    return best_fitness, best_balance, best_routes, fitness_per_generation, end_time - start_time, best_generation


#--- GASA---
def run_gasa():
    NUM_CITIES = 20
    NUM_SALESMEN = 3
    POP_SIZE = 50
    NUM_GENERATIONS = 100
    MUTATION_RATE = 0.1
    DEPOT = 0
    TEMP_INIT = 100
    TEMP_FINAL = 1
    ALPHA = 0.95

    np.random.seed(42)
    coordinates = np.random.rand(NUM_CITIES + 1, 2) * 100
    dist_matrix = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)

    def decode(chrom):
        routes = [[] for _ in range(NUM_SALESMEN)]
        current = 0
        for city in chrom:
            routes[current].append(city)
            current = (current + 1) % NUM_SALESMEN
        return routes

    def total_distance(solution):
        total = 0
        for route in solution:
            if route:
                full = [DEPOT] + route + [DEPOT]
                for i in range(len(full) - 1):
                    total += dist_matrix[full[i], full[i+1]]
        return total

    def ox(p1, p2):
        start, end = sorted(random.sample(range(len(p1)), 2))
        child = [None] * len(p1)
        child[start:end] = p1[start:end]
        ptr = 0
        for city in p2:
            if city not in child:
                while child[ptr] is not None:
                    ptr += 1
                child[ptr] = city
        return child

    def mutate(chrom):
        if random.random() < MUTATION_RATE:
            i, j = random.sample(range(len(chrom)), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]
        return chrom

    def sa(routes):
        def rd(route):
            full = [DEPOT] + route + [DEPOT]
            return sum(dist_matrix[full[i], full[i+1]] for i in range(len(full)-1))

        result = []
        for route in routes:
            if len(route) <= 2:
                result.append(route)
                continue
            current = route.copy()
            cost = rd(current)
            T = TEMP_INIT
            while T > TEMP_FINAL:
                i, j = sorted(random.sample(range(len(current)), 2))
                neighbor = current.copy()
                neighbor[i:j] = reversed(neighbor[i:j])
                delta = rd(neighbor) - cost
                if delta < 0 or math.exp(-delta / T) > random.random():
                    current = neighbor
                    cost = rd(neighbor)
                T *= ALPHA
            result.append(current)
        return result

    cities = list(range(1, NUM_CITIES + 1))
    population = [random.sample(cities, len(cities)) for _ in range(POP_SIZE)]
    best_solution = decode(population[0])
    best_cost = total_distance(best_solution)

    for gen in range(NUM_GENERATIONS):
        scored = [(chrom, total_distance(decode(chrom))) for chrom in population]
        scored.sort(key=lambda x: x[1])
        new_population = [s[0] for s in scored[:10]]

        while len(new_population) < POP_SIZE:
            p1, p2 = random.choices([s[0] for s in scored[:25]], k=2)
            child = ox(p1, p2)
            child = mutate(child)
            decoded = decode(child)
            improved = sa(decoded)
            flat = [city for route in improved for city in route]
            new_population.append(flat)

        population = new_population
        current_best = decode(population[0])
        current_cost = total_distance(current_best)
        if current_cost < best_cost:
            best_solution = current_best
            best_cost = current_cost

    return best_solution, best_cost
