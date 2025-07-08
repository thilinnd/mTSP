import random
import numpy as np

# --- Tính khoảng cách tuyến ---
def calculate_route_distance(route, dist_matrix):
    return sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) + dist_matrix[route[-1]][0]

# --- Dynamic Programming chia tuyến ---
def tsp_split_dp(route, m, dist_matrix):
    n = len(route)
    dp = [[float('inf')] * (m+1) for _ in range(n+1)]
    path = [[-1] * (m+1) for _ in range(n+1)]
    dp[0][0] = 0

    for i in range(1, n+1):
        for k in range(1, m+1):
            for j in range(i):
                sub_route = [0] + route[j:i] + [0]
                cost = calculate_route_distance(sub_route, dist_matrix)
                if max(dp[j][k-1], cost) < dp[i][k]:
                    dp[i][k] = max(dp[j][k-1], cost)
                    path[i][k] = j

    routes = []
    i, k = n, m
    while k > 0:
        j = path[i][k]
        routes.append([0] + route[j:i] + [0])
        i, k = j, k-1

    return routes[::-1]

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

    for _ in range(10):
        i, j = random.sample(range(len(route)), 2)
        new = best[:]
        new[i], new[j] = new[j], new[i]
        new_routes = tsp_split_dp(new, m, dist_matrix)
        new_fitness = fitness_func(new_routes, dist_matrix)
        if new_fitness < best_fitness:
            best = new
            best_fitness = new_fitness

    return best

# --- Hàm chính giải bài toán m-TSP ---
def solve(dist_matrix, m, population_size=30, generations=300):
    n_cities = len(dist_matrix)
    population = [generate_random_individual(n_cities) for _ in range(population_size)]

    best_solution = None
    best_fitness = float('inf')
    fitness_per_generation = []

    for gen in range(generations):
        new_population = []
        evaluated = []

        for individual in population:
            routes = tsp_split_dp(individual, m, dist_matrix)
            fitness = fitness_func(routes, dist_matrix)
            evaluated.append((fitness, individual))

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = routes

        fitness_per_generation.append(best_fitness)

        # Chọn nửa tốt nhất làm bố mẹ
        evaluated.sort()
        parents = [ind for _, ind in evaluated[:population_size // 2]]

        # Tạo thế hệ mới
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = local_search(child, dist_matrix, m)
            new_population.append(child)

        population = new_population

    total_distance = sum(calculate_route_distance(route, dist_matrix) for route in best_solution)

    return total_distance, best_solution, best_fitness, fitness_per_generation
