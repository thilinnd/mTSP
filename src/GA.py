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



# --- NSGA II ---
# --- Đánh giá cá thể với 2 mục tiêu ---
def evaluate_individual(individual, dist_matrix, m):
    routes = tsp_split_dp(individual, m, dist_matrix)
    max_distance = max(calculate_route_distance(route, dist_matrix) for route in routes)
    total_distance = sum(calculate_route_distance(route, dist_matrix) for route in routes)
    return (max_distance, total_distance), routes

# --- Kiểm tra thống trị ---
def dominates(p, q):
    return all(x <= y for x, y in zip(p, q)) and any(x < y for x, y in zip(p, q))

# --- Phân loại theo không thống trị ---
def non_dominated_sort(pop):
    fronts = [[]]
    domination_count = {}
    dominated_set = {}

    for i, (fit_i, _) in enumerate(pop):
        dominated_set[i] = []
        domination_count[i] = 0
        for j, (fit_j, _) in enumerate(pop):
            if i == j:
                continue
            if dominates(fit_i, fit_j):
                dominated_set[i].append(j)
            elif dominates(fit_j, fit_i):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    rank = [0] * len(pop)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1], rank

# --- Tính khoảng cách chen chúc ---
def compute_crowding_distance(front, pop):
    distance = [0.0] * len(front)
    num_objectives = len(pop[0][0])

    for m in range(num_objectives):
        values = [(i, pop[i][0][m]) for i in front]
        values.sort(key=lambda x: x[1])
        min_val = values[0][1]
        max_val = values[-1][1]

        distance[front.index(values[0][0])] = float('inf')
        distance[front.index(values[-1][0])] = float('inf')

        if max_val == min_val:
            continue

        for k in range(1, len(values) - 1):
            prev = values[k - 1][1]
            next = values[k + 1][1]
            d = (next - prev) / (max_val - min_val)
            distance[front.index(values[k][0])] += d

    return distance

# --- NSGA-II chính ---
def solve_nsga2(dist_matrix, m, population_size=30, generations=100):
    n_cities = len(dist_matrix)
    population = [generate_random_individual(n_cities) for _ in range(population_size)]
    evaluated = [evaluate_individual(ind, dist_matrix, m) for ind in population]

    for gen in range(generations):
        offspring = []
        while len(offspring) < population_size:
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            child = local_search(child, dist_matrix, m)
            offspring.append(child)

        evaluated_offspring = [evaluate_individual(ind, dist_matrix, m) for ind in offspring]
        combined = evaluated + evaluated_offspring
        combined_individuals = population + offspring

        fronts, rank = non_dominated_sort(combined)
        new_population = []

        for front in fronts:
            if len(new_population) + len(front) > population_size:
                cd = compute_crowding_distance(front, combined)
                sorted_front = sorted(zip(front, cd), key=lambda x: -x[1])
                for idx, _ in sorted_front:
                    if len(new_population) < population_size:
                        new_population.append(combined[idx][1])
            else:
                for idx in front:
                    new_population.append(combined[idx][1])

        population = new_population
        evaluated = [evaluate_individual(ind, dist_matrix, m) for ind in population]

    # Trả về Pareto front cuối cùng
    final_front, _ = non_dominated_sort(evaluated)
    pareto_solutions = [evaluated[i] for i in final_front[0]]
    return pareto_solutions
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


