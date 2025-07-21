import random
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
import time

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
# --- Đánh giá đa mục tiêu ---
# def evaluate_objectives(individual, dist_matrix, m):
#     routes = tsp_split_dp(individual, m, dist_matrix)
#     total_distance = sum(calculate_route_distance(route, dist_matrix) for route in routes)
#     max_route_distance = max(calculate_route_distance(route, dist_matrix) for route in routes)
#     return total_distance, max_route_distance, routes

# # --- Pareto Dominance ---
# def dominates(ind1, ind2):
#     return all(x <= y for x, y in zip(ind1[:2], ind2[:2])) and any(x < y for x, y in zip(ind1[:2], ind2[:2]))

# # --- Non-dominated Sorting ---
# def non_dominated_sort(population):
#     fronts = [[]]
#     domination_count = {}
#     dominated_solutions = {}

#     for i, p in enumerate(population):
#         dominated_solutions[i] = []
#         domination_count[i] = 0
#         for j, q in enumerate(population):
#             if dominates(p, q):
#                 dominated_solutions[i].append(j)
#             elif dominates(q, p):
#                 domination_count[i] += 1
#         if domination_count[i] == 0:
#             fronts[0].append(i)

#     rank = [0] * len(population)
#     i = 0
#     while fronts[i]:
#         next_front = []
#         for p in fronts[i]:
#             for q in dominated_solutions[p]:
#                 domination_count[q] -= 1
#                 if domination_count[q] == 0:
#                     rank[q] = i + 1
#                     next_front.append(q)
#         i += 1
#         fronts.append(next_front)

#     return fronts[:-1], rank

# # --- Crowding Distance ---
# def compute_crowding_distance(front, population):
#     distance = [0.0] * len(front)
#     for m in range(2):  # 0: total_distance, 1: max_route
#         values = [(population[i][m], i) for i in front]
#         values.sort()
#         min_value = values[0][0]
#         max_value = values[-1][0]
#         distance[front.index(values[0][1])] = float('inf')
#         distance[front.index(values[-1][1])] = float('inf')
#         for k in range(1, len(front) - 1):
#             prev = values[k - 1][0]
#             next = values[k + 1][0]
#             if max_value - min_value == 0:
#                 dist = 0
#             else:
#                 dist = (next - prev) / (max_value - min_value)
#             distance[front.index(values[k][1])] += dist
#     return distance

# # --- NSGA-II chính ---
# def solve_nsga2(dist_matrix, m, population_size=30, generations=100):
#     n_cities = len(dist_matrix)
#     population = [generate_random_individual(n_cities) for _ in range(population_size)]

#     evaluated = []
#     for ind in population:
#         obj = evaluate_objectives(ind, dist_matrix, m)
#         evaluated.append(obj)

#     best_fitness_per_gen = []  # Ghi lại max-route tốt nhất mỗi thế hệ

#     for gen in range(generations):
#         # --- Tạo con ---
#         offspring = []
#         while len(offspring) < population_size:
#             p1, p2 = random.sample(population, 2)
#             child = crossover(p1, p2)
#             child = local_search(child, dist_matrix, m)
#             offspring.append(child)

#         offspring_eval = [evaluate_objectives(ind, dist_matrix, m) for ind in offspring]

#         # --- Kết hợp cha + con ---
#         combined = evaluated + offspring_eval
#         combined_population = population + offspring

#         # --- Pareto sort ---
#         fronts, rank = non_dominated_sort(combined)

#         new_population = []
#         new_evaluated = []

#         for front in fronts:
#             if len(new_population) + len(front) > population_size:
#                 crowding_dist = compute_crowding_distance(front, combined)
#                 sorted_front = [x for _, x in sorted(zip(crowding_dist, front), reverse=True)]
#                 for idx in sorted_front:
#                     if len(new_population) < population_size:
#                         new_population.append(combined_population[idx])
#                         new_evaluated.append(combined[idx])
#             else:
#                 for idx in front:
#                     new_population.append(combined_population[idx])
#                     new_evaluated.append(combined[idx])

#         population = new_population
#         evaluated = new_evaluated

#         # --- Ghi lại fitness tốt nhất mỗi thế hệ ---
#         best_fitness = min(ind[1] for ind in evaluated)
#         best_fitness_per_gen.append(best_fitness)

#     # --- Trả về các nghiệm không trội (Pareto Front 1) ---
#     pareto_front = [ind for ind, val in zip(population, evaluated) if evaluated.count(val) == 1]
#     final_routes = [evaluate_objectives(ind, dist_matrix, m)[2] for ind in pareto_front]

#     return final_routes, best_fitness_per_gen

def calculate_route_distance(route, matrix):
    return sum(matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + matrix[route[-1]][route[0]]

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
    fitness_per_generation = []
    
    no_improve_count = 0
    best_so_far = float('inf')
    gen_best_found = 0  # Thế hệ mà best_so_far được ghi nhận


    start_time = time.time()

    for gen in range(generations):
        fitnesses = [evaluate_fitness(ind, m, matrix) for ind in population]
        current_best = min(f[0] for f in fitnesses)
        fitness_per_generation.append(current_best)

                # Early stopping logic
        if current_best < best_so_far - 1e-3:
            best_so_far = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1

        # if gen % 10 == 0:
        #     print(f"[m={m}] Thế hệ {gen}, fitness tốt nhất: {current_best:.2f}")

        if no_improve_count >= 50:
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
    
    return best_fitness, best_balance, best_routes, fitness_per_generation, end_time - start_time, gen + 1 if no_improve_count < 30 else gen

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

# --- RLGA (Reinforcement Learning Genetic Algorithm) ---
class RLGA_mTSP:
    def __init__(self, distance_matrix: np.ndarray, m: int = 3,
                 pop_size: int = 50, generations: int = 100,
                 epsilon: float = 0.1, epsilon_decay: float = 0.99):
        
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.m = m
        self.pop_size = pop_size
        self.generations = generations
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.depot = 0
        
        # RL parameters - simplified
        self.crossover_rates = [0.7, 0.8, 0.9]
        self.mutation_rates = [0.05, 0.1, 0.15]
        
        # Q-table for RL - simplified state representation
        self.Q = {}
        self.current_pc = 0.8
        self.current_pm = 0.1
        
        # Performance tracking
        self.stagnation_count = 0
        self.last_best_fitness = float('inf')

    def initialize_population(self) -> List[List[int]]:
        """Initialize population with random permutations"""
        cities = list(range(1, self.n_cities))  # Exclude depot
        population = []
        for _ in range(self.pop_size):
            individual = cities.copy()
            random.shuffle(individual)
            population.append(individual)
        return population

    def decode_solution(self, chromosome: List[int]) -> List[List[int]]:
        """Decode chromosome into routes for m salesmen"""
        routes = [[] for _ in range(self.m)]
        for i, city in enumerate(chromosome):
            routes[i % self.m].append(city)
        return routes

    def calculate_fitness(self, chromosome: List[int]) -> Tuple[float, float, float]:
        """Calculate fitness metrics: (total_distance, max_route_length, balance_metric)"""
        routes = self.decode_solution(chromosome)
        route_distances = []
        total_distance = 0
        
        for route in routes:
            if len(route) > 0:
                full_route = [self.depot] + route + [self.depot]
                distance = sum(self.distance_matrix[full_route[i]][full_route[i + 1]] 
                             for i in range(len(full_route) - 1))
                route_distances.append(distance)
                total_distance += distance
            else:
                route_distances.append(0)
        
        max_route_length = max(route_distances) if route_distances else 0
        min_route_length = min(route_distances) if route_distances else 0
        balance_metric = max_route_length - min_route_length
        
        return total_distance, max_route_length, balance_metric

    def get_state(self, population_metrics: List[Tuple[float, float, float]]) -> str:
        """Simplified state representation"""
        total_distances = [metrics[0] for metrics in population_metrics]
        diversity = np.std(total_distances)
        improvement = self.last_best_fitness - min(total_distances)
        
        # Discretize into simple states
        diversity_level = "high" if diversity > np.mean(total_distances) * 0.1 else "low"
        improvement_level = "good" if improvement > 0 else "poor"
        
        return f"{diversity_level}_{improvement_level}"

    def select_action(self, state: str) -> Tuple[float, float]:
        """Epsilon-greedy action selection"""
        if state not in self.Q:
            self.Q[state] = {}
            for pc in self.crossover_rates:
                for pm in self.mutation_rates:
                    self.Q[state][(pc, pm)] = 0.0
        
        if random.random() < self.epsilon:
            # Exploration
            return random.choice(self.crossover_rates), random.choice(self.mutation_rates)
        else:
            # Exploitation
            best_action = max(self.Q[state].items(), key=lambda x: x[1])[0]
            return best_action

    def update_q_value(self, state: str, action: Tuple[float, float], reward: float):
        """Update Q-value with simplified Q-learning"""
        alpha = 0.1
        if state not in self.Q:
            self.Q[state] = {}
            for pc in self.crossover_rates:
                for pm in self.mutation_rates:
                    self.Q[state][(pc, pm)] = 0.0
        
        current_q = self.Q[state][action]
        self.Q[state][action] = current_q + alpha * (reward - current_q)

    def tournament_selection(self, population: List[List[int]], 
                           fitness_metrics: List[Tuple[float, float, float]], k: int = 3) -> List[int]:
        """Tournament selection based on total distance"""
        indices = random.sample(range(len(population)), k)
        winner_idx = min(indices, key=lambda i: fitness_metrics[i][0])  # Use total distance
        return population[winner_idx].copy()

    def crossover_ox(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX)"""
        if random.random() > self.current_pc:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        def create_child(p1, p2):
            child = [None] * size
            child[start:end] = p1[start:end]
            
            pointer = 0
            for city in p2:
                if city not in child:
                    while pointer < size and child[pointer] is not None:
                        pointer += 1
                    if pointer < size:
                        child[pointer] = city
            return child
        
        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
        
        return child1, child2

    def mutate_swap(self, chromosome: List[int]) -> List[int]:
        """Swap mutation"""
        child = chromosome.copy()
        if random.random() < self.current_pm:
            i, j = random.sample(range(len(child)), 2)
            child[i], child[j] = child[j], child[i]
        return child

    def local_search_2opt(self, chromosome: List[int]) -> List[int]:
        """Simple 2-opt local search for improvement"""
        best = chromosome.copy()
        best_metrics = self.calculate_fitness(best)
        best_fitness = best_metrics[0]  # Total distance
        
        # Try a few 2-opt moves
        for _ in range(5):
            i, j = sorted(random.sample(range(len(chromosome)), 2))
            if j - i < 2:
                continue
                
            new_chromosome = best.copy()
            new_chromosome[i:j] = reversed(new_chromosome[i:j])
            
            new_metrics = self.calculate_fitness(new_chromosome)
            new_fitness = new_metrics[0]  # Total distance
            if new_fitness < best_fitness:
                best = new_chromosome
                best_fitness = new_fitness
        
        return best

    def run(self) -> Tuple[List[List[int]], float, float, float, List[float]]:
        """Main RLGA algorithm - returns (routes, total_distance, max_route_length, balance_metric, fitness_history)"""
        # Initialize
        population = self.initialize_population()
        best_solution = None
        best_total_distance = float('inf')
        best_max_route_length = float('inf')
        best_balance_metric = float('inf')
        fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate population
            fitness_metrics = [self.calculate_fitness(ind) for ind in population]
            total_distances = [metrics[0] for metrics in fitness_metrics]
            current_best_total = min(total_distances)
            
            # Track best solution
            if current_best_total < best_total_distance:
                best_total_distance = current_best_total
                best_idx = total_distances.index(current_best_total)
                best_solution = self.decode_solution(population[best_idx])
                best_max_route_length = fitness_metrics[best_idx][1]
                best_balance_metric = fitness_metrics[best_idx][2]
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            fitness_history.append(best_total_distance)
            
            # RL state and action selection
            state = self.get_state(fitness_metrics)
            self.current_pc, self.current_pm = self.select_action(state)
            
            # Create new population
            new_population = []
            
            # Keep best individuals (elitism)
            sorted_indices = sorted(range(len(population)), key=lambda i: total_distances[i])
            elite_count = max(1, self.pop_size // 10)
            for i in range(elite_count):
                new_population.append(population[sorted_indices[i]].copy())
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitness_metrics)
                parent2 = self.tournament_selection(population, fitness_metrics)
                
                child1, child2 = self.crossover_ox(parent1, parent2)
                child1 = self.mutate_swap(child1)
                child2 = self.mutate_swap(child2)
                
                # Apply local search occasionally
                if random.random() < 0.1:
                    child1 = self.local_search_2opt(child1)
                if random.random() < 0.1:
                    child2 = self.local_search_2opt(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.pop_size]
            
            # Calculate reward and update Q-value
            improvement = self.last_best_fitness - current_best_total
            reward = improvement / (abs(self.last_best_fitness) + 1e-6)
            self.update_q_value(state, (self.current_pc, self.current_pm), reward)
            self.last_best_fitness = current_best_total
            
            # Decay epsilon
            self.epsilon *= self.epsilon_decay
        
        return best_solution, best_total_distance, best_max_route_length, best_balance_metric, fitness_history

# --- Hàm chạy RLGA ---
def solve_rlga(distance_matrix, m, pop_size=50, generations=100):
    """Wrapper function for RLGA to match the interface of other algorithms"""
    rlga = RLGA_mTSP(distance_matrix, m=m, pop_size=pop_size, generations=generations)
    routes, total_distance, max_route_length, balance_metric, fitness_history = rlga.run()
    return routes, total_distance, max_route_length, balance_metric, fitness_history

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


