import numpy as np
import random
import math

# ==== THÔNG SỐ ====
NUM_CITIES = 20      # Số điểm cần đến (không tính depot)
NUM_SALESMEN = 3     # Số người giao hàng
POP_SIZE = 50        # Kích thước quần thể
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
DEPOT = 0            # Điểm bắt đầu/kết thúc
TEMP_INIT = 100
TEMP_FINAL = 1
ALPHA = 0.95

# ==== TẠO DỮ LIỆU ====
np.random.seed(42)
coordinates = np.random.rand(NUM_CITIES + 1, 2) * 100  # +1 vì có depot
distance_matrix = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)

# ==== TÍNH TỔNG KHOẢNG CÁCH CHO MỘT GIẢI PHÁP ====
def calculate_total_distance(solution):
    total_distance = 0
    for route in solution:
        if len(route) == 0:
            continue
        full_route = [DEPOT] + route + [DEPOT]
        for i in range(len(full_route) - 1):
            total_distance += distance_matrix[full_route[i], full_route[i+1]]
    return total_distance

# ==== MÃ HÓA GIẢI PHÁP ====
# Dạng: List gồm m tuyến (m salesman), mỗi tuyến là list các city (không chứa depot)
def decode_chromosome(chrom):
    routes = [[] for _ in range(NUM_SALESMEN)]
    current = 0
    for city in chrom:
        routes[current].append(city)
        current = (current + 1) % NUM_SALESMEN
    return routes

# ==== KHỞI TẠO QUẦN THỂ ====
def init_population():
    cities = list(range(1, NUM_CITIES + 1))  # Bỏ depot (0)
    population = []
    for _ in range(POP_SIZE):
        chrom = cities.copy()
        random.shuffle(chrom)
        population.append(chrom)
    return population

# ==== LAI GHÉP OX ====
def crossover(p1, p2):
    start, end = sorted(random.sample(range(len(p1)), 2))
    child = [None] * len(p1)
    child[start:end] = p1[start:end]
    pointer = 0
    for city in p2:
        if city not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = city
    return child

# ==== ĐỘT BIẾN: hoán đổi 2 thành phố ====
def mutate(chrom):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom

# ==== SIMULATED ANNEALING TRÊN TỪNG SALES ROUTE ====
def simulated_annealing(routes, temp_init, temp_final, alpha):
    new_routes = []
    for route in routes:
        if len(route) <= 2:
            new_routes.append(route)
            continue

        current = route.copy()
        current_cost = route_distance(current)
        T = temp_init
        while T > temp_final:
            i, j = sorted(random.sample(range(len(current)), 2))
            neighbor = current.copy()
            neighbor[i:j] = reversed(neighbor[i:j])
            neighbor_cost = route_distance(neighbor)
            delta = neighbor_cost - current_cost
            if delta < 0 or math.exp(-delta / T) > random.random():
                current = neighbor
                current_cost = neighbor_cost
            T *= alpha
        new_routes.append(current)
    return new_routes

# ==== TÍNH KHOẢNG CÁCH MỘT ROUTE ====
def route_distance(route):
    if not route:
        return 0
    full_route = [DEPOT] + route + [DEPOT]
    return sum(distance_matrix[full_route[i], full_route[i+1]] for i in range(len(full_route) - 1))

# ==== MAIN GASA LOOP ====
population = init_population()
best_solution = decode_chromosome(population[0])
best_cost = calculate_total_distance(best_solution)

for gen in range(NUM_GENERATIONS):
    scored = [(chrom, calculate_total_distance(decode_chromosome(chrom))) for chrom in population]
    scored.sort(key=lambda x: x[1])
    new_population = [s[0] for s in scored[:10]]  # elitism

    while len(new_population) < POP_SIZE:
        p1, p2 = random.choices([s[0] for s in scored[:25]], k=2)
        child = crossover(p1, p2)
        child = mutate(child)
        decoded = decode_chromosome(child)
        improved = simulated_annealing(decoded, TEMP_INIT, TEMP_FINAL, ALPHA)
        flat = [city for route in improved for city in route]
        new_population.append(flat)

    population = new_population
    current_best = decode_chromosome(population[0])
    current_cost = calculate_total_distance(current_best)
    if current_cost < best_cost:
        best_solution = current_best
        best_cost = current_cost

    print(f"Generation {gen+1}: Best Cost = {best_cost:.2f}")

# ==== KẾT QUẢ ====
print("\nBest solution (routes):")
for i, route in enumerate(best_solution):
    print(f"Salesman {i+1}: { [DEPOT] + route + [DEPOT] }")
print("Total cost:", round(best_cost, 2))
