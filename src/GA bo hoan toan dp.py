def cached_route_distance(route_tuple):
    return sum(dist_matrix[route_tuple[i]][route_tuple[i + 1]] for i in range(len(route_tuple) - 1)) + dist_matrix[route_tuple[-1]][0]

# --- Tính khoảng cách một route ---
def calculate_route_distance(route, dist_matrix):
    return cached_route_distance(tuple(route))

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

    random.seed(42)  # Cố định seed để đảm bảo kết quả tái lập

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

    # Không còn dùng chia lại bằng DP → dùng chia greedy
    final_routes = tsp_split_dp(best_solution, m, dist_matrix)
    total_distance = sum(calculate_route_distance(route, dist_matrix) for route in final_routes)

    return total_distance, final_routes, best_fitness, fitness_per_generation
