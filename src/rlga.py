import random
import math
import copy

class RLGA:
    def __init__(self, tsp_data, salesman_count, pop_size=50, generations=500):
        self.tsp_data = tsp_data
        self.salesman_count = salesman_count
        self.pop_size = pop_size
        self.generations = generations
        self.num_cities = len(tsp_data)

        self.q_table = {}  # Q-table cho reinforcement learning
        self.actions = [(0.6, 0.01), (0.7, 0.02), (0.8, 0.05), (0.9, 0.1)]  # (Pc, Pm)
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.9

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            perm = list(range(self.num_cities))
            random.shuffle(perm)
            population.append(perm)
        return population

    def evaluate_fitness(self, individual):
        routes = self.tsp_split_dp(individual)
        distances = [self.route_distance(route) for route in routes]
        return max(distances), sum(distances), routes

    def tsp_split_dp(self, chromosome):
        n = len(chromosome)
        dp = [[math.inf for _ in range(self.salesman_count + 1)] for _ in range(n + 1)]
        path = [[-1 for _ in range(self.salesman_count + 1)] for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(n + 1):
            for k in range(self.salesman_count):
                if dp[i][k] < math.inf:
                    for j in range(i + 1, n + 1):
                        route = chromosome[i:j]
                        dist = self.route_distance(route)
                        if dp[j][k + 1] > dp[i][k] + dist:
                            dp[j][k + 1] = dp[i][k] + dist
                            path[j][k + 1] = i
        routes = []
        idx = n
        for k in range(self.salesman_count, 0, -1):
            i = path[idx][k]
            routes.append(chromosome[i:idx])
            idx = i
        return routes[::-1]

    def route_distance(self, route):
        dist = 0
        for i in range(len(route)):
            dist += self.tsp_data[route[i - 1]][route[i]]
        return dist

    def crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = p1[a:b]
        p2_filtered = [x for x in p2 if x not in child[a:b]]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_filtered[j]
                j += 1
        return child

    def mutate(self, individual):
        a, b = sorted(random.sample(range(len(individual)), 2))
        individual[a], individual[b] = individual[b], individual[a]

    def local_search(self, routes):
        best_routes = copy.deepcopy(routes)
        best_dist = sum(self.route_distance(r) for r in best_routes)
        for _ in range(10):
            temp_routes = copy.deepcopy(best_routes)
            r_idx = random.randint(0, len(temp_routes) - 1)
            if len(temp_routes[r_idx]) >= 2:
                a, b = sorted(random.sample(range(len(temp_routes[r_idx])), 2))
                temp_routes[r_idx][a], temp_routes[r_idx][b] = temp_routes[r_idx][b], temp_routes[r_idx][a]
                temp_dist = sum(self.route_distance(r) for r in temp_routes)
                if temp_dist < best_dist:
                    best_dist = temp_dist
                    best_routes = temp_routes
        return best_routes

    def get_state(self, population):
        distances = [self.evaluate_fitness(ind)[1] for ind in population]
        diversity = len(set(tuple(ind) for ind in population))
        improvement = max(distances) - min(distances)
        convergence = sum(distances) / len(distances)
        return (diversity, improvement, convergence)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table.get(state, {})
        if not q_values:
            return random.choice(self.actions)
        return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        next_q = max(self.q_table.get(next_state, {}).values(), default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * next_q - self.q_table[state][action])

    def run(self):
        population = self.initialize_population()
        best_total_distance = float('inf')
        best_solution = None

        for gen in range(self.generations):
            state = self.get_state(population)
            action = self.select_action(state)
            Pc, Pm = action

            new_population = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(population, 2)
                if random.random() < Pc:
                    child = self.crossover(p1, p2)
                else:
                    child = p1[:]
                if random.random() < Pm:
                    self.mutate(child)
                _, _, routes = self.evaluate_fitness(child)
                improved_routes = self.local_search(routes)
                flattened = [city for r in improved_routes for city in r]
                new_population.append(flattened)

            population = new_population
            next_state = self.get_state(population)
            _, total_distances, _ = zip(*(self.evaluate_fitness(ind) for ind in population))
            avg_distance = sum(total_distances) / len(total_distances)
            reward = -avg_distance
            self.update_q_table(state, action, reward, next_state)
            self.epsilon *= self.epsilon_decay

            gen_best_idx = total_distances.index(min(total_distances))
            if total_distances[gen_best_idx] < best_total_distance:
                best_total_distance = total_distances[gen_best_idx]
                _, _, best_solution = self.evaluate_fitness(population[gen_best_idx])

            print(f"Generation {gen}: Best Total Distance = {best_total_distance:.2f}")

        print("\nBest routes per salesman:")
        for idx, route in enumerate(best_solution):
            print(f"Salesman {idx+1}: {route} (distance: {self.route_distance(route):.2f})")
        print(f"Total Distance: {sum(self.route_distance(r) for r in best_solution):.2f}")
