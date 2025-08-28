# A Comparative Study of Genetic Algorithm Variants in Min–Max Optimization: Evidence from Vietnamese Business

## I. Introduction
Fresh dairy products such as Vinamilk yogurt have a short shelf life - approximately 45 days, making it essential to optimize delivery time and logistics efficiency. Our project addresses the Multiple Traveling Salesmen Problem (m-TSP) in the context of cold chain logistics for milk distribution in Vietnam.  

The solution leverages a Genetic Algorithm (GA) with customizable parameters to minimize total travel distance and delivery cost, ensuring products are delivered faster, fresher, and more reliably. 

======
## II. Background and Motivation
Poor route planning in cold supply chains can result in:
* Overstocking or expired products
* Financial losses for distributors
* Damage to brand reputation

Our algorithm helps distributors like Vinamilk enhance decision-making by identifying the most efficient routes for multiple delivery vehicles departing from a shared warehouse.

======
## III. Objectives
This project aims to:
1. Develop a GA-based solution to solve the m-TSP tailored to cold chain logistics in Vietnam's dairy industry.
2. Optimize delivery routes for multiple vehicles starting from a common warehouse, minimizing total travel distance and transportation cost.
3. Improve logistics efficiency for short shelf-life products like yogurt by ensuring faster and more effective delivery schedules.
4. Support data-driven decision making by providing a customizable tool for route planning, allowing users to input the number of vehicles, depot location, and store list.
5. Validate the algorithm on real-world data, specifically within the context of a leading milk production and distribution company in Vietnam, to demonstrate practical applicability.
6. Contribute to the research on hybrid optimization techniques, showcasing the effectiveness of GAs in solving complex, real-world m-TSP instances.

======
## IV. Methodlogy
**1. Data Preprocessing**

The study uses data from 126 Vinamilk stores in Hanoi, with geographic coordinates collected manually. A distance matrix is built using the Open Source Routing Machine (OSRM), deployed locally to calculate actual driving distances based on Vietnam's road network. To bypass OSRM's query limits, data is processed in batches and aggregated into a complete matrix.

**2. Hybrid GA Optimization**

We solve the m-TSP using a GA enhanced with:
* Greedy route splitting for quick fitness evaluation
* Dynamic Programming (DP) for optimal final route segmentation
* Local search for refining solutions

The objective is to minimize the longest route among all delivery agents, promoting balanced and efficient cold-chain delivery.

**3. Benchmarking**

The proposed method is compared with NSGA-II, RLGA, and GASA using six metrics: total distance, max route length, balance, execution time, convergence iterations, and speed. Our hybrid GA approach shows strong performance and practical applicability.

======
## V. Results
1. RLGA outperformed all other algorithms in terms of: 
* Shortest total travel distance
* Lowest maximum route length
* Fastest execution time (up to 10x faster)

However, RLGA exhibited high imbalance, especially for  ```m = 2```, making it less suitable where equal workload distribution is critical.

2. GA with Local Search offered the most balanced performance overall:
* Achieved good route balance, especially at  ```m = 2``` and  ```m = 3```
* Had high convergence speed and low iteration counts
* Demonstrated strong performance across all metrics, making it the most practical and consistent approach

3. NSGA-II performed well on route balancing, but failed to converge within 300-500 generations, limiting its effectiveness in real-world scenarios.
4. GASA balanced speed and solution quality:
* Fast convergence (9-21 iterations)
* Moderate performance across other metrics, but lesss consistent in balancing routes
5. Conclusion
* RLGA is best for large-scale problems requiring fast, high-quality distance optimization
* GASA is a solid middle ground for moderate performance and fast learning
* GA with Local Search is the most balanced and reliable solution in this study - combining strong optimization, fast convergence, and fair workload distribution.
======
## VI. Key Contributions
* Demonstrated the effectiveness of Reinforcement Learning Genetic Algorithm (RLGA) in solving complex m-TSP problems, outperforming traditional methods like GASA and NSGA-II in both solution quality and adaptability.
* Highlighted the importance of adaptive learning mechanisms over static parameter tuning, showing that dynamic adjustment during the search process significantly enhances optimization performance and avoids local optima.
* Provided practical insights for cold chain logistics, espeically in short shelf-life product distribution like Vinamilk's yogurt, where RLGA offers a scalable and flexible alternative to traditional routing software.
* Proposed a hybrid RLGA-based optimization framework that balances exploration and exploitation, enabling robust route planning under real-world operational constraints.
* Offered a reproducible and extensible codebase to support further research and application in multi-agent delivery systems and logistics optimization.

======
## VII. Limitations
* Limited benchmarking
* Fixed number of delivery agents (m)
* Lack of real-world constraints

======
## VIII. Detailed Process
The research and implementation process consists of the following key steps:

1. Data Collection & Preparation
* Target locations: 126 Vinamilk retail stores in the Hanoi area were selected as delivery points
* Coordinate acquisition: Latitude and longtitude were manually collected to ensure precision
* Distance matrix generation:
  * Used the OSRM locally deployed to compute real driving distances between all store pairs
  * Due to OSRM's request limit, the dataset was divided into smaller blocks and queried in batches
  * Results were aggregated into a complete NxN distance matrix (dist_matrix), representing travel costs between all locations.

2. Problem Formulation
The problem is defined as a Multiple Traveling Salesman Problem (m-TSP) with:
* A single depot (warehouse)
* ```m``` delivery agents (salesmen)
* A goal of minimizing maximum route length to balance delivery workload and reduce overall delivery time

3. GA-based Solution
   
3.1. Initial population
* A set of random permutations of all delivery points (excluding the depot) is generated.
* Each permutation represents a possible delivery sequence.
  
3.2. Fitness Evaluation
* Each route is split into ```m``` sub-routes using a greedy split algorithm, aiming for roughly equal route distances.
* The fitness score is the length of the longest sub-route, promoting balanced workloads.

3.3. Evolution Process
* Selection: Top individuals are chosen based on fitness.
* Crossover: New solutions are created by combining parts of selected parents.
* Local Search: A simple 2-opt swap is applied to improve individual routes.

3.4. Final Optimization
After all generations, the best individual is passed through a Dynamic Programming (DP) split, producing the optimal route division for ```m``` agents.

4. Benchmarking & Evaluation
The proposed method is compared with:
* NSGA-II (Multi-objective evolutionary algorithm)
* RLGA (Reinforcement Learning Genetic Algorithm)
* GASA (Genetic Algorithm with Simulated Annealing)

Evaluation metrics:
* Total route distance
* Maximum route length
* Balance score (variance in route lengths)
* Execution time
* Number of iterations to converge
* Convergence speed

5. Result Analysis
* Results showed that RLGA consistently outperforms GASA and NSGA-II in most metrics.
* The adaptive learning mechanism in RLGA improves convergence quality and avoids premature local optima.
* The proposed approach is practical for cold chain logistics, enabling flexible and efficient delivery planning for perishable goods like yogurt.

====
## Authors and Supervisor ##

1. Vuong Thuy Linh
   * Email: linhvuong.31221026306@st.ueh.edu.vn
   * Github: https://github.com/thilinnd
2. Nguyen Vu Thanh Giang
   * Email: giangnguyen.31231026898@st.ueh.edu.vn
   * Github: https://github.com/thanhgiang0607
3. Le Thuy Tien
   * Email: tienle.31231020076@st.ueh.edu.vn
   * Github: https://github.com/ThuyTien1209
4. Huynh Minh Thu
   * Email: thuhuynh.31231020999@st.ueh.edu.vn
   * Github: https://github.com/HuynhThu04
5. Dang Ngoc Hoang Thanh - Supervisor


