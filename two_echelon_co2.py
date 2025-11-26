"""
2-Echelon Network Distribution Strategy for Perishable Goods
Objective: Minimize CO2 Emissions

This module implements:
1. CO2-based objective function with vehicle selection
2. Initialization via clustering + center of gravity
3. ALNS optimization for refrigerator location updates
4. Customer decision: go to nearest delivery point (depot or refrigerator)

CO2 Parameters (from Subject3.pdf):
- Truck (frigorific): 311 g CO2 / (tonne * km), capacity 21 t
- Car: 772 g CO2 / (tonne * km), capacity 1.5 t  
- Bicycle: 0 g CO2, capacity 100 kg
- Refrigerator storage: 42 g CO2 / (tonne * day)
- Bicycle constraint: demand <= 500 kg AND round-trip <= 6 km

Map units: Kilometers (km)
"""

import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
import matplotlib.pyplot as plt
from copy import deepcopy
import random


# =============================================================================
# CO2 EMISSION PARAMETERS
# =============================================================================

@dataclass
class CO2Parameters:
    """CO2 emission parameters for different vehicles and operations."""
    # Vehicle capacities
    truck_capacity_kg: float = 21000  # 21 tonnes
    car_capacity_kg: float = 1500     # 1.5 tonnes
    bicycle_capacity_kg: float = 100  # 100 kg
    
    # CO2 emissions (g per tonne per km)
    truck_co2_per_tonne_km: float = 311.0
    car_co2_per_tonne_km: float = 772.0
    bicycle_co2_per_tonne_km: float = 0.0
    
    # Refrigerator storage emissions
    refrigerator_co2_per_tonne_day: float = 42.0
    
    # Bicycle usage constraints
    bicycle_max_demand_kg: float = 500.0
    bicycle_max_round_trip_km: float = 6.0
    
    # Map scale: 1 unit = how many km
    map_unit_to_km: float = 0.1  # 1 unit = 1 km
    
    def can_use_bicycle(self, demand_kg: float, round_trip_distance_hm: float) -> bool:
        """Check if bicycle can be used for this delivery."""
        round_trip_km = round_trip_distance_hm * self.map_unit_to_km
        return (demand_kg <= self.bicycle_max_demand_kg and 
                round_trip_km <= self.bicycle_max_round_trip_km)
    
    def get_client_pickup_co2(self, demand_kg: float, distance_hm: float) -> Tuple[float, str]:
        """
        Calculate CO2 for a client picking up goods (second echelon).
        Returns (co2_grams, vehicle_type)
        
        Client uses bicycle if possible, otherwise car.
        Round trip distance is 2 * distance_hm.
        """
        round_trip_hm = 2 * distance_hm
        round_trip_km = round_trip_hm * self.map_unit_to_km
        demand_tonnes = demand_kg / 1000.0
        
        if self.can_use_bicycle(demand_kg, round_trip_hm):
            return 0.0, "bicycle"
        else:
            # Use car
            co2 = self.car_co2_per_tonne_km * demand_tonnes * round_trip_km
            return co2, "car"
    
    def get_truck_delivery_co2(self, demand_kg: float, distance_hm: float) -> float:
        """
        Calculate CO2 for truck delivery (first echelon).
        Round trip from depot to refrigerator.
        """
        round_trip_km = 2 * distance_hm * self.map_unit_to_km
        demand_tonnes = demand_kg / 1000.0
        return self.truck_co2_per_tonne_km * demand_tonnes * round_trip_km
    
    def get_refrigerator_storage_co2(self, demand_kg: float, days: float = 1.0) -> float:
        """Calculate CO2 for refrigerator storage."""
        demand_tonnes = demand_kg / 1000.0
        return self.refrigerator_co2_per_tonne_day * demand_tonnes * days


# Default parameters
DEFAULT_CO2_PARAMS = CO2Parameters()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Instance:
    """Data structure for a problem instance."""
    dimension: int
    capacity: int
    coordinates: np.ndarray  # Shape: (n_points, 2), index 0 is depot
    demands: np.ndarray      # Shape: (n_points,), index 0 is depot (demand=0)
    
    @property
    def depot_coords(self) -> np.ndarray:
        return self.coordinates[0]
    
    @property
    def client_coords(self) -> np.ndarray:
        return self.coordinates[1:]
    
    @property
    def client_demands(self) -> np.ndarray:
        return self.demands[1:]
    
    @property
    def n_clients(self) -> int:
        return self.dimension - 1
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return min and max coordinates for the instance."""
        all_coords = self.coordinates
        return np.min(all_coords, axis=0), np.max(all_coords, axis=0)


@dataclass
class Solution:
    """
    Solution for the 2-echelon network problem.
    
    Decision variable: refrigerator_positions (continuous x, y)
    Derived: client assignments based on nearest delivery point
    """
    refrigerator_positions: np.ndarray   # Shape: (n_refrigerators, 2)
    
    # Computed fields (set after evaluation)
    client_assignments: np.ndarray = field(default=None)  # -1 = depot, 0..n-1 = refrigerator
    client_distances: np.ndarray = field(default=None)    # Distance to assigned point
    client_vehicles: np.ndarray = field(default=None)     # Vehicle type per client
    refrigerator_demands: np.ndarray = field(default=None)
    depot_demand: float = 0.0
    
    # CO2 breakdown
    co2_first_echelon: float = 0.0    # Truck depot -> refrigerators
    co2_second_echelon: float = 0.0   # Clients -> delivery points
    co2_refrigerator_storage: float = 0.0  # Refrigerator storage
    total_co2: float = float('inf')
    
    # Also keep distance for reference
    total_distance: float = 0.0
    
    @property
    def n_refrigerators(self) -> int:
        return len(self.refrigerator_positions)
    
    def copy(self) -> 'Solution':
        """Create a deep copy of the solution."""
        new_sol = Solution(
            refrigerator_positions=self.refrigerator_positions.copy()
        )
        if self.client_assignments is not None:
            new_sol.client_assignments = self.client_assignments.copy()
            new_sol.client_distances = self.client_distances.copy()
            new_sol.client_vehicles = self.client_vehicles.copy() if self.client_vehicles is not None else None
            new_sol.refrigerator_demands = self.refrigerator_demands.copy()
            new_sol.depot_demand = self.depot_demand
            new_sol.co2_first_echelon = self.co2_first_echelon
            new_sol.co2_second_echelon = self.co2_second_echelon
            new_sol.co2_refrigerator_storage = self.co2_refrigerator_storage
            new_sol.total_co2 = self.total_co2
            new_sol.total_distance = self.total_distance
        return new_sol


# =============================================================================
# EVALUATION WITH CO2 OBJECTIVE
# =============================================================================

def kmeans_clustering(points: np.ndarray, n_clusters: int, 
                      max_iterations: int = 100, 
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple K-Means clustering implementation.
    
    Args:
        points: Array of shape (n_points, n_features)
        n_clusters: Number of clusters
        max_iterations: Maximum iterations
        random_state: Random seed
    
    Returns:
        labels: Cluster assignment for each point
        centroids: Cluster centers
    """
    np.random.seed(random_state)
    n_points = len(points)
    
    # Initialize centroids using k-means++
    centroids = np.zeros((n_clusters, points.shape[1]))
    
    # First centroid is random
    first_idx = np.random.randint(n_points)
    centroids[0] = points[first_idx]
    
    # Remaining centroids with k-means++ selection
    for k in range(1, n_clusters):
        distances = cdist(points, centroids[:k])
        min_distances = np.min(distances, axis=1)
        probabilities = min_distances ** 2
        probabilities /= probabilities.sum()
        next_idx = np.random.choice(n_points, p=probabilities)
        centroids[k] = points[next_idx]
    
    # Iterate
    labels = np.zeros(n_points, dtype=int)
    
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        distances = cdist(points, centroids)
        new_labels = np.argmin(distances, axis=1)
        
        # Check convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        
        # Update centroids
        for k in range(n_clusters):
            cluster_points = points[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
    
    return labels, centroids


def evaluate_solution(instance: Instance, solution: Solution, 
                     co2_params: CO2Parameters = DEFAULT_CO2_PARAMS) -> Solution:
    """
    Evaluate a solution by computing client assignments and total CO2 emissions.
    
    Each client goes to the nearest delivery point (depot or any refrigerator).
    Vehicle selection is based on demand and distance constraints.
    """
    n_clients = instance.n_clients
    n_refrig = solution.n_refrigerators
    depot = instance.depot_coords
    
    # Compute distances from each client to depot
    client_to_depot = np.linalg.norm(instance.client_coords - depot, axis=1)
    
    # Compute distances from each client to each refrigerator
    client_to_refrig = cdist(instance.client_coords, solution.refrigerator_positions)
    
    # For each client, find minimum distance delivery point
    solution.client_assignments = np.zeros(n_clients, dtype=int)
    solution.client_distances = np.zeros(n_clients)
    solution.client_vehicles = np.empty(n_clients, dtype=object)
    
    for i in range(n_clients):
        # All options: depot (index 0) + refrigerators (indices 1..n_refrig)
        all_distances = np.concatenate([[client_to_depot[i]], client_to_refrig[i]])
        min_idx = np.argmin(all_distances)
        
        if min_idx == 0:
            solution.client_assignments[i] = -1  # Depot
            solution.client_distances[i] = client_to_depot[i]
        else:
            solution.client_assignments[i] = min_idx - 1  # Refrigerator index
            solution.client_distances[i] = all_distances[min_idx]
    
    # Compute demands at each delivery point
    solution.refrigerator_demands = np.zeros(n_refrig)
    solution.depot_demand = 0.0
    
    for i in range(n_clients):
        if solution.client_assignments[i] == -1:
            solution.depot_demand += instance.client_demands[i]
        else:
            solution.refrigerator_demands[solution.client_assignments[i]] += instance.client_demands[i]
    
    # ==========================================================================
    # COMPUTE CO2 EMISSIONS
    # ==========================================================================
    
    # 1. Second echelon CO2: Clients picking up goods
    solution.co2_second_echelon = 0.0
    for i in range(n_clients):
        demand = instance.client_demands[i]
        distance = solution.client_distances[i]
        co2, vehicle = co2_params.get_client_pickup_co2(demand, distance)
        solution.co2_second_echelon += co2
        solution.client_vehicles[i] = vehicle
    
    # 2. First echelon CO2: Truck delivering to refrigerators
    depot_to_refrig = np.linalg.norm(solution.refrigerator_positions - depot, axis=1)
    solution.co2_first_echelon = 0.0
    for k in range(n_refrig):
        if solution.refrigerator_demands[k] > 0:
            co2 = co2_params.get_truck_delivery_co2(
                solution.refrigerator_demands[k], 
                depot_to_refrig[k]
            )
            solution.co2_first_echelon += co2
    
    # 3. Refrigerator storage CO2
    solution.co2_refrigerator_storage = 0.0
    for k in range(n_refrig):
        if solution.refrigerator_demands[k] > 0:
            co2 = co2_params.get_refrigerator_storage_co2(solution.refrigerator_demands[k])
            solution.co2_refrigerator_storage += co2
    
    # Total CO2
    solution.total_co2 = (solution.co2_first_echelon + 
                          solution.co2_second_echelon + 
                          solution.co2_refrigerator_storage)
    
    # Also compute total distance for reference
    total_first_echelon_dist = 0.0
    for k in range(n_refrig):
        if solution.refrigerator_demands[k] > 0:
            total_first_echelon_dist += 2 * depot_to_refrig[k]
    total_second_echelon_dist = 2 * np.sum(solution.client_distances)
    solution.total_distance = total_first_echelon_dist + total_second_echelon_dist
    
    return solution


# =============================================================================
# INITIALIZATION: CLUSTERING + CENTER OF GRAVITY
# =============================================================================

def initialize_clustering_cog(instance: Instance, n_refrigerators: int,
                               use_demand_weights: bool = True,
                               random_state: int = 42,
                               co2_params: CO2Parameters = DEFAULT_CO2_PARAMS) -> Solution:
    """
    Initialize refrigerator positions using clustering + center of gravity.
    
    Process:
    1. Cluster clients using K-Means
    2. For each cluster, compute center of gravity (optionally demand-weighted)
    3. Place refrigerators at these centers
    """
    # Step 1: Cluster clients using custom K-Means
    cluster_labels, kmeans_centroids = kmeans_clustering(
        instance.client_coords, n_refrigerators, random_state=random_state
    )
    
    # Step 2: Compute center of gravity for each cluster
    refrigerator_positions = np.zeros((n_refrigerators, 2))
    
    for k in range(n_refrigerators):
        cluster_mask = cluster_labels == k
        cluster_coords = instance.client_coords[cluster_mask]
        cluster_demands = instance.client_demands[cluster_mask]
        
        if len(cluster_coords) == 0:
            # Empty cluster: use K-Means centroid
            refrigerator_positions[k] = kmeans_centroids[k]
        elif use_demand_weights and np.sum(cluster_demands) > 0:
            # Demand-weighted center of gravity
            weights = cluster_demands / np.sum(cluster_demands)
            refrigerator_positions[k] = np.average(cluster_coords, axis=0, weights=weights)
        else:
            # Simple centroid
            refrigerator_positions[k] = np.mean(cluster_coords, axis=0)
    
    # Create and evaluate solution
    solution = Solution(refrigerator_positions=refrigerator_positions)
    return evaluate_solution(instance, solution, co2_params)


# =============================================================================
# ALNS DESTROY OPERATORS
# =============================================================================

def destroy_random_shift(solution: Solution, instance: Instance, 
                         magnitude: float = 5.0, rng: random.Random = None) -> Solution:
    """Randomly shift one refrigerator position."""
    new_sol = solution.copy()
    rng = rng or random.Random()
    
    k = rng.randint(0, new_sol.n_refrigerators - 1)
    shift = np.array([rng.gauss(0, magnitude), rng.gauss(0, magnitude)])
    new_sol.refrigerator_positions[k] += shift
    
    return new_sol


def destroy_swap_positions(solution: Solution, instance: Instance,
                           rng: random.Random = None) -> Solution:
    """Swap two refrigerator positions."""
    new_sol = solution.copy()
    rng = rng or random.Random()
    
    if new_sol.n_refrigerators < 2:
        return new_sol
    
    i, j = rng.sample(range(new_sol.n_refrigerators), 2)
    new_sol.refrigerator_positions[i], new_sol.refrigerator_positions[j] = \
        new_sol.refrigerator_positions[j].copy(), new_sol.refrigerator_positions[i].copy()
    
    return new_sol


def destroy_move_to_client(solution: Solution, instance: Instance,
                           rng: random.Random = None) -> Solution:
    """Move one refrigerator to a random client location."""
    new_sol = solution.copy()
    rng = rng or random.Random()
    
    k = rng.randint(0, new_sol.n_refrigerators - 1)
    client_idx = rng.randint(0, instance.n_clients - 1)
    new_sol.refrigerator_positions[k] = instance.client_coords[client_idx].copy()
    
    return new_sol


def destroy_move_toward_demand(solution: Solution, instance: Instance,
                               step_size: float = 0.5, rng: random.Random = None) -> Solution:
    """
    Move refrigerator toward the center of its assigned clients (Weiszfeld-like step).
    """
    new_sol = solution.copy()
    rng = rng or random.Random()
    
    k = rng.randint(0, new_sol.n_refrigerators - 1)
    
    # Find clients assigned to this refrigerator
    assigned_mask = solution.client_assignments == k
    if not np.any(assigned_mask):
        return new_sol
    
    assigned_coords = instance.client_coords[assigned_mask]
    assigned_demands = instance.client_demands[assigned_mask]
    
    # Compute demand-weighted centroid
    if np.sum(assigned_demands) > 0:
        weights = assigned_demands / np.sum(assigned_demands)
        target = np.average(assigned_coords, axis=0, weights=weights)
    else:
        target = np.mean(assigned_coords, axis=0)
    
    # Move toward target
    current = new_sol.refrigerator_positions[k]
    new_sol.refrigerator_positions[k] = current + step_size * (target - current)
    
    return new_sol


def destroy_perturb_all(solution: Solution, instance: Instance,
                        magnitude: float = 2.0, rng: random.Random = None) -> Solution:
    """Small perturbation to all refrigerator positions."""
    new_sol = solution.copy()
    rng = rng or random.Random()
    
    for k in range(new_sol.n_refrigerators):
        shift = np.array([rng.gauss(0, magnitude), rng.gauss(0, magnitude)])
        new_sol.refrigerator_positions[k] += shift
    
    return new_sol


def destroy_relocate_worst(solution: Solution, instance: Instance,
                           rng: random.Random = None) -> Solution:
    """
    Relocate the worst performing refrigerator to area with highest unserved demand.
    "Worst" = highest CO2 contribution or furthest from assigned clients.
    """
    new_sol = solution.copy()
    rng = rng or random.Random()
    
    # Find refrigerator with highest average client distance (inefficient)
    avg_distances = np.zeros(new_sol.n_refrigerators)
    for k in range(new_sol.n_refrigerators):
        assigned_mask = solution.client_assignments == k
        if np.any(assigned_mask):
            avg_distances[k] = np.mean(solution.client_distances[assigned_mask])
    
    worst_k = np.argmax(avg_distances)
    
    # Find clients far from any delivery point
    far_clients = np.argsort(solution.client_distances)[-5:]  # Top 5 farthest
    target_client = rng.choice(far_clients)
    
    # Move refrigerator toward this area
    new_sol.refrigerator_positions[worst_k] = instance.client_coords[target_client].copy()
    
    return new_sol


# =============================================================================
# ALNS REPAIR OPERATORS
# =============================================================================

def repair_weiszfeld_step(solution: Solution, instance: Instance,
                          iterations: int = 3, 
                          co2_params: CO2Parameters = DEFAULT_CO2_PARAMS) -> Solution:
    """
    Apply Weiszfeld algorithm steps to improve refrigerator positions.
    Weighted by demand and inverse distance.
    """
    new_sol = solution.copy()
    
    for _ in range(iterations):
        new_sol = evaluate_solution(instance, new_sol, co2_params)
        
        for k in range(new_sol.n_refrigerators):
            assigned_mask = new_sol.client_assignments == k
            if not np.any(assigned_mask):
                continue
            
            assigned_coords = instance.client_coords[assigned_mask]
            assigned_demands = instance.client_demands[assigned_mask]
            
            # Weiszfeld weights: demand / distance
            distances = np.linalg.norm(
                assigned_coords - new_sol.refrigerator_positions[k], axis=1
            )
            distances = np.maximum(distances, 1e-6)  # Avoid division by zero
            
            weights = assigned_demands / distances
            if np.sum(weights) > 0:
                new_pos = np.average(assigned_coords, axis=0, 
                                    weights=weights)
                new_sol.refrigerator_positions[k] = new_pos
    
    return evaluate_solution(instance, new_sol, co2_params)


def repair_centroid_update(solution: Solution, instance: Instance,
                           co2_params: CO2Parameters = DEFAULT_CO2_PARAMS) -> Solution:
    """Update each refrigerator to the demand-weighted centroid of its assigned clients."""
    new_sol = solution.copy()
    new_sol = evaluate_solution(instance, new_sol, co2_params)
    
    for k in range(new_sol.n_refrigerators):
        assigned_mask = new_sol.client_assignments == k
        if not np.any(assigned_mask):
            continue
        
        assigned_coords = instance.client_coords[assigned_mask]
        assigned_demands = instance.client_demands[assigned_mask]
        
        if np.sum(assigned_demands) > 0:
            weights = assigned_demands / np.sum(assigned_demands)
            new_sol.refrigerator_positions[k] = np.average(
                assigned_coords, axis=0, weights=weights
            )
        else:
            new_sol.refrigerator_positions[k] = np.mean(assigned_coords, axis=0)
    
    return evaluate_solution(instance, new_sol, co2_params)


def repair_none(solution: Solution, instance: Instance,
                co2_params: CO2Parameters = DEFAULT_CO2_PARAMS) -> Solution:
    """No repair, just re-evaluate the solution."""
    return evaluate_solution(instance, solution, co2_params)


# =============================================================================
# ALNS ALGORITHM
# =============================================================================

@dataclass
class ALNSConfig:
    """Configuration for ALNS algorithm."""
    max_iterations: int = 2000
    initial_temperature: float = 50.0
    cooling_rate: float = 0.997
    
    # Adaptive weight parameters
    sigma1: float = 33  # New best solution
    sigma2: float = 9   # Better than current
    sigma3: float = 3   # Accepted (worse but accepted)
    reaction_factor: float = 0.1
    
    random_state: int = 42


@dataclass
class ALNSResult:
    """Result from ALNS optimization."""
    best_solution: Solution
    best_co2: float
    initial_co2: float
    improvement: float  # Percentage
    history: List[float] = field(default_factory=list)
    destroy_weights: dict = field(default_factory=dict)
    repair_weights: dict = field(default_factory=dict)


def alns_optimize(instance: Instance, initial_solution: Solution,
                  config: ALNSConfig = None,
                  co2_params: CO2Parameters = DEFAULT_CO2_PARAMS) -> ALNSResult:
    """
    Run ALNS optimization to minimize CO2 emissions.
    """
    config = config or ALNSConfig()
    rng = random.Random(config.random_state)
    np.random.seed(config.random_state)
    
    # Define operators
    destroy_operators = {
        'random_shift': lambda s, i: destroy_random_shift(s, i, magnitude=5.0, rng=rng),
        'swap_positions': lambda s, i: destroy_swap_positions(s, i, rng=rng),
        'move_to_client': lambda s, i: destroy_move_to_client(s, i, rng=rng),
        'move_toward_demand': lambda s, i: destroy_move_toward_demand(s, i, step_size=0.5, rng=rng),
        'perturb_all': lambda s, i: destroy_perturb_all(s, i, magnitude=2.0, rng=rng),
        'relocate_worst': lambda s, i: destroy_relocate_worst(s, i, rng=rng),
    }
    
    repair_operators = {
        'weiszfeld': lambda s, i: repair_weiszfeld_step(s, i, iterations=3, co2_params=co2_params),
        'centroid': lambda s, i: repair_centroid_update(s, i, co2_params=co2_params),
        'none': lambda s, i: repair_none(s, i, co2_params=co2_params),
    }
    
    # Initialize weights
    destroy_weights = {name: 1.0 for name in destroy_operators}
    repair_weights = {name: 1.0 for name in repair_operators}
    
    # Track operator performance
    destroy_scores = {name: 0.0 for name in destroy_operators}
    repair_scores = {name: 0.0 for name in repair_operators}
    destroy_counts = {name: 0 for name in destroy_operators}
    repair_counts = {name: 0 for name in repair_operators}
    
    # Initialize
    current = initial_solution.copy()
    best = initial_solution.copy()
    initial_co2 = initial_solution.total_co2
    
    temperature = config.initial_temperature
    history = [initial_co2]
    
    print(f"Starting ALNS optimization...")
    print(f"Initial CO2: {initial_co2:.2f} g")
    
    for iteration in range(config.max_iterations):
        # Select operators (roulette wheel)
        destroy_name = roulette_select(destroy_weights, rng)
        repair_name = roulette_select(repair_weights, rng)
        
        destroy_op = destroy_operators[destroy_name]
        repair_op = repair_operators[repair_name]
        
        # Apply operators
        candidate = destroy_op(current, instance)
        candidate = repair_op(candidate, instance)
        
        # Update counts
        destroy_counts[destroy_name] += 1
        repair_counts[repair_name] += 1
        
        # Acceptance decision
        delta = candidate.total_co2 - current.total_co2
        
        score = 0
        if candidate.total_co2 < best.total_co2:
            # New best solution
            best = candidate.copy()
            current = candidate.copy()
            score = config.sigma1
        elif delta < 0:
            # Better than current
            current = candidate.copy()
            score = config.sigma2
        elif rng.random() < np.exp(-delta / temperature):
            # Accept worse solution (simulated annealing)
            current = candidate.copy()
            score = config.sigma3
        
        # Update operator scores
        destroy_scores[destroy_name] += score
        repair_scores[repair_name] += score
        
        # Cool down
        temperature *= config.cooling_rate
        
        # Record history
        history.append(best.total_co2)
        
        # Periodic output
        if (iteration + 1) % 500 == 0:
            print(f"Iteration {iteration + 1}: Best CO2 = {best.total_co2:.2f} g")
        
        # Update weights periodically
        if (iteration + 1) % 100 == 0:
            update_weights(destroy_weights, destroy_scores, destroy_counts, 
                          config.reaction_factor)
            update_weights(repair_weights, repair_scores, repair_counts,
                          config.reaction_factor)
            
            # Reset scores
            destroy_scores = {name: 0.0 for name in destroy_operators}
            repair_scores = {name: 0.0 for name in repair_operators}
            destroy_counts = {name: 0 for name in destroy_operators}
            repair_counts = {name: 0 for name in repair_operators}
    
    improvement = 100 * (initial_co2 - best.total_co2) / initial_co2 if initial_co2 > 0 else 0
    
    print(f"\nOptimization complete!")
    print(f"Final CO2: {best.total_co2:.2f} g (improvement: {improvement:.2f}%)")
    
    return ALNSResult(
        best_solution=best,
        best_co2=best.total_co2,
        initial_co2=initial_co2,
        improvement=improvement,
        history=history,
        destroy_weights=destroy_weights,
        repair_weights=repair_weights
    )


def roulette_select(weights: dict, rng: random.Random) -> str:
    """Roulette wheel selection based on weights."""
    names = list(weights.keys())
    values = [weights[n] for n in names]
    total = sum(values)
    r = rng.random() * total
    cumsum = 0
    for name, value in zip(names, values):
        cumsum += value
        if r <= cumsum:
            return name
    return names[-1]


def update_weights(weights: dict, scores: dict, counts: dict, 
                   reaction_factor: float):
    """Update operator weights based on performance."""
    for name in weights:
        if counts[name] > 0:
            performance = scores[name] / counts[name]
            weights[name] = weights[name] * (1 - reaction_factor) + \
                           reaction_factor * performance
            weights[name] = max(weights[name], 0.1)  # Minimum weight


# =============================================================================
# I/O FUNCTIONS
# =============================================================================

def parse_instance(filepath: str) -> Instance:
    """Parse instance from text file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    dimension = None
    capacity = None
    coordinates = []
    demands = []
    
    section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('DIMENSION'):
            parts = line.replace(':', ' ').split()
            dimension = int(parts[-1])
        elif line.startswith('CAPACITY'):
            parts = line.replace(':', ' ').split()
            capacity = int(parts[-1])
        elif line.startswith('NODE_COORD_SECTION'):
            section = 'coords'
        elif line.startswith('DEMAND_SECTION'):
            section = 'demand'
        elif line.startswith('DEPOT_SECTION') or line.startswith('EOF'):
            section = None
        elif section == 'coords':
            parts = line.split()
            if len(parts) >= 3:
                coordinates.append([float(parts[1]), float(parts[2])])
        elif section == 'demand':
            parts = line.split()
            if len(parts) >= 2:
                demands.append(float(parts[1]))
    
    return Instance(
        dimension=dimension,
        capacity=capacity,
        coordinates=np.array(coordinates),
        demands=np.array(demands)
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_solution(instance: Instance, solution: Solution, 
                  title: str = "2-Echelon Solution",
                  save_path: str = None,
                  show_vehicles: bool = True):
    """Plot a solution showing depot, refrigerators, and client assignments."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Color scheme
    colors = plt.cm.tab10.colors
    
    # Plot clients by assignment
    depot_clients = solution.client_assignments == -1
    
    # Clients going to depot
    if np.any(depot_clients):
        ax.scatter(instance.client_coords[depot_clients, 0],
                  instance.client_coords[depot_clients, 1],
                  c='gray', marker='o', s=50, alpha=0.6,
                  label='Clients → Depot')
    
    # Clients going to refrigerators
    for k in range(solution.n_refrigerators):
        assigned = solution.client_assignments == k
        if np.any(assigned):
            color = colors[k % len(colors)]
            
            # Different markers for bicycle vs car
            if show_vehicles and solution.client_vehicles is not None:
                bike_mask = assigned & np.array([v == 'bicycle' for v in solution.client_vehicles])
                car_mask = assigned & np.array([v == 'car' for v in solution.client_vehicles])
                
                if np.any(bike_mask):
                    ax.scatter(instance.client_coords[bike_mask, 0],
                              instance.client_coords[bike_mask, 1],
                              c=[color], marker='o', s=30, alpha=0.7,
                              edgecolors='green', linewidths=2)
                if np.any(car_mask):
                    ax.scatter(instance.client_coords[car_mask, 0],
                              instance.client_coords[car_mask, 1],
                              c=[color], marker='o', s=50, alpha=0.7)
            else:
                ax.scatter(instance.client_coords[assigned, 0],
                          instance.client_coords[assigned, 1],
                          c=[color], marker='o', s=50, alpha=0.7,
                          label=f'Clients → Refrig {k+1}')
    
    # Plot refrigerators
    for k in range(solution.n_refrigerators):
        color = colors[k % len(colors)]
        ax.scatter(solution.refrigerator_positions[k, 0],
                  solution.refrigerator_positions[k, 1],
                  c=[color], marker='^', s=200, edgecolors='black',
                  linewidths=2, zorder=5,
                  label=f'Refrigerator {k+1} ({solution.refrigerator_demands[k]:.0f} kg)')
    
    # Plot depot
    ax.scatter(instance.depot_coords[0], instance.depot_coords[1],
              c='blue', marker='s', s=300, edgecolors='black',
              linewidths=2, zorder=5, label='Depot')
    
    # Draw first echelon routes (depot to refrigerators)
    for k in range(solution.n_refrigerators):
        if solution.refrigerator_demands[k] > 0:
            ax.plot([instance.depot_coords[0], solution.refrigerator_positions[k, 0]],
                   [instance.depot_coords[1], solution.refrigerator_positions[k, 1]],
                   'g--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(f'{title}\nTotal CO2: {solution.total_co2:.2f} g')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_convergence(result: ALNSResult, save_path: str = None):
    """Plot ALNS convergence history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(result.history, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=result.best_co2, color='r', linestyle='--', 
               label=f'Best: {result.best_co2:.2f} g')
    ax.axhline(y=result.initial_co2, color='gray', linestyle=':', 
               label=f'Initial: {result.initial_co2:.2f} g')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('CO2 Emissions (g)')
    ax.set_title(f'ALNS Convergence (Improvement: {result.improvement:.2f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def print_solution_summary(instance: Instance, solution: Solution, 
                           title: str = "Solution Summary",
                           co2_params: CO2Parameters = DEFAULT_CO2_PARAMS):
    """Print detailed solution summary with CO2 breakdown."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"\nTotal CO2 Emissions: {solution.total_co2:.2f} g ({solution.total_co2/1000:.2f} kg)")
    print(f"  - First echelon (truck):    {solution.co2_first_echelon:.2f} g")
    print(f"  - Second echelon (clients): {solution.co2_second_echelon:.2f} g")
    print(f"  - Refrigerator storage:     {solution.co2_refrigerator_storage:.2f} g")
    
    print(f"\nTotal Distance: {solution.total_distance:.2f} km")
    
    print(f"\nRefrigerator positions:")
    for k in range(solution.n_refrigerators):
        pos = solution.refrigerator_positions[k]
        demand = solution.refrigerator_demands[k]
        assigned_count = np.sum(solution.client_assignments == k)
        print(f"  R{k+1}: ({pos[0]:.1f}, {pos[1]:.1f}) - {demand:.0f} kg, {assigned_count} clients")
    
    print(f"\nDepot: {solution.depot_demand:.0f} kg, {np.sum(solution.client_assignments == -1)} clients")
    
    if solution.client_vehicles is not None:
        n_bicycle = np.sum(np.array([v == 'bicycle' for v in solution.client_vehicles]))
        n_car = np.sum(np.array([v == 'car' for v in solution.client_vehicles]))
        print(f"\nVehicle usage: {n_bicycle} bicycles, {n_car} cars")


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_optimization(filepath: str, n_refrigerators: int = 3, 
                     max_iterations: int = 2000,
                     output_dir: str = None,
                     co2_params: CO2Parameters = DEFAULT_CO2_PARAMS):
    """
    Run the full optimization pipeline on an instance file.
    
    Args:
        filepath: Path to the instance file (e.g., "00.txt")
        n_refrigerators: Number of refrigerators to place
        max_iterations: ALNS iterations
        output_dir: Directory for output plots (default: ./outputs)
        co2_params: CO2 emission parameters
    
    Returns:
        Tuple of (instance, initial_solution, alns_result)
    """
    import os
    
    # Default output directory in current folder
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("2-ECHELON NETWORK WITH CO2 MINIMIZATION")
    print("=" * 60)
    
    # Print CO2 parameters
    print(f"\nCO2 Parameters:")
    print(f"  Truck: {co2_params.truck_co2_per_tonne_km} g/(t·km)")
    print(f"  Car: {co2_params.car_co2_per_tonne_km} g/(t·km)")
    print(f"  Bicycle: {co2_params.bicycle_co2_per_tonne_km} g/(t·km)")
    print(f"  Refrigerator: {co2_params.refrigerator_co2_per_tonne_day} g/(t·day)")
    print(f"  Bicycle max demand: {co2_params.bicycle_max_demand_kg} kg")
    print(f"  Bicycle max round-trip: {co2_params.bicycle_max_round_trip_km} km")
    
    # Load instance
    print(f"\nLoading instance: {filepath}")
    instance = parse_instance(filepath)
    
    print(f"Instance: {instance.n_clients} clients")
    print(f"Total demand: {np.sum(instance.client_demands):.0f} kg")
    print(f"Truck capacity: {instance.capacity} kg")
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Step 1: Initialize with clustering + center of gravity
    print(f"\n--- Initialization (K-Means + CoG, {n_refrigerators} refrigerators) ---")
    initial_solution = initialize_clustering_cog(
        instance, n_refrigerators, use_demand_weights=True, co2_params=co2_params
    )
    print_solution_summary(instance, initial_solution, "INITIAL SOLUTION", co2_params)
    
    # Plot initial solution
    initial_plot_path = os.path.join(output_dir, f'{base_name}_initial.png')
    plot_solution(instance, initial_solution, 
                  title=f"Initial Solution - {base_name}",
                  save_path=initial_plot_path)
    
    # Step 2: Optimize with ALNS
    print("\n--- ALNS Optimization ---")
    config = ALNSConfig(
        max_iterations=max_iterations,
        initial_temperature=50.0,
        cooling_rate=0.997,
        random_state=42
    )
    
    result = alns_optimize(instance, initial_solution, config, co2_params)
    
    print(f"\nOptimization complete!")
    print(f"Initial CO2: {result.initial_co2:.2f} g")
    print(f"Best CO2:    {result.best_co2:.2f} g")
    print(f"Improvement: {result.improvement:.2f}%")
    
    print_solution_summary(instance, result.best_solution, "OPTIMIZED SOLUTION", co2_params)
    
    # Plot optimized solution
    optimized_plot_path = os.path.join(output_dir, f'{base_name}_optimized.png')
    plot_solution(instance, result.best_solution,
                  title=f"Optimized Solution - {base_name}",
                  save_path=optimized_plot_path)
    
    # Plot convergence
    convergence_plot_path = os.path.join(output_dir, f'{base_name}_convergence.png')
    plot_convergence(result, save_path=convergence_plot_path)
    
    print(f"\nPlots saved to: {output_dir}")
    print(f"  - {base_name}_initial.png")
    print(f"  - {base_name}_optimized.png")
    print(f"  - {base_name}_convergence.png")
    
    return instance, initial_solution, result


if __name__ == "__main__":
    import sys
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='2-Echelon Network CO2 Optimization with ALNS')
    parser.add_argument('filepath', nargs='?', default=None,
                        help='Path to instance file (e.g., 00.txt)')
    parser.add_argument('-n', '--n_refrigerators', type=int, default=3,
                        help='Number of refrigerators (default: 3)')
    parser.add_argument('-i', '--iterations', type=int, default=2000,
                        help='ALNS iterations (default: 2000)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for plots (default: ./outputs)')
    
    args = parser.parse_args()
    
    if args.filepath is None:
        # Demo mode with random instance
        print("No file specified. Running demo with random instance...")
        print("Usage: python two_echelon_co2.py <filepath> [-n N_REFRIGERATORS] [-i ITERATIONS]")
        print("Example: python two_echelon_co2.py 00.txt -n 3 -i 2000")
        print("\n" + "=" * 60)
        
        # Create demo instance
        np.random.seed(42)
        n_clients = 30
        client_coords = np.random.rand(n_clients, 2) * 100
        depot_coords = np.array([[50, 90]])
        all_coords = np.vstack([depot_coords, client_coords])
        demands = np.concatenate([[0], np.random.randint(100, 2000, n_clients)])
        
        instance = Instance(
            dimension=n_clients + 1,
            capacity=10000,
            coordinates=all_coords,
            demands=demands
        )
        
        co2_params = DEFAULT_CO2_PARAMS
        
        # Run optimization
        initial_solution = initialize_clustering_cog(instance, args.n_refrigerators, 
                                                      co2_params=co2_params)
        print_solution_summary(instance, initial_solution, "INITIAL SOLUTION", co2_params)
        
        config = ALNSConfig(max_iterations=args.iterations)
        result = alns_optimize(instance, initial_solution, config, co2_params)
        
        print(f"\nImprovement: {result.improvement:.2f}%")
        print_solution_summary(instance, result.best_solution, "OPTIMIZED SOLUTION", co2_params)
        
    else:
        # Run on specified file
        run_optimization(
            filepath=args.filepath,
            n_refrigerators=args.n_refrigerators,
            max_iterations=args.iterations,
            output_dir=args.output
        )
