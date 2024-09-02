import numpy as np
import itertools

'''
Deb (2001):
Step 1 Combine parent and offspring populations and create Rt = Pt U Qt.
Perform a non-dominated sorting to R, and identify different fronts: F«,
i = 1,2, ..., etc.

Step 2 Set new population PH1 = 0. Set a counter i = 1.

Until IPH 11+ IFi 1 < N, perform Pt +1 = PH1 UF; and i = i + 1.

Step 3 Perform the Crowding-sort(Fi, <cl procedure (described below on
page 236) and include the most widely spread (N - IPH11l solutions by
using the crowding distance values in the sorted F, to PH1.

Step 4 Create offspring population QH1 from PH1 by using the crowded
tournament selection, crossover and mutation operators.

'''



class Detector():
    def __init__(self, mean, stdev, dim):
        self.vector = np.random.normal(mean, stdev, dim)
        self.compute_fitness()
        self.reach = 2
    
    def mutate(self):
        ''' mutates using gaussian distribution mutation '''
        stdev = self.A / 10.0
        self.x += np.random.normal(0, stdev)
        if self.x > self.A:
            self.x = self.A
        elif self.x < -self.A:
            self.x = -self.A
        
    def recombine(self, parent):
        offspring = Detector(self.A)
        offspring.x = np.mean([self.x, parent.x])
        return offspring

    def compute_fitness(self):
        self.f1 = self.x**2
        self.f2 = (self.x - 2)**2    

    def dominated_by(self, comp_detector):
        ''' checks if this individual is dominated by the comp_individual '''
        if (comp_detector.f1 < self.f1 and comp_detector.f2 <= self.f2) or (comp_detector.f1 <= self.f1 and comp_detector.f2 < self.f2):
            return True
        else:
            return False
    
class NSGAII():

    def __init__(self, n, A):
        # require even n
        if n % 2 != 0:
            raise Exception("Population size n has to be an even number")
        self.pop = np.empty(n*2, dtype=Detector) # contains population P and population Q (both of size n)
        self.n = n
        self.pareto_fronts_log = []
        self.pareto_fronts = []
        self.f1_max = -np.inf
        self.f1_min = np.inf
        self.f2_max = -np.inf
        self.f2_min = np.inf

        # creates a parent population P and adds the first (random) offspring population Q
        for i in range(n*2):
            ind = Detector(A)
            ind.initialize()
            self.pop[i] = ind    
        
        self.store_min_and_max()  

    def reset_distances(self):
        for ind in self.pop:
            ind.distance = 0

    def evolve(self):
        pareto_fronts = self.non_domination_sorting()
        self.pareto_fronts_log.append(pareto_fronts)
        mating_pool = self.create_mating_pool()
        pair_wise_parents = self.crowded_tournament_selection()
        offspring_pool = self.create_offspring_pool(pair_wise_parents)
        self.pop = np.concatenate((mating_pool, offspring_pool))
        self.reset_distances()

    def store_min_and_max(self):
        ''' Stores max and min f_m values in population '''

        for ind in self.pop:
            if ind.f1 > self.f1_max:
                self.f1_max = ind.f1
            if ind.f1 < self.f1_min:
                self.f1_min = ind.f1
            if ind.f2 > self.f2_max:
                self.f2_max = ind.f2
            if ind.f2 < self.f2_min:
                self.f2_min = ind.f2


    def non_domination_sorting(self):
        ''' Sorts the population into non-domination ranks. Rank 1 means non-dominated, rank 2 means domination by rank 1 individuals '''
        # [[0, 1], [0, 2], [1, 2]]
        # iterate through population (P) and new offspring (Q) and compare all against all, put them in pareto fronts
        indexes = np.arange(0, self.n*2)
        comparison_list = list(itertools.combinations(indexes, 2))
        comparison_result = [[] for _ in range(self.n*2)] # preparing list of individuals dominating individual 
        
        for comp_indexes in comparison_list:
            if self.pop[comp_indexes[0]].dominated_by(self.pop[comp_indexes[1]]):
                comparison_result[comp_indexes[0]].append(comp_indexes[1])
            elif self.pop[comp_indexes[1]].dominated_by(self.pop[comp_indexes[0]]):
                comparison_result[comp_indexes[1]].append(comp_indexes[0])

        # find pareto fronts
        self.pareto_fronts = []     
        c = self.n*2
        while c > 0:
            optimal_pareto_individuals = []
            for i, comp_result in enumerate(comparison_result):
                if comp_result != None and len(comp_result) == 0:
                    optimal_pareto_individuals.append(i)
                    comparison_result[i] = None
            c -= len(optimal_pareto_individuals)
            self.pareto_fronts.append(self.ParetoFront(self, [self.pop[i] for i in optimal_pareto_individuals], optimal_pareto_individuals))
            comparison_result = [[item for item in sublist if item not in self.pareto_fronts[-1].individual_indexes] if sublist is not None else None for sublist in comparison_result]
        return self.pareto_fronts

    def create_mating_pool(self):
        mating_pool = np.empty(self.n, dtype=Detector)
        f_counter = 0 # front counter
        c = 0 # individual counter
        for f in range(len(self.pareto_fronts)):
            f_counter = f
            if len(self.pareto_fronts[f].individuals_sorted_by_distance) + c < self.n:
                for ind in self.pareto_fronts[f].individuals_sorted_by_distance:
                    mating_pool[c] = ind
                    c += 1
            else:
                break
        
        last_front_individuals = []
        for ind in self.pareto_fronts[f_counter].individuals_sorted_by_distance:
            mating_pool[c] = ind
            last_front_individuals.append(ind)
            c += 1
            if c >= self.n:
                break 

        self.pareto_fronts = self.pareto_fronts[0:f_counter]
        self.pareto_fronts.append(self.ParetoFront(self, last_front_individuals, None))

        return mating_pool        
    
    def crowded_tournament_selection(self):
        ''' creates a pairwise list of parents to be recombined '''
        pareto_fronts_participating = np.random.choice(np.arange(0, len(self.pareto_fronts)), self.n*2, replace=True)
        pairwise_parents = np.empty(self.n, dtype=Detector)
        c = 0 
        for i in range(0,len(pareto_fronts_participating)-1, 2):
            front_a_index = pareto_fronts_participating[i]
            front_b_index = pareto_fronts_participating[i+1]
            if front_a_index < front_b_index:
                t = self.pareto_fronts[front_a_index].individuals
                pairwise_parents[c] = self.pareto_fronts[front_a_index].individuals[np.random.randint(len(self.pareto_fronts[front_a_index].individuals))]
            elif front_a_index > front_b_index:
                pairwise_parents[c] = self.pareto_fronts[front_b_index].individuals[np.random.randint(len(self.pareto_fronts[front_b_index].individuals))]
            else:
                # same pareto front, find individual with largest crowding distance, winner is the lowest from crowding sorted individuals
                winner_index = np.min(np.random.choice(np.arange(0, len(self.pareto_fronts[pareto_fronts_participating[i]].individuals_sorted_by_distance)), 2, replace=True)) # EFFICIENCY? WHAT IF ONLY ONE INDIVIDUAL IN A FRONT? 
                pairwise_parents[c] = self.pareto_fronts[pareto_fronts_participating[i]].individuals_sorted_by_distance[winner_index]
            c += 1 
        
        return pairwise_parents

    def create_offspring_pool(self, pairwise_parents):
        offspring_pool = np.empty(len(pairwise_parents), dtype=Detector)

        for i in range(0, len(pairwise_parents)-1, 2):
            offspring1 = pairwise_parents[i].recombine(pairwise_parents[i+1])
            offspring1.mutate()
            offspring1.compute_fitness()
            offspring_pool[i] = offspring1 
            offspring2 = pairwise_parents[i].recombine(pairwise_parents[i+1])
            offspring2.mutate()
            offspring2.compute_fitness()
            offspring_pool[i+1] = offspring2
        return offspring_pool

    class ParetoFront():
        ''' creates sorted list for each objective '''
        def __init__(self, outer_population, individuals, individual_indexes):
            self.pop = outer_population # reference to outer class population array
            self.individual_indexes = individual_indexes
            self.individuals = individuals
            self.f1_sorted_list = sorted(self.individuals, key=lambda individual: individual.f1)
            self.f2_sorted_list = sorted(self.individuals, key=lambda individual: individual.f2)
            self.calculate_crowding_distance()

        def calculate_crowding_distance(self):
            ''' calculates the crowding distance for individuals in each front '''
            # set boundary solutions
            self.f1_sorted_list[0].distance = np.inf
            self.f2_sorted_list[0].distance = np.inf
            self.f1_sorted_list[-1].distance = np.inf
            self.f2_sorted_list[-1].distance = np.inf

            # iterate through front and set crowding distance
            for i in range(1, len(self.f1_sorted_list)-1):
                self.f1_sorted_list[i].distance = self.f1_sorted_list[i].distance + (self.f1_sorted_list[i+1].f1 - self.f1_sorted_list[i-1].f1) / (self.pop.f1_max - self.pop.f1_min)
            for i in range(1, len(self.f2_sorted_list)-1):
                self.f2_sorted_list[i].distance = self.f2_sorted_list[i].distance + (self.f2_sorted_list[i+1].f2 - self.f2_sorted_list[i-1].f2) / (self.pop.f2_max - self.pop.f2_min)

            # sort based on crowding distance
            self.individuals_sorted_by_distance = sorted(self.individuals, key=lambda individual: -individual.distance)            

def main():
    d = Detector(0, 2, 300)
    

if __name__ == "__main__":
    main()
        