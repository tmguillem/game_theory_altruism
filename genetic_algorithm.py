import numpy as np
from tqdm import tqdm
from agent import Agent


class GA:
    """
    Class that implements the evolutionary algorithm (genetic algorithm, GA)
    """

    def __init__(self, n_population, m_iterations, k, m, mu, x_init, alpha_init, mutable_parameters, reprod_method, rational):
        """
        Initializes the genetic algorithm with a population of individuals.
        :param n_population: number of individuals in the simulation. Will remain constant
        :type n_population: int
        :param m_iterations: number of evolution iterations
        :type m_iterations: int
        :param k: externality parameter for agent interaction
        :param m: self-utility parameter for agent interaction
        :param mu: mutation standard deviation parameter for agent reproduction
        :param x_init: pre-set x parameter for the population. If None, x is random.
        :param alpha_init: pre-set alpha parameter for population. If None, alpha is random.
        :param mutable_parameters: list of the parameters of the agents that are subject to mutation. None if no
        mutable parameters.
        :reprod_method: utility or payoff based reproduction
        """

        assert divmod(n_population, 2)[1] == 0, "Population must be an even number so n/2 pairings can bee made."

        self.n = n_population
        self.m_iter = m_iterations
        #reproduction method
        self.reprod_method = reprod_method

        #rationality of actors
        self.rational = rational
        # Externality
        self.k = k
        # Self-utility
        self.m = m

        # Mutation std
        self.mutation = mu

        # x parameter (if None, randomized)
        self.x = x_init

        # Alpha parameter (if None, randomized)
        self.alpha = alpha_init

        # Percentage of lowest fit population to remove before reproduction [0, 1]
        self.x_remove = 0

        # Percentage of lowest fit population to replace during reproduction [0, 1]
        self.x_replace = 0.2

        self.mutable_params = mutable_parameters
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize a population of n agents.
        :return: A list with all the agents of the simulation
        """

        return [Agent(k=self.k, m=self.m, mu=self.mutation, x=self.x, alpha=self.alpha,
                      mutable_variables=self.mutable_params, rational = self.rational)
                for _ in range(self.n)]

    def make_pairs(self, algorithm):
        """
        Creates the interaction pairs between individuals, according to one of several pairing algorithm possibilities.
        :param algorithm: pairing algorithm used. One of: [random, ...]
        :return: a n/2 x 2 array with the paired indices along axis 1. E.g. for n=4: [[1, 3], [2, 4]]
        """

        if algorithm == "random":
            pairings = np.linspace(0, self.n - 1, self.n, dtype=int)
            np.random.shuffle(pairings)
            pairings = pairings.reshape(int(self.n/2), 2)
            return pairings

        else:
            raise NotImplementedError("Only 'random' method is currently implemented")

    def reproduce(self, reproduction_rate):
        """
        Reproduces the individuals of the current population until reaching a population of n again. The reproduction
        probability of each individual is given in the parameter. Must be a valid PDF. If the current population is 5
        agents and n=10, then these 5 individuals will pass on to the next generation, and yield a random offspring of 5
        more individuals.

        :param reproduction_rate: a valid probability density function of the same size as the current population, which
        represents the probability of each individual to reproduce.
        :type reproduction_rate: np.ndarray
        """

        # Check validity of PDF
        assert np.isclose(np.sum(reproduction_rate), 1.0)
        assert not np.any(reproduction_rate < 0)

        a = np.linspace(0, len(reproduction_rate) - 1, len(reproduction_rate), dtype=int)

        # Number of individuals replaced:
        n_replaced = int(self.n * self.x_replace)

        # Sample individuals to reproduce from PDF
        to_reproduce = np.random.choice(a, n_replaced, replace=True, p=reproduction_rate)

        offspring = [self.population[i].reproduce() for i in to_reproduce]
        self.population = self.population[n_replaced:] + offspring

    @staticmethod
    def normalize_pdf(params):
        """
        Normalizes a parameter vector so that it corresponds to a valid probability density function: unit sum, and >=0
        :return: the same vector of parameters but normalized to a valid probability density function. I.e. the sum of
        all elements equals one, and none of them is negative
        :rtype: np.ndarray
        """

        negatives = params[params < 0]
        params = params - np.min(negatives) if np.any(negatives) else params
        params = params + 0.0005
        params = params / np.sum(params)

        return params

    def run(self):
        """
        The core loop of the Evolutionary algorithm
        """

        population_summary = self.population_state_summary(initialize=True)

        for _ in tqdm(range(self.m_iter)):

            pairings = self.make_pairs(algorithm="random")

            # Compute the utilities and payoffs of each interaction pair
            for pairing in pairings:
                self.population[pairing[0]].interact(self.population[pairing[1]])

            # Summarize state of population
            population_summary = self.population_state_summary(current_summary=population_summary)

            # Sort payoffs from small to large. Use as fitness
            if self.reprod_method == "utility":
                payoffs = np.array([agent.utility for agent in self.population])
            elif self.reprod_method=="payoff":
                payoffs = np.array([agent.payoff for agent in self.population])
            fitness = self.normalize_pdf(payoffs)
            ind = np.argsort(fitness)

            # Sort individuals in terms of increasing fitness
            self.population = [self.population[i] for i in ind]
            fitness = fitness[ind]

            # Remove worst x% of population
            n_remove = int(self.n * self.x_remove)
            self.population = self.population[n_remove:]
            fitness = fitness[n_remove:]

            # Distribute lost fitness of removed individuals among surviving individuals
            fitness += (1 - np.sum(fitness)) / len(fitness)

            # Correct numerical error
            if np.any(fitness[fitness < 0]):
                fitness = fitness - np.min(fitness[fitness < 0])

            # Reproduce population using the fitness score as reproduction probability
            self.reproduce(fitness)

        return population_summary

    def population_state_summary(self, initialize=False, current_summary=None):
        """
        Resumes the state of the population in a dictionary. If initialize=True, then a new dictionary is created
        with no content. Else, the current_summary is extended with the current state.
        :param initialize: True if no evolution has happened yet and a new dictionary must be created
        :param current_summary: Used if initialize=False. Past states of the population. Will be extended with the new
        information from the current state
        :return: The updated (or new) dictionary with the population state information.
        """

        if initialize:
            # Initialize the summary dictionary, as no evolution has happened yet
            current_summary = {"x": np.zeros((0, self.n)),
                               "alpha": np.zeros((0, self.n)),
                               "v": np.zeros((0, self.n)),
                               "u": np.zeros((0, self.n))}

        assert isinstance(current_summary, dict)

        # Add new relevant content to the dictionary
        current_summary["x"] = np.concatenate(
            (current_summary["x"], np.array([agent.x for agent in self.population])[np.newaxis, :]), axis=0)

        current_summary["alpha"] = np.concatenate(
            (current_summary["alpha"], np.array([agent.alpha for agent in self.population])[np.newaxis, :]), axis=0)

        current_summary["v"] = np.concatenate(
            (current_summary["v"], np.array([agent.utility for agent in self.population])[np.newaxis, :]), axis=0)

        current_summary["u"] = np.concatenate(
            (current_summary["u"], np.array([agent.payoff for agent in self.population])[np.newaxis, :]), axis=0)

        return current_summary
