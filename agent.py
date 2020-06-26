import numpy as np
import copy


class Agent:
    def __init__(self, k, m, mu=0.1, x=None, alpha=None, mutable_variables=None, rational=None, marker = None, memory = [], reciprocal = None):
        """
        Initializes an agent object for the Genetic Algorithm simulation.

        :param k: externality parameter. Must fulfill: -1 < k < 1
        :param m: self-utility parameter. Must fulfill m > 0
        :param mu: mutation standard deviation. Must fulfill mu > 0.
        :param x: a-priori strategy. Must fulfill x >= 0
        :param alpha: Altruism parameter. Must fulfill 0.5 <= alpha <= 1.0. If None, alpha is randomly initialized.
        :param mutable_variables: List of variables that are subject to mutations.
        """

        assert -1 < k < 1
        assert m > 0
        assert mu > 0
        #assert reprod_method == "utility" or "payoff"
        if x is not None:
            assert x >= 0
        if alpha is not None:
            assert 0.5 <= alpha <= 1
        else:
            alpha = np.random.uniform(0.5, 1.0)

        if mutable_variables is None:
            mutable_variables = []

        # Agent parameters (may be mutated)
        self.alpha = alpha
        self.x = x if x is not None else np.exp(np.random.uniform(0, 2))
        self.memory = memory
        self.marker = marker
        self.reciprocal = reciprocal
        # Agent hyperparameters (can't be mutated)
        self.k = k
        self.m = m
        self.rational = rational

        self.payoff = 0
        self.utility = 0
        

        # Mutation stuff
        for var in mutable_variables:
            assert var in ["alpha", "x"]
        self.mutable_params = mutable_variables
        self.mutation_std = mu

    def remember(self, agent_2): #the altruist incorporates the egoist into their memory
        
        if agent_2.alpha>self.alpha and agent_2.marker not in self.memory:
            self.memory.append(agent_2.marker)
        if agent_2.alpha<self.alpha and self.marker not in agent_2.memory:
            agent_2.memory.append(self.marker)
        
        

    def compute_payoff(self, x, y, alpha, beta):
        """
        Computes the payoff of the current agent given the action of the other agent. This is equation (1) from paper.

        :param x: Strategy of current agent
        :param y: Strategy of interacting agent
        :param alpha: Alpha of current agent
        :param beta: Alpha of interacting agent
        """

        if self.rational:
            self.payoff = x * (self.k * y + self.m - x)
        else:
            self.payoff = alpha* x * (self.k * y + self.m - x) + (1-beta)*y*(self.k*x + self.m -y)
            

    def compute_utility(self, u_1, u_2):
        """
        Computes the utility of the current agent given the utility of the other agent. This is equation (8) from paper.
        :param u_1: utility of current agent
        :param u_2: utility of interacting agent
        """

        self.utility = self.alpha * u_1 + (1 - self.alpha) * u_2

    def interact(self, agent_2):
        """
        Computes the payoff and utility of the interaction of the current agent with agent_2, for both players.

        :param agent_2: The agent with whom the current agent will be interacting
        :type agent_2: Agent
        """
        

        if self.rational:
            x, y = self.rational_strategies(agent_2)
        else: 
            x = self.x
            y = agent_2.x
        self.x = x
        agent_2.x = y

        if (self.marker in agent_2.memory and self.alpha>agent_2.alpha): #if agent 2 remembers agent 1 as an egoist, he will mimic
            beta = self.alpha
            alpha = self.alpha

        elif (agent_2.marker in self.memory and self.alpha<agent_2.alpha): #same but the other way around
            beta = agent_2.alpha
            alpha = agent_2.alpha
        else: #if they havent interacted before
            beta = agent_2.alpha
            alpha = self.alpha
            if self.reciprocal:
                self.remember(agent_2)
        
        

        self.compute_payoff(x, y, alpha, beta)
        agent_2.compute_payoff(y, x, beta, alpha)

        self.compute_utility(self.payoff, agent_2.payoff)
        agent_2.compute_utility(agent_2.payoff, self.payoff)

    def reproduce(self):
        """
        Makes an offspring agent with random mutations using the current agent as 'template'.

        :return: a new Agent with random mutations given the current agent.
        :rtype: Agent
        """

        # Copy the current agent
        offspring = copy.deepcopy(self)

        # Perform the mutations
        for param in self.mutable_params:
            mu = np.random.normal(loc=0.0, scale=self.mutation_std)
            if param == "x":
                offspring.x = offspring.x + mu * offspring.x
            elif param == "alpha":
                offspring.alpha = offspring.alpha + mu * offspring.alpha

        # Make sure new parameters fulfill the constraints
        offspring.check_param_constraints()

        return offspring
    


    def check_param_constraints(self):
        """
        Verifies that the current mutable parameters fulfill the constraints imposed. In case there's a violation, set
        the parameter to the closest acceptable value.
        """

        if self.x < 0:
            self.x = 0
        if self.alpha < 0.5:
            self.alpha = 0.5
        if self.alpha > 1:
            self.alpha = 1

    def rational_strategies(self, agent_2):
        """
        Computes the strategies that the agents will take (i.e. equation 12), which assumes mutual knowledge of alpha
        parameters, and rationality.
        :param agent_2: partner agent in the interaction
        :return: the optimal, rational strategies x and y (i.e. x, and x for agent_2)
        """
        if (self.marker in agent_2.memory  and self.alpha>agent_2.alpha): #if agent 2 remembers agent 1 as an egoist, he will mimic
            beta = self.alpha
            alpha = self.alpha

        elif (agent_2.marker in self.memory  and self.alpha<agent_2.alpha): #same but the other way around
            beta = agent_2.alpha
            alpha = agent_2.alpha
        else: #if they havent interacted before
            beta = agent_2.alpha
            alpha = self.alpha
        m = self.m
        k = self.k

        def nash_eq_x(a, b):
            return b * m * (2 * a + k) / (4 * a * b - k ** 2)

        x = nash_eq_x(alpha, beta)
        y = nash_eq_x(beta, alpha)

        return x, y