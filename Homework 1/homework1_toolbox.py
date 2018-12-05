import numpy as np
import matplotlib.pyplot as plt

class MDP:
    
    """
    This class generates a Markov decision process problem
    """
    
    def __init__(self, proba_trans, reward, gamma = 0.95):
        
        """
        Description
        -------------
        Constructor of the MDP class
        
        Attributes
        -------------
        n_states    : Int, number of states.
        n_actions   : Int, number of actions.
        proba_trans : np.array of shape (n_states, n_actions, n_states), the probability transition
                      matrix. 
                      1st dimension : current state.
                      2nd dimension : current action.
                      3rd dimension : next state.
        reward      : np.array of shape (n_states, n_actions), the reward of a current state and action,
                      it is actually the reward of current state and action averaged on the next states, by
                      the tansition probabilities.
                      1st dimension : current state.
                      2nd dimension : current action
        gamma       : Float, the discount factor.
        converged   : Boolean, the state of convergence.
        it          : Int, number of iterations until convergence.
                      
        Returns
        -------------
        self
        
        """

        self.proba_trans = proba_trans
        self.reward = reward
        self.n_states, self.n_actions = reward.shape
        self.gamma = gamma
        self.converged = False
        
    def bellman_operator(self, a, w):
        
        """
        Description
        -------------
        this function applies the Bellman operator to a vector.
        
        Parameters
        -------------
        a : Int in [0, n_actions - 1], the first action to be taken.
        w : np.array of shape (n_states,), the vector we want to apply the bellman operator on.
        
        Returns
        -------------
        Vector result of applying the Bellman operator to w
        """
        
        return self.reward[:, a] + self.gamma*np.dot(self.proba_trans[:, a, :], w)
    
    def bellman_operator_optimal(self, w):
        
        """
        Description
        -------------
        this function applies the optimal Bellman operator to a vector.
        
        Parameters
        -------------
        w : np.array of shape (n_states,), the vector we want to apply the optimal bellman operator on.
        
        Returns
        -------------
        Vector result of applying the optimal Bellman operator to w
        """
        
        return np.max(self.reward + self.gamma*np.dot(self.proba_trans, w), axis = 1)
    
    def policy_evaluation(self, policy):
        
        """
        Description
        -------------
        Evaluate the policy in an infinite time horizon with discount model.
        
        Parameters
        -------------
        policy : np.array of shape (self.n_states,), the policy we want to evaluate.
        
        Returns
        -------------
        V      : np.array of shape (self.n_states,), the value function applied on policy.
        """
        
        proba_policy = np.zeros((self.n_states, self.n_states))
        reward_policy = np.zeros(self.n_states)
        for i in range(self.n_states):
            proba_policy[i, :] = self.proba_trans[i, policy[i], :]
            reward_policy[i] = self.reward[i, policy[i]]
            
        return np.dot(np.linalg.inv(np.eye(self.n_states) - self.gamma*proba_policy), reward_policy)
        
        
        
    def value_iteration(self, epsilon = 1e-2, max_iter = 100):
        
        """
        Description
        -------------
        Perform value iteration to look for the optimal policy.
        
        Parameters
        -------------
        epsilon  : Float, the precision up to which we want a convergence.
        max_iter : Int, the maximum number of iterations.
        
        Returns
        -------------
        policy_optimal : 1D np.array of length self.n_states, where policy_optimal[i] is the optimal action
                         to be taken when i is the current state.
        value_optimal  : np.array of shape (n_states,), the optimal value function.
        
        """
        
        it = 0
        V = np.random.uniform(size = self.n_states)
        converged = False
        value_functions = [V]
        
        while (it < max_iter) and (not converged):
            V_next = self.bellman_operator_optimal(V)
            if np.linalg.norm(V_next - V, np.inf) < epsilon:
                converged = True
                
            V = V_next
            value_functions.append(V)
            it += 1
            
        self.converged = converged
        self.it = it
        self.value_functions = value_functions
        
        policy_optimal = np.argmax(self.reward + self.gamma*np.dot(self.proba_trans, V), axis = 1)
        value_optimal = self.policy_evaluation(policy_optimal)
        
        return policy_optimal, value_optimal
    
    def policy_iteration(self, max_iter = 100):
        
        """
        Description
        -------------
        Perform policy iteration to look for the optimal policy.
        
        Parameters
        -------------
        max_iter : Int, the maximum number of iterations.
        
        Returns
        -------------
        policy_optimal : 1D np.array of length self.n_states, where policy_optimal[i] is the optimal action
                         to be taken when i is the current state.
        value_optimal  : np.array of shape (n_states,), the optimal value function.
        
        """
        
        it = 0
        policy = np.zeros(self.n_states).astype(int)
        V = self.policy_evaluation(policy)
        value_functions = [V]
        converged = False
        
        while (it < max_iter) and (not converged):
            V = self.policy_evaluation(policy)
            value_functions.append(V)
            policy_next = (np.argmax(self.reward + self.gamma*np.dot(self.proba_trans, V), axis = 1)).astype(int)
            if (policy_next == policy).all():
                converged = True
                
            policy = policy_next
            it += 1
            
        self.converged = converged
        self.it = it
        self.value_functions = value_functions
        
        return policy, V
            
class RL:
    
    def __init__(self, gridworld):
        
        """
        Description
        -------------
        Constructor of RL class.
        
        Attributes
        -------------
        gridworld : object of class GridWorld.
        
        Returns
        -------------
        self
        
        """
        
        self.gridworld = gridworld
    
    def estimate_initial_distribution(self, n_samples = 1000):
        
        """
        Description
        -------------
        Estimate the initial distribution mu_0.
        
        Parameters
        -------------
        n_samples : Int, the number of samples to use. default is 100.
        
        Returns
        -------------
        mu_0 : np.array of size (self.gridworld.n_states,) holding the initial distribution probabilities of each state
        """
        
        mu_0 = np.zeros(self.gridworld.n_states)
        for i in range(n_samples):
            mu_0[self.gridworld.reset()] += 1
        
        mu_0 /= n_samples
        return mu_0
    
    def value_trajectory(self, policy, state_init, max_iter = 100):
        
        """
        Description
        -------------
        Compute value function of policy on state_init by simulating a trajectory.
        
        Parameters
        -------------
        policy     : np.array of shape (self.gridworld.n_states, len(self.gridworld.action_names)), it is a stochastic policy,
                     policy[i, j] holds the probability of taking action j from state i.
        state_init : Int in [0, self.gridworld.n_states - 1], the initial state.
        max_iter   : Int, maximum number of iterations to produce a trajectory.
        
        Returns
        -------------
        value      : Float, the computed value function of policy on state_init using the trajectory
        """
        
        gamma = self.gridworld.gamma
        value = 0
        it = 1
        state = state_init
        absorb = False
        # Given policy, we take a random action knowing the current state.
        while (it < max_iter) and (not absorb):
            action = np.argmax(np.random.multinomial(1, policy[state, :]))
            state, reward, absorb = self.gridworld.step(state, action)
            value += gamma**(it - 1)*reward
            it += 1
        
        return value
    
    def value_monte_carlo(self, policy, state_init, n_trajectories = 100, max_iter = 100):
        
        """
        Description
        -------------
        Estimate the value function of policy on state_init using Monte Carlo approximation.
        
        Parameters
        -------------
        policy         : np.array of shape (self.gridworld.n_states, len(self.gridworld.action_names)), it is a stochastic policy,
                         policy[i, j] holds the probability of taking action j from state i.
        state_init     : Int in [0, self.gridworld.n_states - 1], the initial state.
        n_trajectories : Int, the numnber of trajectories used for the estimation.
        max_iter       : Int, maximum number of iterations to produce a trajectory.
        
        Returns
        -------------
        Float, the estimated value function of policy on state_init
        """
        
        values = np.empty(n_trajectories)
        values.fill(state_init)
        values = values.astype(int)
        values = np.vectorize(self.value_trajectory, excluded = set([0, "max_iter"]))(policy, values, max_iter)
        return values.mean()
    
    def value_monte_carlo_vectorized(self, policy, states, n_trajectories = 100, max_iter = 100):
        
        """
        Description
        -------------
        Estimate the value function of policy on states (initial states) using Monte Carlo approximation.
        
        Parameters
        -------------
        policy         : np.array of shape (self.gridworld.n_states, len(self.gridworld.action_names)), it is a stochastic policy,
                         policy[i, j] holds the probability of taking action j from state i.
        states         : Container (list, np.array ...) containing integers in [0, self.gridworld.n_states - 1], the initial states.
        n_trajectories : Int, the numnber of trajectories used for the estimation.
        max_iter       : Int, maximum number of iterations to produce a trajectory.
        
        Returns
        -------------
        np.array of shape (len(states)), the estimated value function of policy on states (initial states)
        """
        
        return np.vectorize(self.value_monte_carlo, excluded = set([0, "n_trajectories", "max_iter"]))(policy, states, n_trajectories, max_iter)
    
    def greedy_eps_policy(self, q, state, epsilon = 1e-1):
        
        """
        Description
        -------------
        Take action in a state using state-action function q with an epsilon-greedy policy.
        
        Parameters
        -------------
        q       : np.array of shape (self.gridworld.n_states, len(self.gridworld.action_names)), the state-action function.
        state   : Int in [0, self.gridworld.n_states - 1], the current state.
        epsilon : Float, 1 - epsilon represents the probability of taking the action that maximizes q in state.
        
        Returns
        -------------
        action  : Int, the action to be taken in state.
        """
        
        available_actions = np.where(q[state, :] != -np.inf)[0]
        if len(available_actions) == 1:
            return available_actions[0]
        
        else:
            if np.random.binomial(1, 1 - epsilon) == 1:
                return np.argmax(q[state, :])

            else:
                # index of the action to be chosen randomly
                other_actions = available_actions[available_actions != np.argmax(q[state, :])]
                return other_actions[np.where(np.random.multinomial(1, np.ones(len(other_actions))/len(other_actions)) == 1)[0][0]]
    
    def q_learning(self, epsilon = 1e-1, cooling = False, max_iter = 100, episodes = 100, decay = 1):
        
        """
        Description
        -------------
        Perform Q-Learning algorithm to estimate the optimal policy and optimal state-action value function.
        
        Parameters
        -------------
        epsilon  : Float, 1 - epsilon represents the probability of taking the action that maximizes the state-action function in a current state.
        cooling  : Boolean, if True perform a cooling schedule on epsilon, that way we shift to exploitation rather thatn exploration over time.
        max_iter : Int, maximum number of iterations to produce a trajectory.
        episodes : Int, number of times we reset the trajectory due to reaching a terminal state or the maximum number of iterations per trajectory.
        decay    : Float, number in ]0.5, 1], the power of the learning rate.
        
        Returns
        -------------
        policy : np.array of shape (self.gridworld.n_states, len(self.gridworld.action_names)), the learned stochastic policy.
        Q      : np.array of shape (self.gridworld.n_states, len(self.gridworld.action_names)), the estimated optimal state-action function.
        """
        
        n_visits = np.zeros((self.gridworld.n_states, len(self.gridworld.action_names))) # matrix of the number of visits of each state-action, 
                                                                                         # it will be used to compute the learning rate.
        gamma = self.gridworld.gamma
        # Initialize Q, at the beginning in each state x we take available actions with a uniform distribution.
        Q = - np.empty((self.gridworld.n_states, len(self.gridworld.action_names)))
        Q.fill(-np.inf)
        for state in range(self.gridworld.n_states):
#             Q[state, self.gridworld.state_actions[state]] = 1/len(self.gridworld.state_actions[state]) 
            Q[state, self.gridworld.state_actions[state]] = 0
            
        V_episodes = [] # Gather the estimated value functions at the end of each episode here.
        cumulated_rewards = [] # Gather the rewards on each episode.
        epsilon_init = epsilon
        cumulated_reward = 0
            
        for i in range(episodes):
            state = self.gridworld.reset()
            it = 0
            absorb = False
            while (it < max_iter) and (not absorb):
                action = self.greedy_eps_policy(Q, state, epsilon)
                next_state, reward, absorb = self.gridworld.step(state, action)
                delta = reward + gamma*max(Q[next_state, :]) - Q[state, action] # The temporal difference.
                n_visits[state, action] += 1
                alpha = 1/n_visits[state, action]**decay # The learning rate
                Q[state, action] = Q[state, action] + alpha*delta # Update the state-action function.
                cumulated_reward += (gamma**it)*reward
                it += 1
                state = next_state
                if cooling:
                    epsilon = 1/(1/epsilon_init + n_visits[state, action]) # This way, the more we visit a state-action, the more we exploit it.
            
            cumulated_rewards.append(cumulated_reward)
            greedy_policy = np.argmax(Q, axis = 1)
            V_episodes.append(np.max(Q, axis = 1))
            
            
        return greedy_policy, V_episodes, cumulated_rewards
                
                
                 
        