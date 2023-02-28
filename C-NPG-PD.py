
'''
Our code customizes the MDP code from:
https://github.com/jb3618columbia/Optimality-Guarantees-for-Policy-Gradient-Methods
in the paper:
Global Optimality Guarantees For Policy Gradient Methods, J. Bhandari and D. Russo
'''
'''
Natural Policy Gradient Primal-Dual Method with Function Approximation 
'''

import numpy as np
import math
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


'''
Input: theta as an array and 
Ouput: array of probabilites corresponding to each state: [\pi_{s_1}(.), ...., \pi_{s_n}(.)]
'''

def theta_to_policy(theta,s,a):
    prob = []

    for i in range(s):
        norm = 0
        flag = 0
        temp_exp = np.zeros([a, 1])
        for k in range(a):
            temp_exp[k] = np.dot(theta,feature[a*i + k,:])
        while np.max(temp_exp)>100:
            flag += 1
            temp_exp = temp_exp - 100
        #print('flag', flag)
        norm = np.sum(np.exp(temp_exp))
        for j in range(a):
            prob.append(np.exp(np.dot(theta,feature[a*i + j,:]) - flag * 100)/norm)

    return np.asarray(prob)


'''
Get \Pi_{\pi}((s) -> (s,a)) in R^{|S| x |S||A|} matrix corresponding to the policy \pi using the prob vector
'''
def get_Pi(prob,s,a):
    Pi = np.zeros((s,s*a))
    for i in range(s):
        Pi[i,i*a:(i+1)*a] = prob[i*a:(i+1)*a]
    
    return Pi

'''
Input: probability vector, state, action
Output: an approximation of \nabla_{\theta} \pi_{\theta}(s,a)

States go from 0 to n-1 and actons from 0 to m-1
'''
def grad_approx(signal,prob,prob_transition,feature,gamma,s,a,d,Ksamples):

    # probability transition in state action
    prob_transition_sa = np.zeros((s*a,s*a))
    for i in range(s):
        for j in range(a):
            temp = np.zeros(s*a)
            for k in range(s):
                 temp[k*a:(k+1)*a] = prob_transition[i*a+j,k]*prob[k*a:(k+1)*a]
            prob_transition_sa[i*a+j, :] = temp

    # run a mixed Markov chain for a stationary state action distribution
    nv_curr = np.zeros(s*a)
    nv = np.ones(s*a)/(s*a)
    nv_init = np.ones(s*a)/(s*a)
    prob_transition_init = np.zeros((s*a,s*a))
    nv_unif = np.ones(s*a)/(s*a)
    for i in range(s*a):
	# swap 1-gamma and gamma
        prob_transition_init[i, :] = (gamma)*prob_transition_sa[i, :] + (1-gamma)*nv_unif
    prob_transition_mix = prob_transition_init
    while np.linalg.norm(nv-nv_curr,ord=1) > 1e-6:
        nv = nv_curr
        nv_curr = nv_init.dot(prob_transition_mix)
        prob_transition_mix = np.matmul(prob_transition_mix,prob_transition_init)

    #
    w = np.zeros(d)
    alpha = 0.1
    for p in range(Ksamples):
        # draw state action pairs
        indx = np.random.choice(s*a, 1, p=nv)
        ap = np.random.choice(a, 1, p=nv[(indx[0]//a)*a : (indx[0]//a +1)*a]/sum(nv[(indx[0]//a)*a : (indx[0]//a +1)*a]))
        # estimate state action value
        i = indx[0]//a
        j = indx[0]%a
        qest = signal[i * a + j]
        length = np.random.geometric(1 - gamma, size=1)
        temp = np.random.choice(s, 1, p=prob_transition[i * a + j, :])
        state = temp[0]
        for r in range(length[0] - 1):
            # action = np.argmax(prob[state * a : (state + 1) * a])
            temp = np.random.choice(a, 1, p=prob[state * a: (state + 1) * a])
            action = temp[0]
            qest += signal[state * a + action]
            # state = np.argmax(prob_transition[state * a + action, :])
            temp = np.random.choice(s, 1, p=prob_transition[state * a + action, :])
            state = temp[0]
        # estimate state value
        # indx = np.random.randint(low=0, high=s, size=1)
        state = j
        length = np.random.geometric(1 - gamma, size=1)
        vest = 0
        for r in range(length[0] - 1):
            # action = np.argmax(prob[state * a : (state + 1) * a])
            temp = np.random.choice(a, 1, p=prob[state * a: (state + 1) * a])
            action = temp[0]
            vest += signal[state * a + action]
            # state = np.argmax(prob_transition[state * a + action, :])
            temp = np.random.choice(s, 1, p=prob_transition[state * a + action, :])
            state = temp[0]

        # estimate \nabla_{\theta} \pi_{\theta}(s,a)
        feature_e = np.zeros((s * a, d))
        for i in range(s):
            temp = np.zeros(d)
            for k in range(a):
                temp += feature[i * a + k, :] * prob[i * a + k]
            for j in range(a):
                feature_e[i * a + j, :] = feature[i * a + j, :] - temp

        G = (np.dot(feature_e[indx[0],:],w) - qest+vest)*feature_e[indx[0],:]
        # SGD update
        w = w - alpha * G

    return w/(Ksamples*(1-gamma))

'''
The overall reward function \ell(\theta)
'''
def ell(qvals,prob,rho):
    V = np.zeros(s)
    for i in range(s):
        V[i] = np.sum([qvals[i*a + j]*prob[i*a + j] for j in range(a)])
    
    ell = np.dot(V,rho)
    return ell

# Estimate state value function over random initialization
def svalue_est_init(prob_transition, prob, signal, gamma, s, a, Ksamples):
    gest_return = 0
    for k in range(Ksamples):
        temp = np.random.randint(low=0, high=s, size=1)
        state = temp[0]
        length = np.random.geometric(1 - gamma, size=1)
        gest = 0
        for j in range(length[0] - 1):
            # action = np.argmax(prob[state * a : (state + 1) * a])
            temp = np.random.choice(a, 1, p=prob[state * a: (state + 1) * a])
            action = temp[0]
            gest += signal[state * a + action]
            # state = np.argmax(prob_transition[state * a + action, :])
            temp = np.random.choice(s, 1, p=prob_transition[state * a + action, :])
            state = temp[0]
        gest_return += gest

    return gest_return / Ksamples


'''
The projection function
Input: a scalar 
Output: a scalar in the interval [0 100]
'''
def proj(scalar):
    offset = 1e8
    if scalar < 0:
        scalar = 0

    if scalar > offset:
        scalar = offset

    return scalar


'''
Policy iteration function
'''

def policy_iter(q_vals, s, a):
    new_policy = np.zeros(s * a)
    for i in range(s):
        idx = np.argmax(q_vals[i * a:(i + 1) * a])
        new_policy[i * a + idx] = 1

    return new_policy

for case in range(40):
    ## Random Seed
    np.random.seed(10)
    ## Problem Setup
    gamma = 0.8
    s, a = 10, 5 #20, 10 #10, 5 #50, 10
    '''
    Randomly generated probability transition matrix P((s,a) -> s') in R^{|S||A| x |S|}
    Each row sums up to one
    '''
    raw_transition = np.random.uniform(0,1,size=(s*a,s))
    prob_transition = raw_transition/raw_transition.sum(axis=1,keepdims=1)
    '''
    Random positive rewards
    '''
    reward = np.random.uniform(0,1,size=(s*a))
    '''
    Random positive utilities 
    '''
    utility = np.random.uniform(0,1,size=(s*a))-0.71
    '''
    Utility constraint offset b
    '''
    b = 0 #7 #8
    '''
    Start state distribution
    '''
    rho = np.ones(s)/s
    '''
    Random feature map
    '''
    d = 35
    # feature = np.random.uniform(0,1,size=(s*a,d))
    feature = np.zeros((s*a,d))
    k = 0
    for p in range(d):
        temp = np.zeros(d)
        temp[k] = 1
        feature[p,:] = temp
        k += 1
    k = 0

    raw_vec = np.random.uniform(0,1,size=(s,a))
    prob_vec = raw_vec/raw_vec.sum(axis=1,keepdims=1)
    init_policy = prob_vec.flatten()

    '''
    Run policy iteration to get the optimal policy and compute the constraint violation
    Feasibility checking: negative constraint violation leads to the Slater condition 
    '''

    curr_policy = np.random.uniform(0,1,size=(s*a))
    new_policy = init_policy
    # print('Starting policy',init_policy)

    while np.count_nonzero(curr_policy - new_policy) > 0:
        curr_policy = new_policy
        Pi = get_Pi(curr_policy,s,a)
        mat = np.identity(s*a) - gamma*np.matmul(prob_transition,Pi)
        q_vals = np.dot(np.linalg.inv(mat),utility)
        new_policy = policy_iter(q_vals,s,a)

    # print('Final policy',new_policy)

    ell_star = ell(q_vals,new_policy,rho)
    print('Feasibility checking: constraint violation', b-ell_star)


    '''
    Natural Primal-Dual Method
    '''
    import time

    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)

    N = 10000
    theta = np.random.uniform(0,1,size=d)
    dual = np.random.uniform(0,1,size=1)
    objective = []
    violation = []
    acc_avg_reward = 0
    acc_avg_violation = 0
    div_number = 100
    for k in range(N):
        prob = theta_to_policy(theta,s,a)
        Pi = get_Pi(prob,s,a)
        mat = np.identity(s*a) - gamma*np.matmul(prob_transition,Pi)

        qrvals = np.dot(np.linalg.inv(mat),reward)
        qgvals = np.dot(np.linalg.inv(mat),utility)
        qvals = qrvals+dual*qgvals

        # Sample-based estimations
        Ksamples = 100
        naturalgradient = grad_approx(reward + dual * utility, prob, prob_transition, feature, gamma, s, a, d, Ksamples)
        # primal natural gradient ascent
        # dual projected sub-gradient descent
        step = 0.1
        dualstep = 0.1
        theta += step*naturalgradient

        vgest_init = svalue_est_init(prob_transition, prob, utility, gamma, s, a, Ksamples)
        dual = proj(dual-dualstep*(vgest_init-0))



        avg_reward = ell(qrvals,prob,rho)
        avg_violation = b-ell(qgvals,prob,rho)
        acc_avg_reward += avg_reward
        acc_avg_violation += avg_violation
        objective.append(acc_avg_reward / (k + 1))
        violation.append(acc_avg_violation / (k + 1))
        if k % div_number == 0:
            print('k', k)
            print('theta', (theta))

            print('Average objective:',acc_avg_reward/(k+1))
            print('Average violation',acc_avg_violation/(k+1))



    ## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
    np.save('fa_objective_'+str(case)+'.npy',objective)
    np.save('fa_violation_'+str(case)+'.npy',violation)

'''
Generate plots in our paper

# Plot optimality gap
f = plt.figure()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.plot(np.array(objective))
plt.yticks(fontsize=20)
plt.yticks(np.linspace(round(min(objective),1), round(max(objective),1), 4, endpoint=True))
plt.xticks(fontsize=20)
plt.xticks(np.linspace(0, len(objective), 5, endpoint=True))
f.savefig("Fig_fa_objective_00.pdf",bbox_inches='tight')

# Plot constraint violation
g = plt.figure()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.plot(np.array(violation))
# zeroline = np.zeros(N)
# plt.plot(zeroline,linestyle='--', color='red')
plt.yticks(fontsize=20)
plt.yticks(np.linspace(round(min(violation),1), round(max(violation),1), 4, endpoint=True))
plt.xticks(fontsize=20)
plt.xticks(np.linspace(0, len(violation), 5, endpoint=True))
g.savefig("Fig_fa_violation_00.pdf",bbox_inches='tight')
'''