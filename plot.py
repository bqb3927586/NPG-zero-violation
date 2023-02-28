import numpy as np
import math

np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib

matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

## Loading the 'data'. This can be loaded to make the figure again.
Num_case = 40
length = 10000

objective = np.zeros([Num_case,length])
violation = np.zeros([Num_case,length])
objective_kappa = np.zeros([Num_case,length])
violation_kappa = np.zeros([Num_case,length])

for case in range(40):
    objective[case, :] = np.load('./data/objective/fa_objective_'+str(case)+'.npy')
    violation[case, :] = np.load('./data/violation/fa_violation_'+str(case)+'.npy')
    objective_kappa[case, :] = np.load('./data/objective/fa_objective_kappa_' + str(case) + '.npy')
    violation_kappa[case, :] = np.load('./data/violation/fa_violation_kappa_' + str(case) + '.npy')

mean_objective = np.mean(objective,axis=0)
mean_violation = np.mean(violation,axis=0)
mean_objective_kappa = np.mean(objective_kappa,axis=0)
mean_violation_kappa = np.mean(violation_kappa,axis=0)

std_objective = np.std(objective,axis=0)
std_violation = np.std(violation,axis=0)
std_objective_kappa = np.std(objective_kappa,axis=0)
std_violation_kappa = np.std(violation_kappa,axis=0)

max_objective = mean_objective + std_objective
min_objective = mean_objective - std_objective
max_violation = mean_violation + std_violation
min_violation = mean_violation - std_violation

max_objective_kappa = mean_objective_kappa + std_objective_kappa
min_objective_kappa = mean_objective_kappa - std_objective_kappa
max_violation_kappa = mean_violation_kappa + std_violation_kappa
min_violation_kappa = mean_violation_kappa - std_violation_kappa

# Plot optimality gap
episode = np.arange(1,8001,1)
episode2 = np.arange(1,8001,10)
f = plt.figure()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
plt.plot(episode,mean_objective_kappa[episode],'r-',linewidth=1, label='kappa=1')
plt.fill_between(episode2, min_objective_kappa[episode2], max_objective_kappa[episode2], color='r', alpha=0.3)
plt.plot(episode,mean_objective[episode],'b--',linewidth=1, label='kappa=0')
plt.fill_between(episode2, min_objective[episode2], max_objective[episode2], color='b', alpha=0.3)
plt.yticks(fontsize=15)
#plt.yticks(np.linspace(2.3, 2.5, 4, endpoint=True))
plt.xticks(fontsize=15)
#plt.xticks(np.linspace(0, len(gap_k20), 5, endpoint=True))
plt.xlabel('iteration',fontsize=15)
plt.ylabel('objective value',fontsize=15)
plt.grid()
plt.legend(loc=2)
f.savefig("Fig_fa_objective_comp_detail.pdf", bbox_inches='tight')

# Plot constraint violation
g = plt.figure()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
plt.plot(episode,mean_violation_kappa[episode],'r--',linewidth=1, label='kappa=1')
plt.fill_between(episode2, min_violation_kappa[episode2], max_violation_kappa[episode2], color='r', alpha=0.3)
plt.plot(episode,mean_violation[episode],'b-',linewidth=1, label='kappa=0')
plt.fill_between(episode2, min_violation[episode2], max_violation[episode2], color='b', alpha=0.3)

# zeroline = np.zeros(N)
# plt.plot(zeroline,linestyle='--', color='red')
plt.yticks(fontsize=15)
#plt.yticks(np.linspace(round(min(violation_k20), 1), round(max(violation_k20), 1), 4, endpoint=True))
plt.xticks(fontsize=15)
#plt.xticks(np.linspace(0, len(violation_k20), 5, endpoint=True))
plt.xlabel('iteration',fontsize=15)
plt.ylabel('constraint violation',fontsize=15)
plt.yscale('log')
plt.grid()
plt.legend(loc=0)
g.savefig("Fig_fa_violation_comp_detail.pdf", bbox_inches='tight')
