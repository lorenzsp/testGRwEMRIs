# plot fisher results
import numpy as np
import matplotlib.pyplot as plt
# import results
M, mu, a, e0, gamma = np.loadtxt("FM_M_1e6_SNR_50_KerrEccEq_T_2.csv",skiprows=1,delimiter=',',unpack=True)

mask = (a == 0.95)
# mask the results
M = M[mask]
mu = mu[mask]
a = a[mask]
e0 = e0[mask]
gamma = gamma[mask]

# make 3d scatter plot with a, e0 on the x-y axis and gamma on the z-axis
fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')

# modify the marker size based on the size value of mu
leg_list = []
for mu_val,mm in zip(np.unique(mu),['o','P','D','X']):
    mask = (mu == mu_val)
    # size s depends on the value of mu
    scatter = ax.scatter(e0[mask], gamma[mask],  marker=mm, s=50, alpha=0.7)

# add legend for the different masses
for mu_val in np.unique(mu):
    leg_list.append(f'$\mu={mu_val}$')
plt.legend(leg_list)
ax.set_xlabel('e0')
ax.set_ylabel('sigma gamma')
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.tight_layout()
plt.savefig('figures/fisher_results.pdf')
