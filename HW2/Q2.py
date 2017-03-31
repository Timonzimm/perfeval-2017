import numpy as np
from scipy.stats import uniform,norm,t,expon,beta
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

uniform_ = uniform(0, 50)
normal = norm(0, 0.5)
student_3 = t(3)
student_10 = t(10)
exp_0dot5 = expon(0.5)
exp_2 = expon(2)
beta_10_90 = beta(10,90)
beta_100_10 = beta(100,10)

fig, ax = plt.subplots(1, 1)
x = np.linspace(-2, 60, 500)
ax.plot(x, uniform_.pdf(x), 'k-', lw=2, label='Uniform(0,50)')
ax.legend(loc='best', frameon=False)
plt.show()


fig, ax = plt.subplots(1, 1)
x = np.linspace(-10, 10, 500)
ax.plot(x, normal.pdf(x), 'k-', lw=2, label='Normal(0, 0.5)')
ax.legend(loc='best', frameon=False)
plt.show()

fig, ax = plt.subplots(1, 1)
x = np.linspace(-10, 10, 500)
ax.plot(x, student_3.pdf(x), 'k-', color='red', lw=2, label='Student(3)')
ax.plot(x, student_10.pdf(x), 'k-', lw=2, label='Student(10)')
ax.legend(loc='best', frameon=False)
plt.show()

fig, ax = plt.subplots(1, 1)
x = np.linspace(-5, 15, 500)
ax.plot(x, exp_0dot5.pdf(x), 'k-', color='red', lw=2, label='Exp(0.5)')
ax.plot(x, exp_2.pdf(x), 'k-', lw=2, label='Exp(2))')
ax.legend(loc='best', frameon=False)
plt.show()

fig, ax = plt.subplots(1, 1)
x = np.linspace(-0.5, 1.5, 1000)
ax.plot(x, beta_10_90.pdf(x), 'k-', color='red', lw=2, label='Beta(10, 90)')
ax.plot(x, beta_100_10.pdf(x), 'k-', lw=2, label='Beta(100, 10)')
ax.legend(loc='best', frameon=False)
plt.show()

r_uniform_ = uniform_.rvs(size=1000)
r_normal = normal.rvs(size=1000)
r_student_3 = student_3.rvs(size=1000)
r_student_10 = student_10.rvs(size=1000)
r_exp_0dot5 = exp_0dot5.rvs(size=1000)
r_exp_2 = exp_0dot5.rvs(size=1000)
r_beta_10_90 = beta_10_90.rvs(size=1000)
r_beta_100_10 = beta_100_10.rvs(size=1000)

standard_normal = norm(0, 1)


fig, axarr = plt.subplots(3, 3)

stats.probplot(r_uniform_, dist=standard_normal, plot=axarr[0, 0])
axarr[0, 0].set_title("Uniform(0, 50)")
stats.probplot(r_normal, dist=standard_normal, plot=axarr[0, 1])
axarr[0, 1].set_title("Normal(0, 0.5)")
stats.probplot(r_student_3, dist=standard_normal, plot=axarr[0, 2])
axarr[0, 2].set_title("Student(3)")
stats.probplot(r_student_10, dist=standard_normal, plot=axarr[1, 0])
axarr[1, 0].set_title("Student(10)")
stats.probplot(r_exp_0dot5, dist=standard_normal, plot=axarr[1, 1])
axarr[1, 1].set_title("Exp(0.5)")
stats.probplot(r_exp_2, dist=standard_normal, plot=axarr[1, 2])
axarr[1, 2].set_title("Exp(2)")
stats.probplot(r_beta_10_90, dist=standard_normal, plot=axarr[2, 0])
axarr[2, 0].set_title("Beta(10, 90)")
stats.probplot(r_beta_100_10, dist=standard_normal, plot=axarr[2, 1])
axarr[2, 1].set_title("Beta(100, 10)")

axarr[2, 2].axis('off')

plt.suptitle('Normal Q-Q plots with respect to different distributions')

plt.show()