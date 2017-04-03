import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

<<<<<<< HEAD
l70 = np.loadtxt('n50_l70.gz')[::1000]
#l108 = np.loadtxt('n50_l108.gz')[::1000]
=======
ll = np.loadtxt('n50_l70.gz')
ss = np.loadtxt('n50_l70_samples.gz')

l = ll[::1000]
s = ss[::1000]
>>>>>>> ed3bd1a16ee9603072d06f0918a19807dbe755d5

def first_positive(N):
  for i, n in enumerate(N):
    if n >= 0:
      return i, n
  return len(N), N[-1]

dev2 = np.diff(l, 2)

# compute the cut by taking the first time the second derivative is positive
cut_index, cut_value = first_positive(dev2)

plt.axvline(x=cut_index, label="Values", linestyle="--", color="red")
plt.plot(l[cut_index:])
#plt.plot(dev2, label="2nd derivative")
plt.show()

np.savetxt('n50_l70_no_tr.gz', ll[cut_index*1000:])
np.savetxt('n50_l70_samples_no_tr.gz', ss[cut_index*1000:])