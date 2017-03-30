import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def first_positive(N):
  for i, n in enumerate(N):
    if n >= 0:
      return i, n
  return len(N), N[-1]

values = np.loadtxt('HW3/test_n50_l108.out')[::2000]
dev2 = np.diff(values, 2)

# compute the cut by taking the first time the second derivative is positive
cut_index, cut_value = first_positive(dev2)

plt.axvline(x=cut_index, label="Values", linestyle="--", color="red")
plt.plot(values)
#plt.plot(dev2, label="2nd derivative")
plt.show()