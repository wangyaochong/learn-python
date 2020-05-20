from matplotlib import pyplot as plt
import matplotlib
import random

interval = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90]
width = [5, 5, 5, 5, 5, 5, 5, 5, 5, 15, 30, 60]
quantity = [836, 2737, 3723, 3926, 3596, 1438, 3273, 642, 824, 613, 215, 47]
plt.bar(range(12), quantity, width=1)
_x = [i - 0.5 for i in range(13)]
plt.xticks(_x, interval + [150])
plt.show()
