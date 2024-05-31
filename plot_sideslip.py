import numpy as np
import matplotlib.pyplot as plt

velocity = np.array([15, 20, 25, 30, 35])
sideslip = np.array([0.1, 0.046, 0.021, 0.01, 0.0035])
plt.figure()
plt.plot(velocity, sideslip, color = "k")
plt.scatter(velocity, sideslip, color = "k")
plt.xlabel("velocity (m/s)")
plt.ylabel("sideslip")
plt.show()