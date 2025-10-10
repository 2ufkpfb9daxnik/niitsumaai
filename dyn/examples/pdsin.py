import numpy as np
import pandas as pd
time = np.arange(0, 200, 0.1)
data = np.sin(time) + 0.1 * np.random.randn(len(time))  # sin波+ノイズ

df = pd.DataFrame({"time": time, "value": data})
df.set_index("time", inplace=True)

import matplotlib.pyplot as plt
df.plot()
plt.show()
