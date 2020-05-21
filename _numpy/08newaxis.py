import numpy as np

data = np.arange(0, 10, 1)
print(data)
print("增维", data.reshape(-1, 1))
print("增加维度", data[:, np.newaxis].shape, data[:, np.newaxis])
print("增加维度", data[np.newaxis, :].shape, data[np.newaxis, :])
