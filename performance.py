import json
import matplotlib.pyplot as plt
import numpy as np

with open('chief_metrics.json') as file:
    data1 = json.load(file)

with open('worker_1_metrics.json') as file:
    data2 = json.load(file)

x = np.arange(1, len(data1["loss"])+1)
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title("Chief")
plt.plot(x, data1["loss"], x, data1["accuracy"])
plt.xlabel("EPOCH")
plt.ylabel("metrics")
plt.legend(["Loss", "Accuracy"])
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Worker")
plt.plot(x, data2["loss"], x, data2["accuracy"])
plt.xlabel("EPOCH")
plt.ylabel("metrics")
plt.legend(["Loss", "Accuracy"])
plt.grid(True)
plt.show()