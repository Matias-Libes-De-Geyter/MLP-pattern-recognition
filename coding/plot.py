import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training_data.csv')
filtered_data = data[data['Epoch'] % 5 == 0]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(filtered_data['Epoch'], filtered_data['Accuracy'], marker='o', color='b', label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(data['Epoch'][::100] - 1)
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(filtered_data['Epoch'], filtered_data['Loss'], marker='o', color='r', label='Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.xticks(data['Epoch'][::100] - 1)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()