import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training_data.csv')
filtered_data = data[data['Epoch'] % int(len(data['Epoch'])/75) == 1] # Here, the more is divided, the less information is plotted

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(filtered_data['Epoch'], filtered_data['Accuracy'], marker='o', color='b', label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(data['Epoch'][::(int(len(data['Epoch'])/10/10)*10)] - 1) # I added some manipulation to get correct xticks.
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(filtered_data['Epoch'], filtered_data['TrainLoss'], marker='o', color='r', label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.xticks(data['Epoch'][::(int(len(data['Epoch'])/10/10)*10)] - 1)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()