import matplotlib.pyplot as plt
import numpy as np

# Podaci
serije = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10']
accuracy = [0.9675, 0.95, 0.9225, 0.905, 0.9225, 0.8925, 0.8825, 0.8775, 0.855, 0.8225]
precision = [0.95169, 0.94117, 0.92462, 0.91752, 0.91219, 0.88292, 0.89637, 0.86124, 0.84134, 0.86440]
recall = [0.985, 0.96, 0.92, 0.89, 0.935, 0.905, 0.865, 0.9, 0.875, 0.765]
f1_score = [0.96805, 0.95049, 0.9223, 0.90355, 0.92345, 0.89382, 0.8804, 0.88019, 0.85784, 0.81167]

# Boje za svaku metriku
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']

# Stvaranje grafa
fig, ax = plt.subplots()
bar_width = 0.2
bar_positions = np.arange(len(serije))

bar1 = ax.bar(bar_positions - bar_width, accuracy, bar_width, label='Accuracy', color=colors[0])
bar2 = ax.bar(bar_positions, precision, bar_width, label='Precision', color=colors[1])
bar3 = ax.bar(bar_positions + bar_width, recall, bar_width, label='Recall', color=colors[2])
bar4 = ax.bar(bar_positions + 2 * bar_width, f1_score, bar_width, label='F1 Score', color=colors[3])

# Postavke grafa
ax.set_xticks(bar_positions)
ax.set_xticklabels(serije)
ax.set_xlabel('Serija')
ax.set_ylabel('Vrijednosti')
ax.set_title('Metrike klasifikacije za svaku seriju')
ax.legend()

# Prikaz grafa
plt.show()

