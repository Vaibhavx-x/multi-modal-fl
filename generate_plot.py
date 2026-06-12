import matplotlib.pyplot as plt

# 1. The Data
alpha_labels = ['α = 0.01\n(Highly Non-IID)', 'α = 0.10\n(Moderately Non-IID)', 'α = 1.00\n(Near-IID)']
accuracies = [63.52, 73.61, 83.87]

# 2. Setup the Figure
plt.figure(figsize=(9, 6))
plt.style.use('seaborn-v0_8-whitegrid') # Gives a clean, professional grid background

# 3. Create the Bar Chart
# Using a gradient of colors to visually represent the progression
colors = ['#e63946', '#457b9d', '#1d3557'] 
bars = plt.bar(alpha_labels, accuracies, color=colors, width=0.6)

# 4. Add Data Labels on Top of Bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, 
             f'{yval}%', ha='center', va='bottom', 
             fontweight='bold', fontsize=12)

# 5. Formatting the Graph
plt.title('Federated Learning Accuracy vs. Data Heterogeneity', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
plt.ylim(50, 95) # Starting at 50 to emphasize the performance difference
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# 6. Save and Show
plt.tight_layout()
plt.savefig('accuracy_vs_alpha.png', dpi=300) # dpi=300 ensures it is crisp for GitHub
print("Plot saved successfully as 'accuracy_vs_alpha.png'")
