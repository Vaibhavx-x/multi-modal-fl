import matplotlib.pyplot as plt

# 1. The Data
alpha_labels = ['α = 0.01\n(Highly Non-IID)', 'α = 0.10\n(Moderately Non-IID)', 'α = 1.00\n(Near-IID)']
accuracies = [63.52, 73.61, 83.87]

# 2. Setup the Figure
plt.style.use('dark_background') # Better for GitHub dark mode
fig, ax = plt.subplots(figsize=(9, 6))

# Set the background to match GitHub's dark mode color
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# 3. Create the Bar Chart
# Using brighter pastel colors to pop against the dark background
colors = ['#ff6b6b', '#4dabf7', '#51cf66'] 
bars = plt.bar(alpha_labels, accuracies, color=colors, width=0.6)

# 4. Add Data Labels on Top of Bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, 
             f'{yval}%', ha='center', va='bottom', 
             fontweight='bold', fontsize=12, color='white')

# 5. Formatting the Graph
plt.title('Federated Learning Accuracy vs. Data Heterogeneity', fontsize=16, fontweight='bold', pad=20, color='white')
plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold', color='white')
plt.ylim(50, 95) # Starting at 50 to emphasize the performance difference
plt.xticks(fontsize=11, color='lightgray')
plt.yticks(fontsize=11, color='lightgray')

# Optional: soften the grid lines
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True) # Put grid behind bars

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')

# 6. Save and Show
plt.tight_layout()
plt.savefig('accuracy_vs_alpha.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print("Plot saved successfully as 'accuracy_vs_alpha.png'")
