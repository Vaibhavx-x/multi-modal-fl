import matplotlib.pyplot as plt

def generate_plot(is_dark_mode: bool):
    # 1. The Data
    alpha_labels = ['α = 0.01\n(Highly Non-IID)', 'α = 0.10\n(Moderately Non-IID)', 'α = 1.00\n(Near-IID)']
    accuracies = [63.52, 73.61, 83.87]

    # 2. Setup the Figure
    if is_dark_mode:
        plt.style.use('dark_background')
        bg_color = '#0d1117' # GitHub dark mode background
        text_color = 'white'
        grid_color = 'gray'
        colors = ['#ff6b6b', '#4dabf7', '#51cf66']
        filename = 'accuracy_vs_alpha_dark.png'
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        bg_color = 'white'
        text_color = 'black'
        grid_color = '#e0e0e0'
        colors = ['#e63946', '#457b9d', '#1d3557'] 
        filename = 'accuracy_vs_alpha_light.png'

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 3. Create the Bar Chart
    bars = plt.bar(alpha_labels, accuracies, color=colors, width=0.6)

    # 4. Add Data Labels on Top of Bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, 
                 f'{yval}%', ha='center', va='bottom', 
                 fontweight='bold', fontsize=12, color=text_color)

    # 5. Formatting the Graph
    plt.title('Federated Learning Accuracy vs. Data Heterogeneity', fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold', color=text_color)
    plt.ylim(50, 95)
    plt.xticks(fontsize=11, color=text_color)
    plt.yticks(fontsize=11, color=text_color)

    # Grid and spines
    if is_dark_mode:
        ax.grid(color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
    else:
        ax.grid(color=grid_color, linestyle='-', linewidth=0.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 6. Save
    plt.tight_layout()
    # Save with fully opaque background
    plt.savefig(filename, dpi=300, facecolor=bg_color, edgecolor='none', transparent=False)
    print(f"Plot saved successfully as '{filename}'")
    plt.close()

if __name__ == "__main__":
    generate_plot(is_dark_mode=False)
    generate_plot(is_dark_mode=True)
