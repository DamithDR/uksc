import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the data
data = {
    'Fluency': [0.421, 0.754, 0.698, 0.909],
    'Accuracy': [0.383, 0.797, 0.625, 0.865]
}
metrics = ['BLEU', 'ROUGE', 'BLEU_l', 'ROUGE_l']

# Create a DataFrame
df = pd.DataFrame(data, index=metrics)

# Split the data for each plot
df_1 = df.loc[['BLEU', 'ROUGE']]       # For Plot 1
df_2 = df.loc[['BLEU_l', 'ROUGE_l']]   # For Plot 2

# Rename the index for df_2 to match BLEU and ROUGE names
df_2.index = ['BLEU', 'ROUGE']

# Define annotation settings for font size
annot_kws = {"size": 14}  # Change font size of the values to 14
label_font_size = 16      # Define a variable for label font size

# # Plot 1: BLEU and ROUGE vs Fluency and Accuracy
# plt.figure(figsize=(6, 5))
# sns.heatmap(df_1, annot=True, cmap="Blues", cbar=True, fmt=".3f", annot_kws=annot_kws)
# # plt.title("BLEU and ROUGE vs Fluency and Accuracy")
# plt.xticks(fontsize=label_font_size)  # Increase font size of x-axis labels
# plt.yticks(fontsize=label_font_size)  # Increase font size of y-axis labels
# plt.tight_layout()
# plt.show()

# Plot 2: BLEU (from BLEU_l) and ROUGE (from ROUGE_l) vs Fluency and Accuracy
plt.figure(figsize=(6, 5))
sns.heatmap(df_2, annot=True, cmap="Blues", cbar=True, fmt=".3f", annot_kws=annot_kws)
# plt.title("BLEU and ROUGE vs Fluency and Accuracy (from BLEU_l and ROUGE_l)")
plt.xticks(fontsize=label_font_size)  # Increase font size of x-axis labels
plt.yticks(fontsize=label_font_size)  # Increase font size of y-axis labels
plt.tight_layout()
plt.show()
