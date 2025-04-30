from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_labels_combined():

    # Load the data
    df_test = pd.read_excel('data/test_data_extended.xlsx', sheet_name='data')
    df_historic = pd.read_excel('data/historic/historic_data_with_reason.xlsx', sheet_name='data')

    # Combine the DataFrames
    df_combined = pd.concat([df_test[['decision_date', 'decision_label']],
                             df_historic[['decision_date', 'decision_label']]],
                            ignore_index=True)

    # Ensure decision_date is in datetime format
    df_combined['decision_date'] = pd.to_datetime(df_combined['decision_date'])

    # Extract year from decision_date
    df_combined['year'] = df_combined['decision_date'].dt.year

    # Group by year and decision_label, then count occurrences
    grouped_counts = df_combined.groupby(['year', 'decision_label']).size().unstack(fill_value=0)

    # Display the counts
    print("Year-wise counts of decision labels:")
    print(grouped_counts)

    # Plot the counts with labels
    plt.figure(figsize=(10, 6))
    ax = grouped_counts.plot(kind='bar', stacked=False, color=['#1f77b4', '#ff7f0e'], width=0.4)

    # Add count labels on top of each bar with increased font size
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:  # Only add label if count is non-zero
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # Center of the bar
                    height + 0.5,  # Slightly above the bar
                    f'{int(height)}',  # Count as integer
                    ha='center', va='bottom', fontsize=14  # Increased font size
                )

    # Customize other plot elements (optional: uncomment to increase font sizes)
    # plt.title('Counts of Allow and Dismiss Decisions by Year', fontsize=14)  # Title font size
    plt.xlabel('Decision Year', fontsize=12)  # X-axis label font size
    plt.ylabel('Count', fontsize=12)  # Y-axis label font size
    plt.legend(title='Decision Label', fontsize=12, title_fontsize=12)  # Legend font size
    plt.xticks(fontsize=14)  # X-axis tick font size
    plt.yticks(fontsize=14)  # Y-axis tick font size

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_labels_old(file_name='data/test_data_extended.xlsx'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_excel(file_name, sheet_name='data')

    # Group by decision_label only
    grouped_df = df.groupby('decision_label').size().reset_index(name='count')

    # Set the figure size
    plt.figure(figsize=(12, 8))

    # Define colors for the bars (e.g., green for Allow, red for Dismiss)
    colors = ['#2ecc71', '#e74c3c']  # You can customize these hex codes

    # Plotting the result using seaborn with custom colors
    bar_plot = sns.barplot(x='decision_label', y='count', data=grouped_df, palette=colors)

    # Increase font size for labels
    plt.xlabel('Decision Label', fontsize=18)  # Increased from 14 to 18
    plt.ylabel('Count', fontsize=18)  # Increased from 14 to 18

    # Increase font size for tick labels
    bar_plot.tick_params(labelsize=16)  # Increased from 12 to 16

    # Annotate the bars with the actual count values
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),  # Format the label
                          (p.get_x() + p.get_width() / 2., p.get_height()),  # Position
                          ha='center', va='center',
                          xytext=(0, 12),  # Slightly increased offset from 9 to 12 for better spacing
                          textcoords='offset points',
                          fontsize=26)  # Increased from 12 to 16

    # Show the plot
    plt.show()


def count_legal_areas():
    df = pd.read_excel('data/test_data.xlsx', sheet_name='data')
    legal_areas = df['legal_area'].tolist()

    all_tags = [word.strip() for area in legal_areas for word in area.split(',')]

    print(f'total unique legal_areas {len(set(all_tags))}')

    tag_counts = Counter(all_tags)
    sorted_tag_counts = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tag_counts:
        print(f"{tag} & {count} \\\\")


def get_model_local_counts():
    models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
              'Meta-Llama-3.1-8B-Instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']
    model_cutoff_dates = ['7/31/2023', '10/31/2023', '10/31/2023', '2/28/2023', '12/31/2023', '9/30/2021', '12/31/2023']

    df = pd.read_excel('data/test_data.xlsx', sheet_name='data')

    for model, filter_date in zip(models, model_cutoff_dates):
        local = df[df['decision_date'] > filter_date]
        print(f'{model} : {len(local)}')


def count_tokens(file_path):
    # Download NLTK data (if not already downloaded)
    nltk.download('punkt')

    # Load the Excel file

    df = pd.read_excel(file_path)

    # Ensure the 'background' column exists
    if 'judgment' not in df.columns:
        raise ValueError("Column 'background' not found in the dataset")

    # Function to count tokens using NLTK
    def count_tokens(text):
        if pd.isna(text):  # Handle missing values
            return 0
        return len(word_tokenize(str(text)))

    # Apply token counting to the 'background' column
    df['token_count'] = df['judgment'].apply(count_tokens)

    # Calculate statistics
    average_tokens = df['token_count'].mean()
    min_tokens = df['token_count'].min()
    max_tokens = df['token_count'].max()

    print(f"Average number of tokens per case in the 'background' column: {average_tokens:.2f}")
    print(f"Minimum number of tokens per case in the 'background' column: {min_tokens}")
    print(f"Maximum number of tokens per case in the 'background' column: {max_tokens}")


if __name__ == '__main__':
    # plot_labels('data/historic/historic_data_with_reason.xlsx')
    plot_labels_combined()

    # count_legal_areas()

    # get_model_local_counts()
    # file_path = 'data/test_data_extended.xlsx'
    # file_path = 'data/historic/historic_data_with_reason.xlsx'
    # count_tokens(file_path)
