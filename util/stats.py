from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_labels():
    df = pd.read_excel('data/test_data.xlsx', sheet_name='data')

    grouped_df = df.groupby(['year', 'decision_label']).size().reset_index(name='count')
    # Set the figure size
    plt.figure(figsize=(12, 8))

    # Plotting the result using seaborn
    bar_plot = sns.barplot(x='year', y='count', hue='decision_label', data=grouped_df)

    # Increase font size for labels and title
    plt.xlabel('Year', fontsize=14)  # X-axis label
    plt.ylabel('Count', fontsize=14)  # Y-axis label
    # plt.title('Count of Decision Labels by Year', fontsize=16)  # Title

    # Increase font size for tick labels
    bar_plot.tick_params(labelsize=12)

    # Increase legend font size
    plt.legend(title='Decision Label', title_fontsize=14, fontsize=12)

    # Annotate the bars with the actual count values and increase annotation font size
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),  # Format the label
                          (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of the label
                          ha='center', va='center',
                          xytext=(0, 9),  # Offset the text slightly above the bar
                          textcoords='offset points',
                          fontsize=12)  # Font size of annotations

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


if __name__ == '__main__':
    # plot_labels()

    # count_legal_areas()

    get_model_local_counts()
