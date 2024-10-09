import pandas as pd

# Create a data dictionary based on your input
data = {
    'Language_Code': ['nyn', 'swa', 'teo', 'ibo', 'lgg', 'lug', 'eng'],
    'Language_Name': ['Runyankole', 'Swahili', 'Ateso', 'Igbo', 'Lugbara', 'Luganda', 'English'],
    'Jaccard_Similarity': [0.0124, 0.0157, 0.0171, 0.0421, 0.0418, 0.0156, 0.0374],
    'Overlapping_Words': [440, 387, 416, 460, 650, 446, 567],
    'Language_Family': [
        'Niger-Congo (Bantu)', 'Niger-Congo (Bantu)', 'Eastern Nilotic', 'Niger-Congo', 'Central Sudanic', 'Niger-Congo (Bantu)', 'Indo-European'
    ],
    'Number_of_Speakers_M': [3.22, None, 1.57, 25, 1.10, 5.56, 379],
    'Region': ['West', None, 'East', None, 'North', 'Central', None],
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Create a bar plot for Jaccard Similarity
plt.figure(figsize=(10, 6))
sns.barplot(x='Language_Name', y='Jaccard_Similarity', data=df, palette='viridis')

# Add labels and title
plt.xlabel('Language', fontsize=12)
plt.ylabel('Jaccard Similarity with Acholi', fontsize=12)
plt.title('Jaccard Similarity between Acholi and Other Languages', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

# Create a bar plot for Number of Overlapping Words
plt.figure(figsize=(10, 6))
sns.barplot(x='Language_Name', y='Overlapping_Words', data=df, palette='magma')

# Add labels and title
plt.xlabel('Language', fontsize=12)
plt.ylabel('Number of Overlapping Words with Acholi', fontsize=12)
plt.title('Overlapping Words between Acholi and Other Languages', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

# Create a bar plot with language families
plt.figure(figsize=(10, 6))
sns.barplot(x='Language_Name', y='Jaccard_Similarity', hue='Language_Family', data=df)

# Add labels and title
plt.xlabel('Language', fontsize=12)
plt.ylabel('Jaccard Similarity with Acholi', fontsize=12)
plt.title('Jaccard Similarity by Language Family', fontsize=14)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Adjust legend position
plt.legend(title='Language Family', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()

# Create a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Overlapping_Words', y='Jaccard_Similarity', hue='Language_Family', data=df, s=100)

# Add labels and title
plt.xlabel('Number of Overlapping Words', fontsize=12)
plt.ylabel('Jaccard Similarity with Acholi', fontsize=12)
plt.title('Overlapping Words vs. Jaccard Similarity', fontsize=14)

# Annotate points with language names
for i in range(df.shape[0]):
    plt.text(x=df['Overlapping_Words'][i]+5, y=df['Jaccard_Similarity'][i],
             s=df['Language_Name'][i], fontsize=9)

# Adjust legend position
plt.legend(title='Language Family', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()


# Handle missing values in Number_of_Speakers_M
df['Number_of_Speakers_M'] = df['Number_of_Speakers_M'].fillna(0)

# Create a bubble chart
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Overlapping_Words',
    y='Jaccard_Similarity',
    size='Number_of_Speakers_M',
    hue='Language_Family',
    data=df,
    sizes=(50, 500),
    alpha=0.7,
)

# Add labels and title
plt.xlabel('Number of Overlapping Words', fontsize=12)
plt.ylabel('Jaccard Similarity with Acholi', fontsize=12)
plt.title('Overlap vs. Jaccard Similarity (Bubble Size: Number of Speakers)', fontsize=14)

# Annotate points with language names
for i in range(df.shape[0]):
    plt.text(x=df['Overlapping_Words'][i]+5, y=df['Jaccard_Similarity'][i],
             s=df['Language_Name'][i], fontsize=9)

# Adjust legend position
plt.legend(title='Language Family', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()

# Example: Save the figure
plt.savefig('jaccard_similarity.png', dpi=300)
plt.show()
