from main import train_df
import matplotlib.pyplot as plt
import seaborn as sns

custom_palette = ['blue', 'red', 'green', 'blue']
plt.figure(figsize=(8, 6))
sns.barplot(data=train_df, x="Category", y="Count", palette=custom_palette)
plt.title("Distribution of Tumor Types")
plt.xlabel("Tumor Type")
plt.ylabel("Count")
plt.show()