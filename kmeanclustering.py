import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
# Load the generated crisis data from the CSV file
crisis_data = pd.read_csv("community_crises_data.csv")
warnings.filterwarnings("ignore")

# Preprocess the data: Combine relevant text fields into a single column
crisis_data['Text'] = crisis_data['What'] + ' ' + crisis_data['Where'] + ' ' + crisis_data['How']

# Convert the text data into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(crisis_data['Text'])

# Perform K-means clustering
num_clusters = 5  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Assign cluster labels to the data
crisis_data['Cluster'] = kmeans.labels_

# Print the number of crises assigned to each cluster
print("Number of crises assigned to each cluster:")
print(crisis_data['Cluster'].value_counts())

# Save the clustered data to a new CSV file
clustered_csv_file = "clustered_community_crises_data.csv"
crisis_data.to_csv(clustered_csv_file, index=False)

print(f"Clustered crisis data saved to '{clustered_csv_file}'")
