# Audio Database Indexing (Clustering)
The purpose of this project is to perform clustering (indexing) on an audio database.

## Audio Descriptors
The following audio descriptors have been defined in the code:

-Energy envelope  
-Temporal centroid  
-Effective duration  
-Global energy  
-Zero crossing rate  

## Normalizing a vector  
A function for normalizing a vector is included in the code. This function calculates the mean and standard deviation of the input vector and returns the normalized vector as well as the mean and standard deviation.

## Distance Descriptors
A function has been implemented to calculate the Euclidean distances between the test title and the learning data, and the k closest titles are determined.

## K-Nearest Neighbor Classification
A function has been implemented for K-nearest neighbor (KNN) classification. The function takes as input the test title, the normalized descriptors of the learning data, and the number of nearest neighbors (k) and returns the class of the title.

## Usage
The code can be used to perform audio database clustering by defining audio descriptors and normalizing the descriptors. The KNN classifier can then be applied to classify the test audio files based on their descriptors.
