import lab2

# # euclidian and manhattan distance
# print(lab2.find_euclidian_dist())
# print(lab2.find_manhattan_dist())


# KNN classifier


# # encoding
# sample_data = [["red", "small"], ["blue", "large"], ["green", "medium"]]
# encoded_data, label_mapping = lab2.label_encode_categorical(sample_data)
# print("Encoded data:", encoded_data)
# print("Label mapping:", label_mapping)

# # one hot encoding
# sample_data2 = [["red", "small"], ["blue", "large"], ["green", "medium"]]
# encoded_data, new_features = lab2.one_hot_encode_categorical(sample_data2)
# print("Encoded data:", encoded_data)
# print("New features:", new_features)



## applying it to an actaual dataset
# import arff  # to parse the dataset in the form of the arff file
#
# arff_file_path = "C:\\Backup\\amrita\\YEAR 2\\SEM-4\\Machine Learning\\Lab Sessions\\Lab Session2\\dataset1.arff"
# with open(arff_file_path, 'r') as f:
#     arff_data = arff.load(f)
#
# data = arff_data['data']
# attributes = arff_data['attributes']
#
#
# encoded_data, label_mapping = lab2.label_encode_categorical(data)
# print("encoded data : ", encoded_data)
# print("label mapping : ", label_mapping)

## knn classifier final working
import math
from collections import Counter

def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        return math.inf  # Return a very large value to indicate incomparability
    sum_of_squares = 0
    for i in range(len(v1)):
        sum_of_squares += (v1[i] - v2[i]) ** 2
    return math.sqrt(sum_of_squares)


def get_neighbors(training_data, test_instance):
    distances = []
    for train_data in training_data:
        distance = euclidean_distance(train_data[0], test_instance)
        distances.append((distance, train_data[1]))
    return distances


def Knn_classifier(training_data, test_instance, k_value):
    neighbors = get_neighbors(training_data,test_instance)
    neighbors.sort()
    nearest_neighbors = neighbors[:k_value]
    classes = [neighbor[1] for neighbor in nearest_neighbors]
    class_counter = Counter(classes)
    most_common_class = class_counter.most_common(1)[0][0]
    return most_common_class

training_data = [([150, 50], 'medium'), ([155, 55], 'medium'), ([160, 60], 'large'), ([161, 59], 'large'), ([158, 65], 'large')]
test_instance=[157,54]
k_value = 1 # Define k_value before using it
result = Knn_classifier(training_data, test_instance,k_value)
print("Predicted class:", result)

# ## one hot encoding using dataset
import arff  # to parse the dataset in the form of the arff file

arff_file_path = "C:\\Backup\\amrita\\YEAR 2\\SEM-4\\Machine Learning\\Lab Sessions\\Lab Session2\\dataset1.arff"
with open(arff_file_path, 'r') as f:
    arff_data = arff.load(f)

data = arff_data['data']
attributes = arff_data['attributes']

encoded_data, new_features = lab2.one_hot_encode_categorical(data)
print("Encoded data:", encoded_data)
print("New features:", new_features)