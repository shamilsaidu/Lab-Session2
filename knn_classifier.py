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
