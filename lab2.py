import math
import arff  # to parse the dataset in the form of the arff file

# testing of the parsing of the data set using the arff library
# arff_file_path = "C:\\Backup\\amrita\\YEAR 2\\SEM-4\\Machine Learning\\Lab Sessions\\Lab Session2\\dataset1.arff"
# with open(arff_file_path, 'r') as f:
#     arff_data = arff.load(f)
#
# data = arff_data['data']
# attributes = arff_data['attributes']
# print("Attributes:")
# for attr in attributes:
#     print(attr)
#
# print("\nData:")
# for row in data:
#     print(row)


def find_euclidian_dist():
    dim1 = int(input("enter the dimensions of first vector"))
    dim2 = int(input("enter the dimensions of second vector"))

    if dim1 != dim2:
        return "vector dimensions have to be the same for finding distance"

    vector1 = []
    vector2 = []
    ans = 0

    for i in range(dim1):
        item = int(input(f'enter the {i+1} th value of vector 1'))
        vector1.append(item)

    for i in range(dim1):
        item = int(input(f'enter the {i+1} th value of vector 1'))
        vector2.append(item)

    else:
        for i in range(dim1):
            ans = ans + (pow(vector1[i]-vector2[i], 2))

        euclid_dist = math.sqrt(ans)
        return euclid_dist


def find_manhattan_dist():
    dim1 = int(input("enter the dimensions of first vector"))
    dim2 = int(input("enter the dimensions of second vector"))

    if dim1 != dim2:
        return "vector dimensions have to be the same for finding distance"

    vector1 = []
    vector2 = []
    ans = 0

    for i in range(dim1):
        item = int(input(f'enter the {i+1} th value of vector 1'))
        vector1.append(item)

    for i in range(dim1):
        item = int(input(f'enter the {i+1} th value of vector 1'))
        vector2.append(item)

    else:
        for i in range(dim1):
            ans = ans + (abs(vector1[i]-vector2[i]))

        euclid_dist = math.sqrt(ans)
        return euclid_dist


def label_encode_categorical(data):
    # Create an empty dictionary to store mappings
    label_mapping = {}
    # An empty list for encoded data
    encoded_data = []

    # Iterate through each column in the dataset (should be categorical)
    for col_index in range(len(data[0])):
        # maintains a count for the number of labels
        label_counter = 0
        # Create a dictionary to store label mappings for the current column
        col_label_mapping = {}

        # Iterate through each row in the dataset
        for row_index in range(len(data)):
            # Get the value of the current cell
            cell_value = data[row_index][col_index]

            # If the cell value is not already in the label mapping dictionary,assign it a new label
            if cell_value not in col_label_mapping:
                col_label_mapping[cell_value] = label_counter
                label_counter += 1

            # Replace the cell value with its corresponding label
            data[row_index][col_index] = col_label_mapping[cell_value]

        # Add the label mappings for the current column to the overall label mapping dictionary
        label_mapping[col_index] = col_label_mapping

    return data, label_mapping


def one_hot_encode_categorical(data):

    # Create an empty list to store the names of the new features and one to store encoded data
    new_features = []
    encoded_data = []

    # Iterate through each column (assumed to be categorical) in the dataset
    for col_index in range(len(data[0])):
        # Create a dictionary to store unique categories in the current column
        categories = {}

        # Iterate through each row in the dataset
        for row_index in range(len(data)):
            # Get the value of the current cell
            cell_value = data[row_index][col_index]

            # If the cell value is not already in the categories dictionary, add it
            if cell_value not in categories:
                categories[cell_value] = len(categories)

        # Create new binary features for each unique category in the current column
        for category in categories:
            # Create a new feature name by appending the category name to the original feature name
            new_feature_name = f"{col_index}_{category}"
            new_features.append(new_feature_name)

            # Create a new binary feature vector for the current category
            new_feature = [1 if data[row_index][col_index] == category else 0 for row_index in range(len(data))]
            encoded_data.append(new_feature)

    # Transpose the encoded data to convert rows to columns
    encoded_data = list(map(list, zip(*encoded_data)))

    return encoded_data, new_features




