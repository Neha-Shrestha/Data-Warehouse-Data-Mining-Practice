import numpy as np
import pandas as pd

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
        self.value = None

def entropy(class_labels):
    _, counts = np.unique(class_labels, return_counts=True)
    probabilities = counts / len(class_labels)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data, attribute, class_attribute):
    total_entropy = entropy(data[class_attribute])
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[attribute] == values[i]).dropna()[class_attribute]) for i in range(len(values))])
    return total_entropy - weighted_entropy

def id3(data, class_attribute, attributes):
    if len(np.unique(data[class_attribute])) == 1:
        node = Node(None)
        node.value = np.unique(data[class_attribute])[0]
        return node
    elif len(attributes) == 0:
        node = Node(None)
        node.value = np.unique(data[class_attribute])[np.argmax(np.unique(data[class_attribute], return_counts=True)[1])]
        return node
    else:
        best_attribute = max(attributes, key=lambda attr: information_gain(data, attr, class_attribute))
        node = Node(best_attribute)
        for value in np.unique(data[best_attribute]):
            sub_data = data.where(data[best_attribute] == value).dropna()
            if len(sub_data) == 0:
                node.children[value] = np.unique(data[class_attribute])[np.argmax(np.unique(data[class_attribute], return_counts=True)[1])]
            else:
                node.children[value] = id3(sub_data, class_attribute, [attr for attr in attributes if attr != best_attribute])
        return node

def predict(node, sample):
    if node.value is not None:
        return node.value
    else:
        try:
            prediction = node.children[sample[node.attribute]]
            return predict(prediction, sample)
        except KeyError:
            # If the value of the attribute is not present in the training data,
            # return the most common class label of the current node's children
            children_values = [child.value for child in node.children.values()]
            return max(set(children_values), key=children_values.count)

# Example dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Specify the target variable name
class_attribute = 'PlayTennis'

# Specify the list of attribute names
attributes = ['Outlook', 'Temperature', 'Humidity', 'Windy']

# Train the decision tree
root = id3(data, class_attribute, attributes)

# Example prediction:
sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Windy': 'True'}
print("Prediction:", predict(root, sample))
