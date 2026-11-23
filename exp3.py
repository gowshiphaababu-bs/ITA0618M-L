import pandas as pd
import math
from collections import Counter

# Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],

    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],

    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],

    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],

    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes',
             'No', 'Yes', 'No', 'Yes', 'Yes',
             'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Entropy
def entropy(target):
    counts = Counter(target)
    total = len(target)

    return -sum((count / total) * math.log2(count / total)
                for count in counts.values())

# Information Gain
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])

    weighted_entropy = 0

    for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(
            subset[target]
        )

    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(df, target, attributes):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    if not attributes:
        return df[target].mode()[0]

    best_attribute = max(
        attributes,
        key=lambda attr: information_gain(df, attr, target)
    )

    tree = {best_attribute: {}}

    for value in df[best_attribute].unique():
        subset = df[df[best_attribute] == value]

        remaining = [a for a in attributes if a != best_attribute]

        tree[best_attribute][value] = id3(
            subset.drop(columns=[best_attribute]),
            target,
            remaining
        )

    return tree

# Build Decision Tree
attributes = list(df.columns[:-1])

tree = id3(df, 'Play', attributes)

print("Decision Tree:")
print(tree)

# Classification
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    value = sample[attribute]

    if value not in tree[attribute]:
        return None

    return predict(tree[attribute][value], sample)

new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

print("\nNew Sample:", new_sample)
print("Predicted Class:", predict(tree, new_sample))
