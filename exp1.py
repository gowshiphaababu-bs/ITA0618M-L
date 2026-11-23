import pandas as pd

# Training dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# FIND-S Algorithm
def find_s(data):
    hypothesis = ['0'] * (len(data.columns) - 1)

    for index, row in data.iterrows():
        if row.iloc[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = row.iloc[i]
                elif hypothesis[i] != row.iloc[i]:
                    hypothesis[i] = '?'

    return hypothesis

print("Training Data:")
print(df)

print("\nFinal Most Specific Hypothesis:")
print(find_s(df))
