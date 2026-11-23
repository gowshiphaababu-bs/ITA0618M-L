import pandas as pd

# Create CSV dataset
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
df.to_csv('training_data.csv', index=False)

# Read CSV file
df = pd.read_csv('training_data.csv')

# Candidate Elimination
def candidate_elimination(data):
    S = ['0'] * (len(data.columns) - 1)
    G = [['?' for _ in range(len(S))]]

    for _, row in data.iterrows():
        example = list(row.iloc[:-1])
        target = row.iloc[-1]

        if target == 'Yes':

            G = [g for g in G if all(
                g[i] == '?' or g[i] == example[i]
                for i in range(len(S))
            )]

            for i in range(len(S)):
                if S[i] == '0':
                    S[i] = example[i]
                elif S[i] != example[i]:
                    S[i] = '?'

        else:
            new_G = []

            for g in G:
                if all(g[i] == '?' or g[i] == example[i]
                       for i in range(len(S))):

                    for i in range(len(S)):
                        if g[i] == '?':
                            if S[i] != '?' and S[i] != '0':
                                new_h = g.copy()
                                new_h[i] = S[i]
                                new_G.append(new_h)
                else:
                    new_G.append(g)

            G = new_G

    return S, G

S, G = candidate_elimination(df)

print("Specific Boundary (S):")
print(S)

print("\nGeneral Boundary (G):")
for g in G:
    print(g)
