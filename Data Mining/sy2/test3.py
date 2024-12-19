import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Define the dataset
data = [
    ['f', 'a', 'c', 'd', 'g', 'i', 'm', 'p'],
    ['a', 'b', 'c', 'f', 'l', 'm', 'o'],
    ['b', 'f', 'h', 'j', 'o'],
    ['b', 'c', 'k', 's', 'p'],
    ['a', 'f', 'c', 'e', 'l', 'p', 'm', 'n']
]

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)
df = df.stack().reset_index()
df.columns = ['Transaction', 'Item', 'Value']
df = df.pivot(index='Transaction', columns='Value', values='Item').notnull().astype(int)

# Apply the Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Generate the association rules
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
# # 生成关联规则
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7, num_itemsets=len(frequent_itemsets))
# Print the results
print(frequent_itemsets)
print(rules)
