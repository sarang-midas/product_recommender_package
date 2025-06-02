import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load CSV file
df = pd.read_csv("transactions_large.csv")

# Normalize and split items to lowercase
transactions = df['Items'].apply(lambda x: [item.strip().lower() for item in x.split(',')])

# One-hot encode
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori with lower support
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Association rules with lower confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Debug print all antecedents
print("All antecedents in rules:")
for ant in rules['antecedents']:
    print(ant)

# User input normalized to lowercase
item = input("\nEnter a product to get recommendations (e.g., Milk): ").strip().lower()

# Filter rules ignoring case
def antecedent_contains_item(antecedent, item):
    return item in [i.lower() for i in antecedent]

recommendations = rules[rules['antecedents'].apply(lambda x: antecedent_contains_item(x, item))]

if not recommendations.empty:
    print(f"\nProducts frequently bought with '{item}':")
    for _, row in recommendations.iterrows():
        consequents = ', '.join(list(row['consequents']))
        print(f"→ {consequents} (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.2f})")
else:
    print(f"\nNo strong recommendations found for '{item}'.")
