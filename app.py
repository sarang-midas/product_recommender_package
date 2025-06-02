import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

@st.cache_data
def load_data():
    df = pd.read_csv("transactions_large.csv")
    transactions = df['Items'].apply(lambda x: [item.strip().lower() for item in x.split(',')])
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    return rules

def antecedent_contains_item(antecedent, item):
    return item in [i.lower() for i in antecedent]

def get_recommendations(rules, item):
    recommendations = rules[rules['antecedents'].apply(lambda x: antecedent_contains_item(x, item))]
    return recommendations

def main():
    st.title("Product Association & Recommendation Analytics")
    st.write("Enter a product to get product recommendations based on association rules.")

    rules = load_data()
    product = st.text_input("Enter a product (e.g., milk):").strip().lower()

    if product:
        recommendations = get_recommendations(rules, product)
        if not recommendations.empty:
            st.write(f"### Products frequently bought with '{product}':")
            for _, row in recommendations.iterrows():
                consequents = ', '.join(list(row['consequents']))
                st.write(f"→ **{consequents}** (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.2f})")
        else:
            st.write(f"No strong recommendations found for '{product}'.")

if __name__ == "__main__":
    main()
