# %% import dataframe from pickle file
from os import access
import pandas as pd

df = pd.read_pickle("UK.pkl")
df.head()

# %% convert dataframe to invoice-based transactional format
dataset=[]
for inv_num, gdf in df.groupby("InvoiceNo"):
    dataset.append(gdf["Description"].to_list()) 

# %% Alternate way
# dataset = df.groupby("InvoiceNo").apply(lambda gdf: gdf["Description"].to_list())

# %% apply apriori algorithm to find frequent items and association rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support= 0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, min_threshold=0.1)

# %% count of frequent itemsets that have more then 1/2/3 items,
# and the frequent itemsets that has the most items
length = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets["length"] = length
print((frequent_itemsets["length"] > 1).sum())
print((frequent_itemsets["length"] > 2).sum())
print((frequent_itemsets["length"] > 3).sum())

# %% top 10 lift association rules

rules.sort_values("lift",ascending=False).head(10)

# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")

# %% scatterplot support vs lift
sns.scatterplot(x= rules["support"], y = rules["lift"])
plt.xlabel("support")
plt.ylabel("lift")
plt.title("support vs lift")
# %%
