# %% Import excel to dataframe
import pandas as pd

df = pd.read_excel("Online Retail.xlsx")

# %%  Show the first 10 rows
df.head(10)


# %% Generate descriptive statistics regardless the datatypes
df.describe(include="all")

# %%
df.isna().sum()

# %% Remove all the rows with null value and generate stats again
df = df.dropna()
df.describe(include="all")

# if you wanna make changes to the same df
# or this way - df.dropna(inplace=True)


# %% Remove rows with invalid Quantity (Quantity being less than 0)
df = df[df["Quantity"] > 0]


# %% Remove rows with invalid UnitPrice (UnitPrice being less than 0)
df = df[df["UnitPrice"] > 0]


# %% Only Retain rows with 5-digit StockCode
has_5_chars = df["StockCode"].astype(str).apply(len)==5
is_numeric = df["StockCode"].astype(str).str.isnumeric()

df = df[has_5_chars & is_numeric]


# %% strip all description
df["Description"] = df["Description"].str.strip()

# %% Generate stats again and check the number of rows
# df.describe()

df.describe(include="all")

# %% Plot top 5 selling countries
import matplotlib.pyplot as plt
import seaborn as sns

top5_selling_countries = df["Country"].value_counts()[:5]

sns.barplot(
    x=top5_selling_countries.index, 
    y=top5_selling_countries.values
)
plt.xlabel("Country")
plt.ylabel("Amount")
plt.title("Top 5 Selling Countries")


# %% Plot top 20 selling products, drawing the bars vertically to save room for product description
top_products = df.groupby("Description").sum().reset_index()
top_products = top_products.sort_values("Quantity",ascending=False)

# %%
sns.barplot(
    data = top_products.head(20),
    y = "Description",
    x = "Quantity"
)


# %% Focus on sales in UK
df = df[df["Country"] == "United Kingdom"]
df

#%% Show gross revenue by year-month
from datetime import datetime

df["YearMonth"] = df["InvoiceDate"].apply(
    lambda dt: datetime(year=dt.year, month=dt.month, day=1)
)

#%%
df["GrossRevenue"] = df["Quantity"] * df["UnitPrice"]


# %%
sns.lineplot(
    data = df.groupby("YearMonth").sum().reset_index(),
x = "YearMonth",
y = "GrossRevenue"
)

# %% save df in pickle format with name "UK.pkl" for next lab activity
# we are only interested in InvoiceNo, StockCode, Description columns
df[["InvoiceNo", "StockCode" , "Description"]].to_pickle("UK.pkl")


# %%
