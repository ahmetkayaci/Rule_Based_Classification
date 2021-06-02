import numpy as np
import pandas as pd

def load_persona():
    df = pd.read_csv("datasets/persona.csv")
    return df

df = load_persona()
pd.pandas.set_option('display.max_columns',None)

### Dataset Observation ###

def check_df(dataframe,head=5, tail=5):
    print("\n\n##### S H A P E #####\n")
    print(dataframe.shape)
    print("\n\n##### H E A D #####\n")
    print(dataframe.head(head))
    print("\n\n##### T A I L #####\n")
    print(dataframe.tail(tail))
    print("\n\n##### I N F O #####\n")
    print(dataframe.info())
    print("\n\n##### D E S C R I B E #####\n")
    print(dataframe.describe().T)
    print("\n\n##### N A #####\n")
    print(dataframe.isnull().sum())
    print("\n\n##### Q U A N T I L E S #####\n")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

### Unique SOURCE number and frequency information ###
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

### Unique PRICE number ###
df["AGE"].nunique()
df["PRICE"].nunique()

### number of sales by PRICE ###
df["PRICE"].value_counts()

### number of sales by COUNTRIES ###
df["COUNTRY"].value_counts()

### total gain obtained from sales by COUNTRIES ###
df.groupby(["COUNTRY"]).agg({"PRICE":"sum"}).sort_values(by="PRICE", ascending=False)

### number of sales by SOURCE kinds ###
df["SOURCE"].value_counts()
df.groupby(["SOURCE"]).agg({"PRICE" : "count"})

## price mean by COUNTRIES
df.groupby(["COUNTRY"]).agg({"PRICE" : "mean"}).sort_values(by="PRICE", ascending=False)

### price mean by SOURCES ###
df.groupby(["SOURCE"]).agg({"PRICE" : "mean"})

### price mean by COUNTRIES and SOURCES ###
df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE" : ["mean","count"]})


### What are the total gains broken down by COUNTRY, SOURCE, SEX, AGE? ###

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE" : "sum"})


### sort the output by PRICE ###

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE" : "sum"}).sort_values(by="PRICE", ascending=False)
agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE" : "sum"}).sort_values(by="PRICE", ascending=False)
agg_df


### convert the index name to variable name ###

agg_df.reset_index(inplace=True)
agg_df.head()
df.head()
agg_df["AGE"].value_counts()


### convert age variable to categorical variable and add it to agg_df ###

labels_List = ["0_19","20_24","24_31","31_41","41_70"]
bins = [0,19,24,31,41,70]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=bins, labels = labels_List)

agg_df[agg_df["AGE"] == 24]
agg_df[agg_df["AGE"] == 19]
agg_df[agg_df["AGE"] == 31]
agg_df[agg_df["AGE"] == 30]
agg_df[agg_df["AGE"] == 20]


### define new level-based customers (persona) ###

agg_df["customers_level_based"] = [col[0].upper() + "_" +  col[1].upper() + "_" + col[2].upper() + "_" + col[5].upper() for col in agg_df.values]
agg_df = agg_df.groupby(["customers_level_based"]).agg({"PRICE" : "mean"})

agg_df.head()


### Segment your new customers (persona) ###

pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby(["SEGMENT"]).agg({"PRICE" : ["max","mean","sum"]})

#1.key
agg_df[agg_df["SEGMENT"] == "C"]
#2.key
agg_df[agg_df["SEGMENT"].apply(lambda x:x == "C")]

#1.key
agg_df[agg_df["SEGMENT"].apply(lambda x:x == "C")].describe().T
#2.key
check_df(agg_df[agg_df["SEGMENT"] == "C"])


### Classify new customers by segments and predict how much will achieve gain ###

agg_df = agg_df.reset_index()
new_user = "TUR_ANDROID_FEMALE_31_41"
agg_df[agg_df["customers_level_based"] == new_user]

new_user2 = "FRA_IOS_FEMALE_31_41"
agg_df[agg_df["customers_level_based"] == new_user2]

