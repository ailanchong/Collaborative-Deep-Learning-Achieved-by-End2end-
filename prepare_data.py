# -*- coding:utf-8 -*- 
import numpy as np  
import pandas as pd 


#order_products_prior_df = pd.read_csv("./order_products__prior.csv")
orders_df = pd.read_csv("./orders.csv")
print(order_products__prior_df.head())

user_products = order_products_prior_df.merge(orders_df, how='left', on='order_id')

counts = user_products.groupby(['user_id','product_id']).size().rename('count').reset_index()

if (not os.path.isdir('data')):
    os.system("mkdir data")
print(counts.head())
np.save("data/i.npy", counts['user_id'].values)
np.save("data/j.npy", counts['product_id'].values)
np.save("data/V_ij.npy", counts['count'].values)
 