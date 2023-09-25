import pandas as pd 


df = pd.read_csv('all_labeled_data.csv', header=None)
ds= df.sample(frac=1)
ds.to_csv('newfile.csv')