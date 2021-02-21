import pandas as pd

data = pd.read_csv('api.csv', header=2)
data = data.rename(data.loc[data.index]['Country Name'])
data = data.drop(data.columns[:4], 1)
data.dropna(axis='columns', how='all', inplace=True)
data.dropna(axis='rows', how='all', inplace=True)
print(data)
#Первый график
data.sum(axis='rows', numeric_only=True).plot(title='ВВП стран с 1960 по 2019').get_figure().show()
#Второй график
data.loc[data.sum(axis='columns').sort_values(ascending=False)[:5].index].transpose().plot(title='ВВВ топ 5 стран с 1960 по 2019').get_figure().show()
