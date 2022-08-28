import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('Housing.csv')

column = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
label = preprocessing.LabelEncoder()

mapping = { }
for col in column:
  df[col] = label.fit_transform(df[col])
  le_name_mapping = dict(zip(label.classes_,label.transform(label.classes_)))
  mapping[col] = le_name_mapping

# X = df[['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']]
# y = df['price']

X = df.iloc[:,1:13].values
y = df.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=73)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(r2_score(y_test,y_pred))

result = model.predict([[8100,4,1,2,1,1,1,0,1,2,1,0]])
print(result)

with open('model.pkl','wb') as files:
    pickle.dump(model,files)

