import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Reads the table from the txt file.
fields = ['mpg', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin',
          'Car Name']
auto_mpg_table = pd.read_csv('auto-mpg.data', delim_whitespace=True, names=fields)

df = auto_mpg_table.drop(columns=['Car Name'])
df = df.replace('?', None)
df = df.dropna()

# Splits data 80:20 training:testing
# Random State is 200 for duplication purposes.
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

#print(f"Train Size: {len(train)}")

# Displays the Abs of the correlation between mpg and other columns in order to find x-axis
print(f'Correlation mpg/Acceleration = {abs(df["mpg"].corr(df["Acceleration"]))}')
print(f'Correlation mpg/Weight = {abs(df["mpg"].corr(df["Weight"]))}')

# We will be using Weight as our X-Axis beccause its correlation is closer to 1

y_train_list = []
x_train_list = []
for index,row in train.iterrows():
    x_train_list.append(float(row['Weight']))
    y_train_list.append(float(row['mpg']))

x_test_list = []
y_test_list = []
for index,row in test.iterrows():
    x_test_list.append(float(row['Weight']))
    y_test_list.append(float(row['mpg']))
model = linear_model.LinearRegression()

training_x = np.c_[x_train_list]
training_y = np.c_[y_train_list]
testing_x = np.c_[x_test_list]
testing_y = np.c_[y_test_list]
model.fit(training_x,training_y)

y_pred = model.predict(testing_x)


plt.xlabel("Weight")
plt.ylabel("mpg")

# Graphs the training data.
#plt.scatter(training_x, training_y, color = 'b')

# Graphs the testing data
plt.scatter(testing_x, testing_y, color = 'b')

plt.plot(testing_x, y_pred, color='r')


train_square_error = mean_squared_error(testing_y, y_pred)
print(f'Mean Squared Error: {train_square_error}')

tree_reg = DecisionTreeRegressor()
tree_reg.fit(training_x,training_y)

y_tree_pred = tree_reg.predict(training_x)
plt.plot(training_x, y_tree_pred, color='g')

tree_mse = mean_squared_error(training_y, y_tree_pred)
tree_rmse = np.sqrt(tree_mse)
print(f'Tree MSE is: {tree_mse}')
print(f'Tree RMSE is: {tree_rmse}')

scores = cross_val_score(tree_reg, training_x, training_y,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#print(f'Scores: {tree_rmse_scores.mean()}')

scores = cross_val_score(model, training_x, training_y,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)
#print(f'Scores: {lin_rmse_scores.mean()}')

plt.show()
