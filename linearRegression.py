import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("Book.csv")
plt.scatter(data.videos,data.views,color="red")
plt.xlabel("Number of videos")
plt.ylabel("Total Views")

x= np.array(data.videos.values)
y=np.array(data.views.values)

model=LinearRegression()
model.fit(x.reshape(-1,1),y)

required_val=int(input("Enter the views of your channel"))

new_val = np.array([required_val]).reshape(-1,1)
predit=model.predict(new_val)

print("The Total predicted views of your chanel is ",predit)