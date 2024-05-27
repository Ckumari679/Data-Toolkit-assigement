#!/usr/bin/env python
# coding: utf-8

# In[1]:


#QUESTION 1


# In[2]:


#Demonstrate three different methods for creating identical 2D arrays in NumPy Provide the code for each
#method and the final output after each method


# In[3]:


import numpy as np


# In[4]:


arr1=np.array([[3,3,3],[3,3,3]])
arr1


# In[5]:


arr=np.array([3,3,3,3,3,3])
arr2=arr.reshape((2,3))
arr2


# In[6]:


ar = np.ones((2,3),dtype=int)
arr3 = ar+2
arr3


# In[7]:


#QUESTION 2


# In[8]:


#Using the Numpy function, generate an array of 100 evenly spaced numPers Petween 1 and 10 and
#Reshape that 1D array into a 2D array


# In[9]:


array_1=np.linspace(1,10,(100))
array_1


# In[10]:


array_2=array_1.reshape((10,10))
array_2


# In[11]:


array_2=array_1.reshape((10,10))
array_2


# In[12]:


#QUESTION 3


# In[13]:


#1.  The difference in np.array, np.asarray and np.asanyarray


# In[14]:


'''np.array is use for create a new array'''

np.array([1,2,3])


# In[15]:


'''np.asarray is use to convert any input to an numpy array'''

l=[1,2,3,4]
np.asarray(l)


# In[16]:


'''np.asany array is also used to create a numpy nd array but also create a sub classes'''

a=[2,3,4]
np.asanyarray(a)


# In[17]:


#2 The difference between Deep copy and shallow copy


# In[18]:


#shallow copy
    
#taking a list and convert intu array
l=[1,2,3,4,5]
arr=np.array(l)
arr


# In[19]:


#taking a another varible 'x' and putting 'arr'into it
x=arr
x


# In[20]:


#now i change any index value of arr
arr[1]=200
x


# In[21]:


#i am changing the value of 'arr' but 'x' was automatic change this is called shallow copy


# In[ ]:





# In[22]:


#deep copy

#we take an another varible'y' and insert a copy of arr
y=arr.copy()
y


# In[23]:


#now i am changing any index value of arr
arr[1]=500
arr


# In[24]:


y


# In[25]:


#hear we can see that arr has chang index value but y remains as it is ,this is called deep copy


# In[ ]:





# In[ ]:









# In[26]:


#QUESTION 4


# In[27]:


# Generate a 3x3 array with random floating-point numBers Between 5 and 20 Then, round each numBer in
#the array to 2 decimal places


# In[28]:


array_random=np.random.uniform(5,21,(3,3))
array_random


# In[29]:


array_random=np.round(array_random,2)
array_random


# In[30]:


#question 5


# In[31]:


# Create a NumPy array with random integers Petween 1 and 10 of shape (5, 6)) After creating the array
#perform the following operations:


# In[32]:


import numpy as np

random_array=np.random.randint(1,11,(5,6))
random_array


# In[33]:


# all even integer

even= random_array[random_array%2==0]
even


# In[34]:


#all odd integer

odd=random_array[random_array % 2!=0]
odd


# In[ ]:










# In[35]:


#QUESTION 6


# In[36]:


#Create a 3D NumPy array of shape (3, 3,3) containing random integers Petween 1 and 10 Perform the following operations


# In[37]:


array1=np.random.randint(1,11,(3,3,3))


# In[38]:


array1


# In[39]:


# 1  maximum indices 

max_value=np.argmax(array1,axis=2)
max_value


# In[40]:


# 2  Perform element wise multiplication of between both array


# In[41]:


multiplication=np.multiply(array1,max_value)

#multiplication=array1*max_value

multiplication


# In[42]:


#question 7


# In[43]:


#Clean and transform the 'Phone' column in the sample dataset to remove non-numeric characters and
#convert it to a numeric data type Also display the taPle attriPutes and data types of each column


# In[44]:


#QUESTION 8


# In[45]:


#Perform the following tasK using people dataset:


# In[ ]:





# In[48]:


#a) Read the 'dataYcsv' file using pandas, skipping the first 50 rows.
import pandas as pd
df=pd.read_csv('People Data.csv')
df


# In[49]:


#for skipping 50 rows
df_skip50=pd.read_csv('People Data.csv',skiprows=50)
df_skip50


# In[50]:


#b) Only read the columns: 'Last Name', ‘Gender’,’Email’,‘Phone’ and ‘Salary’ from the file
df=pd.read_csv('People Data.csv',usecols=['Last Name','Gender','Email','Phone','Salary'])
df


# In[51]:


#c) Display the first 10 rows of the filtered dataset.

df=pd.read_csv('People Data.csv',nrows=10)
df


# In[52]:


#d) Extract the ‘Salary’' column as a Series and display its last 5 valuesX

df=pd.read_csv('People Data.csv' ,usecols=['Salary']  )
df.tail(5)


# In[53]:


#question 9


# In[54]:


# Filter and select rows from the People_Dataset, where the “Last Name' column contains the name 'DuKe', 
#'Gender' column contains the word Female and ‘Salary’ should Pe less than 85000


# In[55]:


import pandas as pd
#READ DATA SET
df=pd.read_csv('People Data.csv')

row=df[(df['Last Name'].str.contains('Duke'))   &(df['Gender']==('Female'))   &(df['Salary']<85000)]

row
                                 


# In[56]:


#QUESTION 10


# In[57]:


#Create a 7*5 Dataframe in Pandas using a series generated from 35 random integers Petween 1 to 6)?


# In[58]:


import numpy as np
import pandas as pd

#generate a random array between 1 to 6
arr=np.random.randint(1,7, 35)

series=pd.Series(arr)

data_frame = pd.DataFrame(series.values.reshape(7,5))

data_frame


# In[59]:


#QUESTION 11


# In[60]:


#Create two different Series, each of length 50, with the following criteria:


#a) The first Series should contain random numbers ranging from 10 to 50

import numpy as np
import pandas as pd

series1=pd.Series(np.random.randint(10,51, size=50) ,name='col1')

series1                 
                  


# In[61]:


#b) The second Series should contain random numbers ranging from 100 to 1000.

import numpy as nnp
import pandas as pd

arr=np.random.randint(100,1001,size=50)
series2=pd.Series(arr,name='col2')
series2
                 


# In[62]:


data_frame=pd.concat([series1,series2] ,axis=1)
data_frame


# In[63]:


#QUESTION 12


# In[64]:


#K%g Perform the following operations using people data set:

#a) Delete the 'Email', 'Phone', and 'Date of birth' columns from the dataset.

import pandas as pd

data=pd.read_csv('People Data.csv')

data.drop(['Email','Phone','Date of birth'],axis=1,inplace=True)


#b) Delete the rows containing any missing values.

data.dropna(inplace=True)


# In[65]:


#c) Print the final output also
data


# In[66]:


#QUESTION 13  


# In[67]:


#g Create two NumPy arrays, x and y, each containing 100 random float values between 0 and 1. Perform the following tasks using Matplotlib and NumPy:


# In[68]:


import numpy as np

#generate random value for x
x=np.random.rand(100)
x


# In[69]:


#generate random number for y

y=np.random.rand(100)
y


# In[70]:


#Create a scatter lot using x and y, setting the color of the oints to red and the marker style to 'o'.
import matplotlib.pyplot as plt
plt.scatter(x,y ,color = 'red',marker='o',label='random points')


# In[71]:


#b) Add a horizontal line at y = 0.5 using a dashed line style and label it as 'y = 0.5'.

plt.axhline(y=0.5,color='red',linestyle='--' ,label='y=0.5')


# In[72]:


# c) Add a vertical line at x = 0.5 using a dotted line style and label it as 'x = 0.5'.
plt.axvline(x=0.5,linestyle=':',color='blue',label='x=0.5')


# In[73]:


#d) Label the x-axis as 'X-axis' and the y-axis as 'Y-axis'.
plt.xlabel('x-axis')
plt.ylabel('y-axis')


# In[74]:


#e) Set the title of the lot as 'Advanced Scatter Plot of Random Values'.
plt.title('Advanced Scatter Plot of Random Values')


# In[75]:


#f) Dislay a legend for the scatter lot, the horizontal line, and the vertical line.
plt.legend()
plt.show()


# In[ ]:









# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a time-series dataset
dates = pd.date_range('2024-01-01', periods=365)
temperature = np.random.normal(25, 5, 365)
humidity = np.random.normal(50, 10, 365)

data = {'Date': dates, 'Temperature': temperature, 'Humidity': humidity}
df = pd.DataFrame(data)



# In[79]:


#a. Plot Temperature and Humidity on the same plot with different y-axes
#b) Label the x-axis as 'Date'.
#c) Set the title of the lot as 'Temerature and Humidity Over Time'.

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (°C)', color=color)
ax1.plot(df['Date'], df['Temperature'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Humidity (%)', color=color)
ax2.plot(df['Date'], df['Humidity'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Temperature and Humidity Over Time')
plt.show()


# In[ ]:





# In[ ]:








# In[80]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Create a NumPy array with 1000 samples from a normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)
data


# In[81]:


# Plot a histogram of the data with 30 bins
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')



# In[82]:


# Overlay a line plot representing the normal distribution's probability density function (PDF)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, 0, 1)  # PDF of standard normal distribution with mean=0 and std=1
plt.plot(x, p, 'k', linewidth=2, label='PDF')


# In[83]:


# Label the axes
plt.xlabel('Value')
plt.ylabel('Frequency/Probability')



# In[84]:


# Set the title of the plot
plt.title('Histogram with PDF Overlay')



# In[85]:


plt.title('Histogram with PDF Overlay')


# In[86]:


#QUESTION17


# In[87]:


# Create a Seaborn scatter plot of two random arrays, color points based on their position relative to the
#origin (quadrants), add a legend, label the axes, and set the title as 'Quadrant-wise Scatter Plot'.


# In[88]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate two random arrays
x = np.random.randn(100)
y = np.random.randn(100)



# In[89]:


# Determine quadrant for each point
quadrant = np.zeros(100)
quadrant[(x > 0) & (y > 0)] = 1 
quadrant[(x < 0) & (y > 0)] = 2
quadrant[(x < 0) & (y < 0)] = 3
quadrant[(x > 0) & (y < 0)] = 4  



# In[90]:


# Create a DataFrame for Seaborn plotting
data = {'x': x, 'y': y, 'Quadrant': quadrant}
df = pd.DataFrame(data)

# Define color palette for each quadrant
palette = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red'}


# In[91]:


# Create the scatter plot
sns.scatterplot(data=df, x='x', y='y', hue='Quadrant', palette=palette)



# In[ ]:














# In[92]:


# Create the scatter plot
sns.scatterplot(data=df, x='x', y='y', hue='Quadrant', palette=palette)

# Label the axes and set the title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Quadrant-wise Scatter Plot')

# Show legend
plt.legend(title='Quadrant')


# In[ ]:




