import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns 

#Read Excel

data = pd.read_excel('macau_weather_ds_novst.xlsx')

#Classify all the data on a monthly basis

data.set_index('Date', inplace=True)  #set column 'Date' as index

monthly_precipitation = data.groupby(pd.Grouper(freq='M'))['Rainfall(mm)'].mean() 

#Draw graphs 

#Set style
sns.set(style="whitegrid")  
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))  

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']    
for i, month in enumerate(months):

    month_data = monthly_precipitation[monthly_precipitation.index.month == i + 1]  
      
    row, col = divmod(i, 4)  
    ax = axes[row, col]  
        
    ax.plot(month_data.index.year, month_data.values, marker='o', linestyle='-', color='b')  
      
    ax.set_title(f'{month} Precipitation in Macau')  
    ax.set_xlabel('Year')  
    ax.set_ylabel('Precipitation (mm)')  
      
    # Delete useless grid lines 
    ax.grid(False)  
  
plt.tight_layout()  
  
plt.show()