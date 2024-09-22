import pandas as pd

#Read Excel and delete all columns with 'VST'

data = pd.read_excel('Macau_weather_dataset.xlsx')

print(data['Total rainfall (mm)'].eq('VST').sum())  
    
data_cleaned = data[~data['Total rainfall (mm)'].eq('VST')]  

#Save the new dataset

data_cleaned.to_excel('macau_weather_ds_novst.xlsx', index=False)

