### ONLY PLOTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from unidecode import unidecode


### Load files
dfcity = pd.read_csv('/home/joao/Dropbox/Aula/Pesquisa/michel/city_date_merged.csv')

exclude_cols = ['Ct_city', 'Ct_patient_mean', 'Ct_target_patient_only','has_city_ct', 'has_patient_ct',
         'State_norm', 'City', 'State', 'patient_city_name_mode', 'patient_state_name_mode', 
         'target_source', 'city_Latitude', 'city_Longitude', 'state_Longitude', 'state_Latitude', 'city_TotalCases', 
         'city_TotalDeaths', 'state_TotalCases', 'state_TotalDeaths', 'city_Ct_Value', 'city_Patience_Count', 
         'ct_p10', 'ct_p25', 'ct_p75', 'ct_p90', 'ct_mean', 'age_mean', 'age_p25', 'age_p75']

dfcity = dfcity.drop(columns=exclude_cols)

# Exogenous variables

    # age_std, age_p50, male_ratio, positivity_rate, city_NewCases, city_NewDeaths, city_Hosp_Count, city_Hosp_Deaths, city_Stringency_Index, state_Stringency_Index, city_Vax_AllDoses, city_Vax_Dose1/2/3 (we can probably consider them as strict exo variables , i.e. as future covariates)

### Describe data structure
# dfcity.head()
# dfcity.shape
# dfcity.info()
# dfcity.dtypes

### Dates as a datetime
dfcity['Date'] = pd.to_datetime(dfcity['Date'])

### First NORMALIZE the cities names
dfcity['City_name'] = dfcity['City_norm'].apply(unidecode)

### How many cities are there?
dfcity['City_name'].nunique()

### Change target variable name to ct_target
dfcity = dfcity.rename(columns={'Ct_target_pref_city': 'ct_target'})

### Number of missing data in ct_target
missing_ct_target = dfcity['ct_target'].isnull().sum()
print(f'Missing data points in ct_target: {missing_ct_target}')

### Check for duplicates with same date and city name
duplicates = dfcity[dfcity.duplicated(subset=['City_name', 'Date'], keep=False)]

### Remove duplicates only when target_source = patient
dfcity = dfcity.drop_duplicates(subset=['City_name', 'Date'], keep='last')

### Insert missing dates, so all the cities are using the same period
all_dates = pd.date_range(start=dfcity['Date'].min(), end=dfcity['Date'].max(), freq='D')
all_cities = dfcity['City_name'].unique()
new_index = pd.MultiIndex.from_product([all_cities, all_dates], names=['City_name', 'Date'])
dfull = dfcity.set_index(['City_name', 'Date']).reindex(new_index).reset_index()
dfull.shape
dfull['City_name'].nunique()

### EXTRA (export the dfull dataset)
# dfull.to_csv('/home/joao/Dropbox/Aula/Pesquisa/michel/city_date_merged_all_dates.csv', index=False)

### Create a table with the number of missings for each city in dfull
missing_counts = dfull.groupby('City_name')['ct_target'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count')
missing_counts = missing_counts.sort_values(by='missing_count', ascending=False)
print(missing_counts)

### Create a table with the percentage of missings for each city in dfull
total_counts = dfull.groupby('City_name')['ct_target'].count().reset_index(name='total_count')
missing_percentage = pd.merge(missing_counts, total_counts, on='City_name')
missing_percentage['missing_percentage'] = (missing_percentage['missing_count'] / (missing_percentage['missing_count'] + missing_percentage['total_count'])) * 100
missing_percentage = missing_percentage.sort_values(by='missing_percentage', ascending=False)
print(missing_percentage)

### Create a 7 days rolling median to cover daily missings
#dfull['ct_targetd'] = dfull.groupby('City_name')['ct_target'].transform(lambda x: x.rolling(window=7, min_periods=1).median())

### Create a non-linear spline interpolation to cover daily missings
#dfull['ct_targetd'] = dfull.groupby('City_name')['ct_target'].transform(lambda x: x.interpolate(method='spline', order=2, limit=7, limit_direction='both'))

def robust_spline_or_median(series):
    # CONFIGURATION
    limit_days = 7  # Max consecutive NaNs to fill
    k = 2           # Spline Order (Quadratic)
    
    # 1. Count valid points to see if Spline is mathematically possible
    # Quadratic (k=2) needs at least 3 points
    valid_points = series.count()
    
    # 2. STRATEGY A: Try Quadratic Spline
    if valid_points > k:
        try:
            return series.interpolate(method='spline', order=k, limit=limit_days, limit_direction='both')
        except:
            pass # If it fails (e.g. singular matrix), fall through to Strategy B
            
    # 3. STRATEGY B: Fallback to Rolling Median
    # We calculate the rolling median of the EXISTING data
    # min_periods=1 ensures we get a value even if we only have 1 data point nearby
    rolling_median = series.rolling(window=limit_days, min_periods=1).median()
    
    # We use this calculated median to fill the NaNs in the original series
    # limit=limit_days ensures we respect your rule of not filling huge gaps
    return series.fillna(rolling_median, limit=limit_days)

# Apply the function
dfull['ct_targetd'] = dfull.groupby('City_name')['ct_target'].transform(robust_spline_or_median)

### Is there any missing data after rolling?
missing_after_rolling = dfull['ct_targetd'].isnull().sum()
print(f'Missing data points after 7-day rolling median: {missing_after_rolling}')

### Create a table with the number of missings for each city in dfull using ct_targetd (daily)
missing_counts = dfull.groupby('City_name')['ct_targetd'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count')
missing_counts = missing_counts.sort_values(by='missing_count', ascending=False)
print(missing_counts)

### Create a table with the percentage of missings for each city in dfull using ct_targetd (daily)
total_counts = dfull.groupby('City_name')['ct_targetd'].count().reset_index(name='total_count')
missing_percentage = pd.merge(missing_counts, total_counts, on='City_name')
missing_percentage['missing_percentage'] = (missing_percentage['missing_count'] / (missing_percentage['missing_count'] + missing_percentage['total_count'])) * 100
missing_percentage = missing_percentage.sort_values(by='missing_percentage', ascending=False)
print(missing_percentage)

# ### Create a weekly based dataframe (median grouping by week)
# ### First create a dfmini dataset with only Date, City_name and ct_target
# dfmini = dfull[['Date', 'City_name', 'ct_target']]

# ### Then convert the daily dataset (dfull) to weekly dataset (dfweek)
# dfweek = dfmini.groupby(['City_name', pd.Grouper(key='Date', freq='W')]).median()
# dfweek = dfweek.reset_index()

# ### Create a table with the number of missings for each city in dfweek using ct_target (weekly)
# missing_counts = dfweek.groupby('City_name')['ct_target'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count')
# missing_counts = missing_counts.sort_values(by='missing_count', ascending=False)
# print(missing_counts)

# ### Create a table with the percentage of missings for each city in dfweek using ct_target (weekly)
# total_counts = dfweek.groupby('City_name')['ct_target'].count().reset_index(name='total_count')
# missing_percentage = pd.merge(missing_counts, total_counts, on='City_name')
# missing_percentage['missing_percentage'] = (missing_percentage['missing_count'] / (missing_percentage['missing_count'] + missing_percentage['total_count'])) * 100
# missing_percentage = missing_percentage.sort_values(by='missing_percentage', ascending=False)
# print(missing_percentage)

# ### Create a monthly based dataframe (median grouping by month)
# ### Then convert the daily dataset (dfull) to monthly dataset (dfmonth)
# dfmonth = dfmini.groupby(['City_name', pd.Grouper(key='Date', freq='ME')]).median()
# dfmonth = dfmonth.reset_index()

# ### Create a table with the number of missings for each city in dfmonth using ct_target (monthly)
# missing_counts = dfmonth.groupby('City_name')['ct_target'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count')
# missing_counts = missing_counts.sort_values(by='missing_count', ascending=False)
# print(missing_counts)

# ### Create a table with the percentage of missings for each city in dfmonth using ct_target (monthly)
# total_counts = dfmonth.groupby('City_name')['ct_target'].count().reset_index(name='total_count')
# missing_percentage = pd.merge(missing_counts, total_counts, on='City_name')
# missing_percentage['missing_percentage'] = (missing_percentage['missing_count'] / (missing_percentage['missing_count'] + missing_percentage['total_count'])) * 100
# missing_percentage = missing_percentage.sort_values(by='missing_percentage', ascending=False)
# print(missing_percentage)
# ### END OF FILE