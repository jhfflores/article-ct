## packages
import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import Chronos2Pipeline

## files
#patient = pd.read_excel('/home/joao/Dropbox/Aula/Pesquisa/michel/patient.xlsx', na_values='NaN')
#city = pd.read_csv('/home/joao/Dropbox/Aula/Pesquisa/michel/city.csv')
state = pd.read_excel('/home/joao/Dropbox/Aula/Pesquisa/michel/state.xlsx', na_values='NaN')
#patient.head()
#city.head()
#city[city['Ct_Value'].notna()].head()
state[state['Ct_Value'].notna()].head()

### main work using city

## remove NaN (just for testing)
statenout = state.dropna().copy()
stateCTout = state[state['Ct_Value'].notna()].copy()
statenout.shape; stateCTout.shape; state.shape;

#citynout = city.dropna().copy()
#cityCTout = city[city['Ct_Value'].notna()]
#citynout.shape; cityCTout.shape; city.shape;

## datetime sorting
statenout['Date'] = pd.to_datetime(statenout['Date'], format = '%y-%m-%d')
statenout.head()
statenout = statenout.sort_values(by='Date')
statenout.head()
statenout = statenout.drop(columns = 'State')
statenout_mean = statenout.groupby('Date').mean().reset_index()
statenout_med = statenout.groupby('Date').median().reset_index()
statenout_mean.head(); statenout_med.head()

statenout_mean.shape; statenout_med.shape

statenout_med.columns

### CHRONOS-2

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# With Chronos2 we can't have missing dates, so we need to fill the dataset
# testing if some days are missing
diffs = statenout_med["Date"].diff().value_counts()
print(diffs)
# as it is, fill with missing data
statenout_med_mod = statenout_med.set_index('Date').copy()
statenout_med_mod = statenout_med_mod.asfreq('D')
statenout_med_mod = statenout_med_mod.reset_index()
# verify
diffs = statenout_med_mod["Date"].diff().value_counts()
print(diffs)
# rewriting
statenout_med = statenout_med_mod.copy()

# dataset clean up and basic adjusting
statenout_med['id'] = 'serie_state'  # this is necessary
context = statenout_med.drop(columns = ['Latitude','Longitude']).iloc[:750]  # do not think these variables where necessary

# real values for error/comparison
test_df = statenout_med.drop(columns = ['Latitude','Longitude']).iloc[750:]
future_df = test_df.drop(columns="Ct_Value")


# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context,
    future_df=future_df,
    prediction_length=44,  # Number of steps to forecast (based on the .iloc definition above)
    quantile_levels=[0.25, 0.5, 0.75],  # Quantile for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="Date",  # Column with datetime information
    target="Ct_Value",  # Column(s) with time series values to predict
)

# basic reviewing data
future_df.shape
pred_df.head()
pred_df.shape
pred_df.info()
test_df.shape

# to verify errors
pred = pred_df['predictions'].reset_index(drop=True)
test = test_df['Ct_Value'].reset_index(drop=True)
error = pred-test
error2 = error**2
error.head()
error.describe()
error2.head()
error2.describe()
