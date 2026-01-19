### all the packages (need clean up)
import torch
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models import TimeXer
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import matplotlib.pyplot as plt
import runpy
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

### FIRST WE DO THE DATA EXPLORATION AND PREPARATION AS IN data_exploring.py
# (Assuming the prepared dataframe is named 'dfull' with columns: 'City_name', 'Date', 'ct_targetd')

data_script = runpy.run_path("/home/joao/Dropbox/Aula/Pesquisa/michel/data_prepare_daily.py")
dfull = data_script['dfull']

dfull.head()


# Sort appropriately to ensure the counter is correct
dfull = dfull.sort_values(['City_name', 'Date'])

# Create the 'time_idx' column
# This converts dates into a number: 0, 1, 2, ... based on the minimum date in the whole dataset.
dfull['time_idx'] = (dfull['Date'] - dfull['Date'].min()).dt.days

# Check the result
print(dfull[['City_name', 'Date', 'time_idx', 'ct_targetd']].head())
# Rename 'ct_targetd' to 'value' for TimeXer compatibility
dfull = dfull.rename(columns={'ct_targetd': 'value'})

#############
### Here we are isolating SAO PAULO to train TimeXer only with this city data
dfull_sp = dfull[dfull['City_name'] == 'SAO PAULO'].copy()

# Ensure the time_idx is correct and sorted
dfull_sp = dfull_sp.sort_values('time_idx')

# Create a TimeSeriesDataSet
# This handles data preparation, feature engineering, and splitting
max_prediction_length = 14  # predict 14 days into the future
max_encoder_length = 30  # look back 30 days

training_cutoff = dfull_sp["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    dfull_sp[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    group_ids=["City_name"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["City_name"],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["value"],
    target_normalizer=GroupNormalizer(groups=["City_name"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, dfull_sp, predict=True, stop_randomization=True)

# Create DataLoaders
batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

# 4. Initialize and Train the TimeXer Model
timexer = TimeXer.from_dataset(
    training,
    hidden_size=64,
    n_heads=4,
    e_layers=2,
    d_ff=128,
    dropout=0.1,
    patch_length=5,
    learning_rate=1e-3
)

trainer = pl.Trainer(
    max_epochs=10,
    gradient_clip_val=0.1,
)

trainer.fit(
    timexer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)

# Make Predictions
best_model_path = trainer.checkpoint_callback.best_model_path
best_timexer = TimeXer.load_from_checkpoint(best_model_path)

# Predict on the validation set or new data
raw_predictions = best_timexer.predict(val_dataloader, mode="raw", return_x=True)
predictions = best_timexer.predict(val_dataloader)

print("Sample raw predictions:", raw_predictions.output.prediction[0])
print("Sample predictions:", predictions[0])


######################################
######################################
######################################
### Evaluation pipeline
######################################
######################################
######################################

training_dataset = train_dataloader.dataset

def prepare_clean_recent_data(df_full, window_size=45):
    """
    Returns a DataFrame containing only the last 'window_size' rows for each city, 
    excluding cities that contain any NaN values during this period.
    """
    # 1. Select only the last 45 rows for each city
    # group_keys=False prevents the creation of an unnecessary MultiIndex
    print("Adjusting last data...")
    df_recent = df_full.groupby('City_name', group_keys=False).apply(
        lambda x: x.sort_values('time_idx').tail(window_size)
    )
    
    # 2. Identify 'clean' cities (No NaN values within the critical window)
    # Groups by city and checks if ALL target values are non-null
    valid_status = df_recent.groupby('City_name')['value'].apply(lambda x: x.notna().all())

    # Approved cities list
    approved_cities = valid_status[valid_status == True].index.tolist()
    rejected_cities = valid_status[valid_status == False].index.tolist()
    
    print(f"Approved cities: {len(approved_cities)}")
    print(f"Rejected cities (with NaN in the last {window_size} days): {len(rejected_cities)}")
    
    # 3. Dataframe filter with only approved cities
    df_clean = df_recent[df_recent['City_name'].isin(approved_cities)].copy()

    return df_clean, approved_cities

#######     execute the data preparation for evaluation    ########
# window_size = 30 (Encoder) + 15 (security margin)  = 45
dfinal, cidades_aptas = prepare_clean_recent_data(dfull, window_size=45)

def evaluate_clean_cities(df_clean_recent, model, train_dataset):
    results = []
    cities = df_clean_recent['City_name'].unique()
    
    print(f"Starting evaluation on {len(cities)} filtered cities...")
    
    for city in cities:
        # Take city-specific data
        df_city = df_clean_recent[df_clean_recent['City_name'] == city].copy()
        
        # Identity Swap 
        df_city['City_name'] = 'SAO PAULO'
        
        try:
            # Since df_city only has 45 rows, TimeSeriesDataSet will use ALL of it.
            # predict=True ensures it uses the tail end of the data for prediction.
            inference_ds = TimeSeriesDataSet.from_dataset(
                train_dataset, df_city, predict=True, stop_randomization=True
            )
            
            inference_loader = inference_ds.to_dataloader(train=False, batch_size=1)
            
            # Prediction and Observed extraction
            y_pred = model.predict(inference_loader).flatten().cpu().numpy()
            y_true = torch.cat([y[0] for x, y in iter(inference_loader)]).flatten().cpu().numpy()
            
            # Evaluation metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            
            results.append({
                'City': city,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE (%)': mape
            })
            
        except Exception as e:
            print(f"Unexpected error in {city}: {e}")
            continue

    # Final table
    return pd.DataFrame(results).set_index('City').sort_values('MAPE (%)')

# --- DOES EVERYTHING ---
relatorio_final = evaluate_clean_cities(dfinal, best_timexer, training_dataset)
print(relatorio_final)