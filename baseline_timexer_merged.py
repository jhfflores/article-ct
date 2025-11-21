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

## import file
orig_df = pd.read_csv('/home/joao/Dropbox/Aula/Pesquisa/michel/city_date_merged.csv')

# info in general
orig_df.head()
orig_df.shape
orig_df.info()
orig_df.dtypes
# State as a category
orig_df["State_norm"] = orig_df["State_norm"].astype("category")
# Dates as a datetime
orig_df['Date'] = pd.to_datetime(orig_df['Date'])
# indexing by Date
orig_df = orig_df.set_index('Date')
# groupby Date
group_df = orig_df.groupby(['Date', 'State_norm'])['city_Ct_Value'].median().reset_index()
group_df.head()




for state_name, subdf in orig_df.groupby("State_norm"):
    subdf = subdf.sort_index()
    plt.plot(subdf.index, subdf["city_Ct_Value"], label=state_name)

# Labels and legend
plt.xlabel("Date")
plt.ylabel("X value")
plt.title("Time series of X by state")
plt.legend(title="State")

plt.tight_layout()
plt.show()

def plot_state(df, state_name):
    subdf = df[df["State_norm"] == state_name]

    plt.figure(figsize=(12, 5))
    plt.plot(subdf.index, subdf["city_Ct_Value"])
    plt.title(f"Time series of city_Ct_Value – {state_name}")
    plt.xlabel("Date")
    plt.ylabel("X value")
    plt.tight_layout()
    plt.show()

plot_state(group_df, "RS")

# 2. Create a TimeSeriesDataSet
# This handles data preparation, feature engineering, and splitting
max_prediction_length = 10
max_encoder_length = 50

training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx", "exogenous_var"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["value"],
    target_normalizer=GroupNormalizer(groups=["group_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)


# 3. Create DataLoaders
batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=1)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=1)

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

# 5. Make Predictions
best_model_path = trainer.checkpoint_callback.best_model_path
best_timexer = TimeXer.load_from_checkpoint(best_model_path)

# Predict on the validation set or new data
raw_predictions = best_timexer.predict(val_dataloader, mode="raw", return_x=True)
predictions = best_timexer.predict(val_dataloader)

print("Sample raw predictions:", raw_predictions.output.prediction[0])
print("Sample predictions:", predictions[0])