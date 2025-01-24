from data_loader import load_data
from data_cleaner import clean_data

df = load_data(
    data_source_name="kaggle_brisT1D", dataset_type="train"
)

print(df.columns)