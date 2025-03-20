import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
STOCK_DATA_FOLDER = f"{PARENT_DIR}/stock_data"

YEARS_TO_FORECAST = 10
RETURN_PERIOD = 5
SELF_BUSINESS_RETURNS_DAYS = [5, 15, 30, 60]

NUM_FOLDS = 10
SEED = 42
SCORING = "neg_mean_squared_error"
