import pandas as pd
from datetime import datetime
import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Bandit_Public import *
from helpers_public import *
# Set random seed
np.random.seed(123)
# Constants
TODAY_DT = pd.to_datetime("2020-09-01")
GITTINS_DISCOUNT_FACT = 0.9
LOG_FILE = "log.txt"
MIN_TEST = 0

# Read input data
pass_manifest = pd.read_csv("../sample_input_data_fake/pass_manifest.csv")
plf_data = pd.read_csv("../sample_input_data_fake/hist_db_working.csv", dtype={
    "result_id": float,
    "country": str,
    "city": str,
    "date_entry": str,
    "point_entry": str,
    "to_test": str,
    "test_result": str,
    "sent_for_test": float
})
port_budgets = pd.read_csv("../sample_input_data_fake/port_budgets.csv")
grey_list_se = pd.read_csv("../sample_input_data_fake/grey_list_start_end.csv", parse_dates=["start_date", "end_date"])
city_types = pd.read_csv("../sample_input_data_fake/city_types.csv")
white_list = pd.read_csv("../sample_input_data_fake/countries_allowed.csv", header=None, names=["country"])

# Clean and preprocess data
pass_manifest["dt_entry"] = TODAY_DT
pass_manifest = label_eb_types_city(pass_manifest, "dt_entry", "isGrey")
pass_manifest = pass_manifest.drop(columns=["dt_entry"])

plf_data = clean_hist_plf_data(plf_data, TODAY_DT, LOG_FILE)

testing_last_48 = plf_data[(plf_data["date_entry"] <= TODAY_DT) & (plf_data["date_entry"] > TODAY_DT - pd.Timedelta(days=2))]
testing_last_48 = testing_last_48.groupby("eb_type")["sent_for_test"].sum().reset_index(name="tests_last_48")

plf_data = plf_data[plf_data["date_entry"] <= TODAY_DT - pd.Timedelta(days=2)]

port_budgets, pass_manifest = adjust_budgets(pass_manifest, port_budgets, white_list)

# Summarize data for bandit
hist_data = plf_data.groupby(["eb_type", "isCtryFlagged", "isCtryGrey"]).agg(
    num_arrivals=("eb_type", "size"),
    num_tested=("sent_for_test", "sum"),
    num_pos=("test_result", lambda x: (x == "positive").sum()),
    num_inconclusive=("test_result", lambda x: x.notna().sum() - (x == "positive").sum() - (x == "negative").sum())
).reset_index()
pass_manifest_unique = pass_manifest.drop_duplicates(subset=['eb_type']).reset_index(drop=True)
hist_data = hist_data.merge(pass_manifest_unique[["eb_type", "cntry_flagged", "isGrey"]], on="eb_type", how="outer")
hist_data = hist_data.fillna({"num_arrivals": 0, "num_pos": 0, "num_tested": 0, "num_inconclusive": 0})
hist_data["isCtryFlagged"] = np.where(hist_data["isCtryFlagged"].isna(), hist_data["cntry_flagged"], hist_data["isCtryFlagged"])
hist_data["isCtryGrey"] = np.where(hist_data["isCtryGrey"].isna(), hist_data["isGrey"], hist_data["isCtryGrey"])
hist_data = hist_data.drop(columns=["isGrey", "cntry_flagged"])
hist_data = hist_data.merge(testing_last_48, on="eb_type", how="left")
hist_data["tests_last_48"] = hist_data["tests_last_48"].fillna(0)

# Fit EB models
hist_data["prev"] = hist_data["num_pos"] / hist_data["num_tested"]
# Constants for default values in case moment matching fails
DEFAULT_MOM1_GREY = 0.00487
DEFAULT_MOM2_GREY = 0.0000419

# White-listed countries
hist_data['isCtryFlagged'] = hist_data['isCtryFlagged'].astype(bool)
hist_data['isCtryGrey'] = hist_data['isCtryGrey'].astype(bool)
t_moments_src = hist_data.query('~isCtryFlagged and ~isCtryGrey')
t_moments_src.to_csv("hist_data.csv", index=False)
print(len(t_moments_src))

# Ensure no NaN values affect calculations
t_moments_src = t_moments_src.dropna(subset=['prev', 'num_pos', 'num_tested'])

# Calculating mom1 and mom2
t_moments = {
    "mom1": t_moments_src['prev'].mean(),
    "mom2": ((t_moments_src['prev'] * (t_moments_src['num_pos'] - 1) / 
             (t_moments_src['num_tested'] - 1)).mean())
}

# Check if moment matching failed, and set default values if necessary
if t_moments['mom1'] <= t_moments['mom2'] or t_moments['mom2'] <= t_moments['mom1']**2:
    print("Whitelist MOMENT MATCHING PROCEDURE FAILED. Default values used.")
    t_moments['mom1'], t_moments['mom2'] = 0.00612, 0.0000593

# Similar blocks for black-list and grey-list countries
t_moments_black_src = hist_data.query('isCtryFlagged')
t_moments_black_src = t_moments_black_src.dropna(subset=['prev', 'num_pos', 'num_tested'])

t_moments_black = {
    "mom1": t_moments_black_src['prev'].mean(),
    "mom2": (t_moments_black_src['prev'] * (t_moments_black_src['num_pos'] - 1) / (t_moments_black_src['num_tested'] - 1)).mean()
}
if t_moments_black['mom1'] <= t_moments_black['mom2'] or t_moments_black['mom2'] <= t_moments_black['mom1']**2:
    print("Black-list MOMENT MATCHING PROCEDURE FAILED. Default values used.")
    t_moments_black['mom1'], t_moments_black['mom2'] = 0.00678, 0.0001
    
t_moments_grey_src = hist_data.query('isCtryGrey')
t_moments_grey_src = t_moments_grey_src.dropna(subset=['prev', 'num_pos', 'num_tested'])
t_moments_grey = {
    "mom1": t_moments_grey_src['prev'].mean(),
    "mom2": (t_moments_grey_src['prev'] * (t_moments_grey_src['num_pos'] - 1) / (t_moments_grey_src['num_tested'] - 1)).mean()
}

if t_moments_grey['mom1'] <= t_moments_grey['mom2'] or t_moments_grey['mom2'] <= t_moments_grey['mom1']**2:
    print("Grey-list MOMENT MATCHING PROCEDURE FAILED. Default values used.")
    t_moments_grey['mom1'], t_moments_grey['mom2'] = DEFAULT_MOM1_GREY, DEFAULT_MOM2_GREY

# Applying the moments to the historical data
hist_data['mom1'] = np.where(hist_data['isCtryFlagged'], t_moments_black['mom1'], t_moments['mom1'])
hist_data['mom2'] = np.where(hist_data['isCtryFlagged'], t_moments_black['mom2'], t_moments['mom2'])
hist_data['mom1'] = np.where((hist_data['isCtryGrey'] & ~hist_data['isCtryFlagged']), t_moments_grey['mom1'], hist_data['mom1'])
hist_data['mom2'] = np.where((hist_data['isCtryGrey'] & ~hist_data['isCtryFlagged']), t_moments_grey['mom2'], hist_data['mom2'])

hist_data = fit_eb_MM(hist_data, "mom1", "mom2", "MM")
hist_data = add_eb_preds(hist_data, "eb_prev", "MM", "num_pos", "num_tested")
hist_data.to_csv("hist_data.csv", index=False)

# Generate current estimates
curr_estimates = hist_data[["eb_type", "isCtryFlagged", "isCtryGrey", "eb_prev", "prev", "num_pos", "num_tested", "num_arrivals", "alpha.post", "beta.post"]]
curr_estimates = curr_estimates.copy()
# curr_estimates["low"] = curr_estimates.apply(lambda x: beta.ppf(0.05, x["alpha.post"], x["beta.post"]), axis=1)
# curr_estimates["up"] = curr_estimates.apply(lambda x: beta.ppf(0.95, x["alpha.post"], x["beta.post"]), axis=1)
curr_estimates.loc[:, "low"] = curr_estimates.apply(lambda x: beta.ppf(0.05, x["alpha.post"], x["beta.post"]), axis=1)
curr_estimates.loc[:, "up"] = curr_estimates.apply(lambda x: beta.ppf(0.95, x["alpha.post"], x["beta.post"]), axis=1)

hist_data['isCtryFlagged'] = hist_data['isCtryFlagged'].astype(bool)
temp_hist_data = hist_data.loc[~hist_data["isCtryFlagged"], ["eb_type", "alpha.post", "beta.post", "num_tested", "tests_last_48"]]
# Run Gittins bandit
temp_manifest = pass_manifest[~pass_manifest["cntry_flagged"]][["id", "eb_type", "point_entry"]]
temp_hist_data = hist_data[~hist_data["isCtryFlagged"]][["eb_type", "alpha.post", "beta.post", "num_tested", "tests_last_48"]]
temp_port_budget = port_budgets[["Entry_point", "Capacity", "updated_capacity", "Target_Capacity"]]

today_testing = gittins_bandit(temp_manifest, temp_port_budget, temp_hist_data, GITTINS_DISCOUNT_FACT)

# Marry bandit results to flagged people and output
today_testing["flagged"] = False
pass_manifest = pass_manifest.merge(today_testing, on="id", how="left")
pass_manifest["flagged"] = np.where(pass_manifest["cntry_flagged"], True, pass_manifest["flagged"])
pass_manifest["flagged"] = np.where(pass_manifest["isGrey"] & pass_manifest["to_test"], True, pass_manifest["flagged"])
pass_manifest["to_test"] = np.where(pass_manifest["cntry_flagged"], True, pass_manifest["to_test"])

# # Write outputs
pass_manifest[["id", "to_test", "flagged"]].to_csv(f"../sample_outputs/test_results_{TODAY_DT.strftime('%Y-%m-%d')}.csv", index=False)