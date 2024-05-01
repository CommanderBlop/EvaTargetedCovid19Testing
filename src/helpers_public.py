import pandas as pd
import numpy as np
from datetime import datetime

def adjust_budgets(pass_manifest, port_budgets, white_list=None, min_testing_budget=-1e6, log_file="log.txt"):
    # Read in the white_list
    if white_list is None:
        white_list = pd.read_csv("../sample_input_data_fake/countries_allowed.csv", names=["country"])
    
    pass_manifest["cntry_flagged"] = ~pass_manifest["country"].isin(white_list["country"])
    
    tests_for_flag = pass_manifest[pass_manifest["cntry_flagged"]].groupby("point_entry").size().reset_index(name="num_flagged")
    
    # Update the budgets on each port
    port_budgets = port_budgets.merge(tests_for_flag, left_on="Entry_point", right_on="point_entry", how="left")
    port_budgets["num_flagged"] = port_budgets["num_flagged"].fillna(0)
    port_budgets["updated_capacity"] = np.where(port_budgets["Capacity"] > 0, port_budgets["Capacity"] - port_budgets["num_flagged"], 0)
    
    # Artificially increase them up if debugging
    port_budgets["updated_capacity"] = np.maximum(min_testing_budget, port_budgets["updated_capacity"])
    
    # Check if certain ports are exceeding budgets due to flags and log.
    if port_budgets["updated_capacity"].min() < 0:
        with open(log_file, "a") as f:
            f.write("\n \n Some ports budgets Exhausted on Black-Listed Countries!\n ")
            port_budgets[port_budgets["updated_capacity"] < 0].to_csv(f, index=False)
        
        # Round them to zero for purpose of run.
        port_budgets["updated_capacity"] = np.maximum(0, port_budgets["updated_capacity"])
    
    # Clean it up to look right for bandit.
    port_budgets = port_budgets[["Entry_point", "Capacity", "updated_capacity", "Target_Capacity"]]
    
    return port_budgets, pass_manifest

def mm_beta_dist(mom1, mom2):
    # MM estimates not defined in this case
    if mom1 <= mom2:
        return pd.DataFrame({"alpha": [np.nan], "beta": [-1]})
    if mom2 <= mom1 ** 2:
        return pd.DataFrame({"alpha": [-1], "beta": [np.nan]})
    
    var = mom2 - mom1 ** 2
    alpha = (mom1 ** 2 * (1 - mom1) / var) - mom1
    beta = ((mom1 * (1 - mom1) / var) - 1) * (1 - mom1)
    return pd.DataFrame({"alpha": [alpha], "beta": [beta]})

def fit_eb_MM(df, col_mom1, col_mom2, method_name):
    df = df.reset_index(drop=True)
    df[method_name] = df.apply(lambda x: mm_beta_dist(x[col_mom1], x[col_mom2]), axis=1)
    return df

# def mm_beta_dist(mom1, mom2):
#     # This function should implement whatever mm_beta_dist does in R
#     # Placeholder: compute alpha and beta from moment estimates
#     alpha = mom1 * ((mom1 * (1 - mom1) / mom2) - 1)
#     beta = (1 - mom1) * ((mom1 * (1 - mom1) / mom2) - 1)
#     return alpha, beta

# def fit_eb_MM(df, col_mom1, col_mom2, method_name):
#     # Assuming df[col_mom1] and df[col_mom2] exist and are valid
#     results = df.apply(lambda x: mm_beta_dist(x[col_mom1], x[col_mom2]), axis=1, result_type='expand')
#     df[method_name] = results.apply(lambda x: {'alpha': x[0], 'beta': x[1]}, axis=1)
#     return df

# def add_eb_preds(df, pred_name, param_name, pos_name, total_name):
#     df = df.reset_index(drop=True)
#     df = df.join(df[param_name].apply(pd.Series))
#     df["alpha.post"] = df["alpha"] + df[pos_name]
#     df["beta.post"] = df["beta"] + df[total_name] - df[pos_name]
#     df[pred_name] = df["alpha.post"] / (df["alpha.post"] + df["beta.post"])
#     df[param_name] = df[["alpha", "beta", "alpha.post", "beta.post"]].apply(lambda x: x.to_dict(), axis=1)
#     return df

# def add_eb_preds(df, pred_name, param_name, pos_name, total_name):
#     df = df.reset_index(drop=True)
#     param_df = pd.json_normalize(df[param_name])
#     df = pd.concat([df, param_df], axis=1)
#     df["alpha.post"] = df["alpha"] + df[pos_name]
#     df["beta.post"] = df["beta"] + df[total_name] - df[pos_name]
#     df[pred_name] = df["alpha.post"] / (df["alpha.post"] + df["beta.post"])
#     df[param_name] = df[["alpha", "beta", "alpha.post", "beta.post"]].apply(lambda x: x.to_dict(), axis=1)
#     return df

def add_eb_preds(df, pred_name, param_name, pos_name, total_name):
    # Assuming df[param_name] contains 'alpha' and 'beta' as columns in a nested DataFrame
    # First, ensure the column is not nested. If nested, we need to unnest it
    if isinstance(df[param_name].iloc[0], pd.DataFrame):
        param_df = pd.concat(df[param_name].tolist()).reset_index(drop=True)
    else:
        param_df = df[param_name]
    
    # Calculate new alpha and beta
    param_df['alpha.post'] = param_df['alpha'] + df[pos_name]
    param_df['beta.post'] = param_df['beta'] + df[total_name] - df[pos_name]
    
    # Calculate the prediction
    df[pred_name] = param_df['alpha.post'] / (param_df['alpha.post'] + param_df['beta.post'])
    
    # If you need to nest back the parameters, you can adjust this part accordingly
    # For now, we will just add the new alpha and beta back to the main DataFrame
    df = pd.concat([df, param_df[['alpha.post', 'beta.post']]], axis=1)
    return df


def label_eb_types(dat, dt_col_name, out_ctry_col_name, END_OF_TIME=pd.to_datetime("21210101", format="%Y%m%d")):
    # Read in the grey_list data
    grey_list_se = pd.read_csv("../sample_input_data_fake/grey_list_start_end.csv", parse_dates=["end_date"])
    grey_list_se["end_date"] = grey_list_se["end_date"].fillna(END_OF_TIME)
    
    dat = dat.merge(grey_list_se, on="country", how="left")
    dat["data_pt_grey"] = (dat["start_date"] <= dat[dt_col_name]) & (dat[dt_col_name] <= dat["end_date"])
    dat["data_pt_grey"] = dat["data_pt_grey"].fillna(False)
    dat["eb_type"] = np.where(dat["data_pt_grey"], dat["country"] + "*", dat["country"])
    dat[out_ctry_col_name] = dat["data_pt_grey"]
    dat = dat.drop(columns=["data_pt_grey", "start_date", "end_date"])
    return dat

def label_eb_types_city(dat, dt_col_name, out_ctry_col_name, END_OF_TIME=pd.to_datetime("21210101", format="%Y%m%d")):
    # Read in the grey_list data
    grey_list_se = pd.read_csv("../sample_input_data_fake/grey_list_start_end.csv", parse_dates=["end_date"])
    grey_list_se["end_date"] = grey_list_se["end_date"].fillna(END_OF_TIME)
    
    # Read in the cities with significant differences from country.
    city_types = pd.read_csv("../sample_input_data_fake/city_types.csv")
    city_types["country"] = city_types["country"].str.upper()
    city_types["city"] = city_types["city"].str.upper()
    city_types["ctry_city"] = city_types["country"] + "_" + city_types["city"]
    city_types["eb_type"] = city_types["ctry_city"]
    
    # First go through and label the eb_type as either XX or XX_city if in the city types (neglecting grey-listing)
    dat["ctry_city"] = dat["country"] + "_" + dat["city"]
    dat = dat.merge(city_types[["ctry_city", "eb_type"]], on="ctry_city", how="left")
    dat["eb_type"] = dat["eb_type"].fillna(dat["country"])
    dat = dat.drop(columns=["ctry_city"])
    
    # Now go through and back label anyone that was grey
    dat = dat.merge(grey_list_se, on="country", how="left")
    dat["data_pt_grey"] = (dat["start_date"] <= dat[dt_col_name]) & (dat[dt_col_name] <= dat["end_date"])
    dat["data_pt_grey"] = dat["data_pt_grey"].fillna(False)
    dat["eb_type"] = np.where(dat["data_pt_grey"], dat["eb_type"] + "*", dat["eb_type"])
    dat[out_ctry_col_name] = dat["data_pt_grey"]
    dat = dat.drop(columns=["data_pt_grey", "start_date", "end_date"])
    return dat

def clean_hist_plf_data(plf_data, today_dt, log_file, use_city_types=True):
    # Dangerous: Namibia is stored as NA in iso_codes which confuses pandas. Rename it to NA_
    plf_data["country"] = np.where(plf_data["country"].isna(), "NA_", plf_data["country"])
    
    # Back label points by their eb-type based on grey_list
    if use_city_types:
        plf_data = label_eb_types_city(plf_data, "date_entry", "isCtryGrey")
    else:
        plf_data = label_eb_types(plf_data, "date_entry", "isCtryGrey")
    
    # White list needed to mark flagged countries in historical data
    ctry_white_list = pd.read_csv("../sample_input_data_fake/countries_allowed.csv", names=["country"])
    
    plf_data["to_test_raw"] = plf_data["to_test"]
    plf_data["sent_for_test_raw"] = plf_data["sent_for_test"]
    plf_data["test_result_raw"] = plf_data["test_result"]
    plf_data["to_test"] = np.where(plf_data["to_test"].isna(), "not_seen", plf_data["to_test"])
    plf_data["sent_for_test"] = np.where(plf_data["test_result"].notna(), True, plf_data["sent_for_test"])
    plf_data["sent_for_test"] = np.where(plf_data["sent_for_test"].isna(), False, True)
    plf_data["test_result"] = np.where(plf_data["test_result"].isna() & plf_data["sent_for_test"].notna(), "negative", plf_data["test_result"])
    plf_data["isCtryFlagged"] = ~plf_data["country"].isin(ctry_white_list["country"])
    plf_data["date_entry"] = pd.to_datetime(plf_data["date_entry"], format="%Y-%m-%d")
    return plf_data