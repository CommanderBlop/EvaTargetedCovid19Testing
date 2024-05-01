# import pandas as pd
# import numpy as np
# from scipy.optimize import root_scalar
# from scipy.stats import beta

# np.random.seed(123)

# def gittins_bandit(manifest, ports, types, g):
#     # Rename columns
#     ports = ports.rename(columns={"Entry_point": "portID", "updated_capacity": "testsLeft"})
#     manifest = manifest.rename(columns={"eb_type": "Country", "point_entry": "portID"})
#     types = types.rename(columns={"eb_type": "Country", "alpha.post": "a", "beta.post": "b", "num_tested": "n"})
    
#     # Convert factors to string
#     manifest["Country"] = manifest["Country"].astype(str)
#     manifest["portID"] = manifest["portID"].astype(str)
#     types["Country"] = types["Country"].astype(str)
#     ports["portID"] = ports["portID"].astype(str)
    
#     # Get total tests
#     N = round(ports["testsLeft"].sum())
    
#     # Create table to store Gittins index for each allowable country
#     gittins = types.copy()
#     gittins["index"] = np.nan
#     gittins["numLeft"] = np.nan
#     gittins["alloc"] = 0
    
#     # Set everyone to initially not test
#     manifest["to_test"] = False
    
#     # Randomly permute manifest across ports
#     manifest = manifest.sample(frac=1).reset_index(drop=True)
    
#     # Sort ports by remaining tests left
#     ports = ports.sort_values("testsLeft", ascending=False)
    
#     # Reset index to default
#     gittins.reset_index(drop=True, inplace=True)
    
#     # Compute index for each country
#     for i in range(len(gittins)):
#         # Get params (WIDENING: add .5 to a and b when computing index)
#         # Artificial certainty-equivalent update for tests in last 48 hours
#         succ = gittins.loc[i, "a"] / (gittins.loc[i, "a"] + gittins.loc[i, "b"])
#         a = gittins.loc[i, "a"] + gittins.loc[i, "tests_last_48"] * succ + 0.25
#         b = gittins.loc[i, "b"] + gittins.loc[i, "tests_last_48"] * (1 - succ) + 0.25
        
#         # Compute Gittins index
#         def f(x):
#             return x - (a / (a + b)) * (1 - g * beta.cdf(x, a + 1, b)) + g * x * (1 - beta.cdf(x, a, b))
        
#         gittins.loc[i, "index"] = root_scalar(f, bracket=[0, 1]).root
        
#         # Compute number of passengers left from this country that can be tested
#         gittins.loc[i, "numLeft"] = (manifest["Country"] == gittins["Country"][i]).sum()
    
#     gittins = gittins.sort_values("index", ascending=False)
    
#     gittins = gittins.groupby('Country').agg({
#     'a': 'last', 
#     'b': 'last',
#     'n': 'sum',
#     'index': 'last'
#     }).reset_index()

#     # Save initial Gittins table (archival w/ date & sample outputs)
#     gittins[["Country", "a", "b", "n", "index"]].to_csv(f"../sample_outputs/country_gittins_{pd.Timestamp.today().strftime('%Y-%m-%d')}.csv", index=False)
#     gittins["numLeft"] = np.nan
#     gittins["alloc"] = 0
#     # Remove countries with no passengers to test OR no tests left at relevant ports
#     rem_ports = ports[ports["testsLeft"] > 0]["portID"]
#     rem_countries = manifest[(manifest["portID"].isin(rem_ports)) & (manifest["to_test"] == False)]["Country"].unique()
#     gittins = gittins[(gittins["numLeft"] > 0) & (gittins["Country"].isin(rem_countries))]
    
#     print(gittins)
#     # Allocate tests one by one, & re-compute indices until (we run out of tests or types to test)
#     while N > 0 and len(gittins) > 0:
#         # Allocate test to highest index
#         loc = gittins.loc[gittins["index"].idxmax(), "Country"]
#         ind_loc = gittins["index"].idxmax()
        
#         # Pick first untested passenger from that country (at a port with most tests left) to test
#         ind = manifest[(manifest["to_test"] == False) & (manifest["Country"] == loc) & (manifest["portID"].isin(rem_ports))].index
        
#         # Look at possible ports & pick the one w/ most tests; THEN pick a passenger
#         tmp = ports[ports["portID"].isin(manifest.loc[ind, "portID"])].sort_values("testsLeft", ascending=False)["portID"].iloc[0]
#         ind = manifest[(manifest["to_test"] == False) & (manifest["Country"] == loc) & (manifest["portID"] == tmp)].index[0]
#         manifest.loc[ind, "to_test"] = True
        
#         # Update succ, fail, Gittins, curr prev (WIDENING: add .5 to a and b when computing index)
#         # a = gittins.loc[ind_loc, "a"] + gittins.loc[ind_loc, "a"] / (gittins.loc[ind_loc, "a"] + gittins.loc[ind_loc, "b"]) + 0.25
#         # b = gittins.loc[ind_loc, "b"] + gittins.loc[ind_loc, "b"] / (gittins.loc[ind_loc, "a"] + gittins.loc[ind_loc, "b"]) + 0.25
#         # # gittins.loc[ind_loc, ["a", "b", "index"]] = [a - 0.25, b - 0.25, root_scalar(f, bracket=[0, 1]).root]
#         # gittins.loc[i, "index"] = root_scalar(lambda x: x - (a / (a + b)) * (1 - g * beta.cdf(x, a + 1, b)) + g * x * (1 - beta.cdf(x, a, b)), bracket=[0, 1]).root
        
#         a = gittins.loc[ind_loc, "a"] + gittins.loc[ind_loc, "a"] / (gittins.loc[ind_loc, "a"] + gittins.loc[ind_loc, "b"]) + 0.25
#         b = gittins.loc[ind_loc, "b"] + gittins.loc[ind_loc, "b"] / (gittins.loc[ind_loc, "a"] + gittins.loc[ind_loc, "b"]) + 0.25

#         f = lambda x: x - (a / (a + b)) * (1 - g * beta.cdf(x, a + 1, b)) + g * x * (1 - beta.cdf(x, a, b))
#         index = root_scalar(f, bracket=[0, 1]).root

#         gittins.loc[ind_loc, ["a", "b", "index"]] = [a - 0.25, b - 0.25, index]
#         # Update num passengers left & assign an allocation to that country
#         gittins.loc[ind_loc, "numLeft"] -= 1
#         gittins.loc[ind_loc, "alloc"] += 1
        
#         # Update num tests in that port and overall
#         ind_port = ports[ports["portID"] == manifest.loc[ind, "portID"]].index[0]
#         ports.loc[ind_port, "testsLeft"] -= 1
#         N -= 1
        
#         # Remove countries with no passengers to test OR no tests left at relevant ports
#         rem_ports = ports[ports["testsLeft"] > 0]["portID"]
#         rem_countries = manifest[(manifest["portID"].isin(rem_ports)) & (manifest["to_test"] == False)]["Country"].unique()
#         gittins = gittins[(gittins["numLeft"] > 0) & (gittins["Country"].isin(rem_countries))]
    
#     # Remove extraneous columns
#     manifest = manifest[["id", "to_test"]]
#     print(manifest)
#     return manifest

import pandas as pd
import numpy as np
from scipy.stats import beta
from scipy.optimize import root_scalar
import random

def gittins_bandit(manifest, ports, types, g):
    # Rename columns
    ports = ports.rename(columns={"Entry_point": "portID", "updated_capacity": "testsLeft"})
    manifest = manifest.rename(columns={"eb_type": "Country", "point_entry": "portID"})
    types = types.rename(columns={"eb_type": "Country", "alpha.post": "a", "beta.post": "b", "num_tested": "n"})

    # Convert factors to string
    manifest["Country"] = manifest["Country"].astype(str)
    manifest["portID"] = manifest["portID"].astype(str)
    types["Country"] = types["Country"].astype(str)
    ports["portID"] = ports["portID"].astype(str)

    # Get total tests
    N = round(ports["testsLeft"].sum())

    # Randomly permute manifest across ports
    manifest = manifest.sample(frac=1).reset_index(drop=True)

    # Sort ports by remaining tests left
    ports = ports.sort_values(by="testsLeft", ascending=False)

    # Initialize Gittins indices
    types["index"] = np.nan
    types["numLeft"] = manifest.groupby('Country').size()
    types["alloc"] = 0

    # Set everyone to initially not test
    manifest["to_test"] = False

    # Compute index for each country
    for i, row in types.iterrows():
        a = row["a"] + row["tests_last_48"] * (row["a"] / (row["a"] + row["b"])) + 0.25
        b = row["b"] + row["tests_last_48"] * (1 - row["a"] / (row["a"] + row["b"])) + 0.25
        f = lambda x: x - (a / (a + b)) * (1 - g * beta.cdf(x, a + 1, b)) + g * x * (1 - beta.cdf(x, a, b))
        result = root_scalar(f, bracket=[0, 1], method='brentq')
        types.at[i, 'index'] = result.root

    # Allocate tests one by one
    while N > 0 and not types.empty:
        max_index = types['index'].idxmax()
        country = types.loc[max_index, "Country"]
        available_passengers = manifest[(manifest["Country"] == country) & (~manifest["to_test"])]

        if available_passengers.empty:
            types.drop(index=max_index, inplace=True)
            continue

        selected_passenger = available_passengers.iloc[0]
        manifest.at[selected_passenger.name, 'to_test'] = True
        # print(ports["portID"], selected_passenger["portID"])
        ind_port = ports[ports["portID"] == manifest.loc[max_index, "portID"]].index[0]
        ports.loc[ind_port, "testsLeft"] -= 1
        N -= 1

        # Update indices
        types.at[max_index, "numLeft"] -= 1
        types.at[max_index, "alloc"] += 1
        if types.at[max_index, "numLeft"] == 0:
            types.drop(index=max_index, inplace=True)

    return manifest

# Example data needs to be defined or loaded here to run the function.
