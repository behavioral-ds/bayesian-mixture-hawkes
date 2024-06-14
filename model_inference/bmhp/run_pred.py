import pickle, os, shutil, sys
import pandas as pd
import numpy as np
import logging
from cmdstanpy import CmdStanModel
from time import perf_counter
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import gaussian_kde

import pickle
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, from_csv

exe_dict = {
    0: "stan_files/mixture-pl_0+a0_z0x-sep_modified",
    1: "stan_files/mixture-pl_0+a0y_z0x-sep_modified",
    2: "stan_files/mixture-pl_0+a0_z0xy-sep_modified",
    3: "stan_files/mixture-pl_0+a0y_z0xy-sep_modified"
}

# code for which data to use, input is model index
label_dict = {
    0: "0x(m)",
    1: "0x(m)_0y(t)",
    2: "0x(m)_0y(t)",
    3: "0x(m)_0y(t)"
}

# code for category index
category_dict = {
    0: "ATNIX",
    1: "FAKENIX"
}

magnitude_dict = {
    0: "data/f_m_atnix.p",
    1: "data/f_m_fakenix.p"
}

def run_pred(data, magnitude_data, category_index, publisher_index, model_index, outdir, stan_file):
    model = CmdStanModel(exe_file=stan_file)

    pmf_size = magnitude_data[publisher_index][2].shape[0]
    pmf_x = list(magnitude_data[publisher_index][2]["transformed"].values)
    pmf_vals = list((magnitude_data[publisher_index][2]["count"] / magnitude_data[publisher_index][2]["count"].sum()).values)

    data["pmf_size"] = pmf_size
    data["pmf_x"] = pmf_x
    data["pmf_vals"] = pmf_vals

    fit = from_csv(f"output/{category_index}/{publisher_index}/{model_index}/")
    new_quantities = model.generate_quantities(data=data, previous_fit=fit)
    df = new_quantities.draws_pd()
    df[[x for x in df.columns if 'pred_holdout[' in x or 'actual_holdout[' in x or 'qred_holdout[' in x]].to_csv(outdir + "preds.csv", index=False)
    
def main():

    ### GET ARTIST AND MODEL INDICES ###
    # Get the starting integer and the ending integer.
    if len(sys.argv) != 4: # filename, artist_index, model_index
        print("Usage: %s start end" % sys.argv[0])
        print("Type in pbs index.")
        sys.exit()
    try:
        category_index = int(sys.argv[1])
        publisher_index = int(sys.argv[2])
        model_index = int(sys.argv[3])
    except:
        print("One of your arguments was not an integer.")
        sys.exit()

    stan_file = exe_dict[model_index]
    
    outdir = f"preds/{category_index}/{publisher_index}/{model_index}/"

    magnitude_data = pickle.load(open(magnitude_dict[category_index], "rb"))
        
    logo_data = pickle.load(open(f"data/LOGO_{label_dict[model_index]}_{category_dict[category_index]}_full.p", "rb"))
    logo_data = [x[1] for x in logo_data] # x[0] is name of publisher
    
    # only the selected publisher
    logo_data = logo_data[publisher_index]
    
    if label_dict[model_index] != "0": # remove standard scalers
        logo_data = logo_data[0]

    run_pred(logo_data, magnitude_data, category_index, publisher_index, model_index, outdir, stan_file)

if __name__ == "__main__":
    main()
