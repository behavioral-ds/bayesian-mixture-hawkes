import pickle, os, shutil, sys
import pandas as pd
import numpy as np
import logging
from cmdstanpy import CmdStanModel
from time import perf_counter
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

# input: 0 1 4 <- category_index, publisher_index, model_index

max_workers = 4 # each worker = 4 cores (= number of chains)

exe_dict = {
    0: "stan_files/theta-pl_0z-corr-threeclass-pl",
    1: "stan_files/theta-pl_z0y-corr-threeclass-pl",
    2: "stan_files/theta-pl_z0y_a0y-corr-threeclass-pl",
    3: "stan_files/theta-pl_z0xy_a0y-corr-threeclass-pl"
}

# code for which data to use, input is model index
label_dict = {
    0: "0",
    1: "2",
    2: "2",
    3: "2"
}

# code for category index
category_dict = {
    0: "ATNIX",
    1: "FAKENIX"
}


####### Function to run fit on fold #######
def run_stan_fit(data, category_index, publisher_index, model_index, outdir, logliks_dir, stan_file):
    logging.basicConfig(filename=f"log/{category_index}_{publisher_index}_{model_index}.log", level=logging.INFO, format='%(asctime)s | %(message)s', force=True)

    logging.info(f"processing data for category {category_index}; publisher {publisher_index}; model {model_index}")    

    data["d_train"] = [x/60/60 for x in data["d_train"]] # convert to hours
    data["d_holdout"] = [x/60/60 for x in data["d_holdout"]] # convert to hours

    model = CmdStanModel(exe_file=stan_file)
    stan_fit = model.sample(data=data, 
                            chains=4,
                            parallel_chains=4,
                            threads_per_chain=1, # on heph, unfortunately 1 thread/core
                            adapt_delta=0.9,
                            iter_warmup=500,
                            iter_sampling=500
                           )    
    logging.info(f"finished sampling. saving now.")    

    stan_fit.save_csvfiles(outdir)
    
    stan_draws = stan_fit.draws_pd()
        
    # # total loglikelihood per video
    # loglik_columns = [x for x in stan_draws.columns if x.startswith("log_lik[")]
    # np.save(logliks_dir+"logliks", stan_draws[loglik_columns].values)
    
    # total loglikelihood per cascade
    loglik_columns = [x for x in stan_draws.columns if x.startswith("log_lik_per_cascade[")]
    np.save(logliks_dir+"loglikscascade", stan_draws[loglik_columns].values)

    logging.info(f"done.")    
    pickle.dump(stan_fit.diagnose(), open(logliks_dir+"diagnose.p", "wb"))

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
    #######
    # logo_data = pickle.load(open(f"/data/pbcalder/rnixcnix-bayes/stan/THETA_{label_dict[model_index]}_{category_dict[category_index]}_minE10_maxE50_maxC100_full.p", "rb"))
    logo_data = pickle.load(open(f"data/THETA_{label_dict[model_index]}_FAKENIX_split_{publisher_index}.p", "rb"))
        
    logo_data = logo_data[1]
    
    if label_dict[model_index] != "0": # remove standard scalers
        logo_data = logo_data[0]

    stan_file = exe_dict[model_index]
    
    outdir = f"output/{category_index}/{publisher_index}/{model_index}/" # temporary location, will transfer after processing
    logliks_dir = outdir+"logliks/"
    
    try:
        os.makedirs(logliks_dir) # recursively make directories for category -> publisher -> model -> logliks
    except:
        pass
    
    #######
    
    run_stan_fit(logo_data, category_index, publisher_index, model_index, outdir, logliks_dir, stan_file)
                         
        
if __name__ == "__main__":
    main()
