import pickle, os, shutil, sys
import pandas as pd
import numpy as np
import logging
from cmdstanpy import CmdStanModel
from time import perf_counter
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import gaussian_kde

# input: 0 1 4 <- category_index, publisher_index, model_index

max_workers = 4 # each worker = 4 cores (= number of chains)

exe_dict = {
    0: "stan_files/mixture-pl_0+a0_z0x-sep",
    1: "stan_files/mixture-pl_0+a0y_z0x-sep",
    2: "stan_files/mixture-pl_0+a0_z0xy-sep",
    3: "stan_files/mixture-pl_0+a0y_z0xy-sep"
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


def estimate_MAP(data):
    kde = gaussian_kde(data)
    no_samples = 100 # number of interpolation points
    samples = np.linspace(min(data), max(data), no_samples)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]

    return maxima

####### Function to run fit on fold #######
def run_stan_fit(data, category_index, publisher_index, model_index, outdir, logliks_dir, stan_file):
    logging.basicConfig(filename=f"log/{category_index}_{publisher_index}_{model_index}.log", level=logging.INFO, format='%(asctime)s | %(message)s', force=True)

    logging.info(f"processing data for category {category_index}; publisher {publisher_index}; model {model_index}")    
    
    model = CmdStanModel(exe_file=stan_file)
    stan_fit = model.sample(data=data, 
                            chains=4,
                            parallel_chains=4,
                            threads_per_chain=1, # on heph, unfortunately 1 thread/core
                            adapt_delta=0.9,
                           )    
    logging.info(f"finished sampling. saving now.")    

    stan_fit.save_csvfiles(outdir)
    
    stan_draws = stan_fit.draws_pd()
    
    # cascade size prediction per cascade
    actual_cascadesizes = np.array(data['N_cascadesize_per_cascade_holdout'])
    
    pred_draws = stan_draws[[x for x in stan_draws.columns if 'pred_cascadesize_holdout' in x]].iloc[:1000,:]
    
    pred_cascadesizes_mean = pred_draws.mean(axis=0).values
    pred_cascadesizes_median = pred_draws.median(axis=0).values
    
    pred_cascadesizes_mode = []
    for col in pred_draws.columns:
        pred_cascadesizes_mode.append(estimate_MAP(pred_draws[col]))
    pred_cascadesizes_mode = np.array(pred_cascadesizes_mode)

    ares = np.stack([
        actual_cascadesizes, 
        pred_cascadesizes_mean,
        pred_cascadesizes_median,
        pred_cascadesizes_mode
    ])
    np.save(logliks_dir+"ares", ares)
    
    # total loglikelihood per video
    loglik_columns = [x for x in stan_draws.columns if x.startswith("log_lik[")]
    np.save(logliks_dir+"logliks", stan_draws[loglik_columns].values)

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
    # logo_data = pickle.load(open(f"/data/pbcalder/rnixcnix-bayes/stan/LOGO_{label_dict[model_index]}_{category_dict[category_index]}_full.p", "rb"))
    logo_data = pickle.load(open(f"data/LOGO_{label_dict[model_index]}_{category_dict[category_index]}_full.p", "rb"))
    logo_data = [x[1] for x in logo_data] # x[0] is name of publisher
        
    # only the selected publisher
    logo_data = logo_data[publisher_index]
    
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
                         
        
        
#     ####### Collect log-likelihoods over all folds #######
#     log_liks = []
#     for i in range(len(logo_data)):
#         log_liks.append(np.load(logliks_dir+str(i).zfill(2)+".npy"))
#     logo_likelihood = pd.DataFrame(np.stack(np.array(log_liks)).T)

#     # calc: see https://mc-stan.org/docs/stan-users-guide/log-sum-of-exponentials.html
#     max_vals = logo_likelihood.max(axis=0)
#     ptwise_likelihood = max_vals + ((logo_likelihood - max_vals).applymap(np.exp)).sum(axis=0).map(np.log)
#     lpd = np.sum(ptwise_likelihood)
#     se = np.sqrt(len(ptwise_likelihood)) * np.std(ptwise_likelihood)

#     pickle.dump([
#         fit_duration,
#         logo_likelihood,
#         ptwise_likelihood,
#         lpd,
#         se
#     ], open(f"{outdir}"+"loglik_collector.p", "wb"))
#     #######

if __name__ == "__main__":
    main()
