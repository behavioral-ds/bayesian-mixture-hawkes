from cmdstanpy import CmdStanModel

for stan_file in [
    "stan_files/theta-pl_0z-corr-threeclass-pl.stan",
    "stan_files/theta-pl_z0y-corr-threeclass-pl.stan",
    "stan_files/theta-pl_z0y_a0y-corr-threeclass-pl.stan",
    "stan_files/theta-pl_z0xy_a0y-corr-threeclass-pl.stan"
]:
    print(stan_file)
    model = CmdStanModel(stan_file=stan_file, cpp_options={ 'STAN_THREADS' : True }, compile='force')