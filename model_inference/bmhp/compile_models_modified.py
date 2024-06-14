from cmdstanpy import CmdStanModel

for stan_file in [
    "stan_files/mixture-pl_0+a0_z0x-sep_modified.stan",
    "stan_files/mixture-pl_0+a0y_z0x-sep_modified.stan",
    "stan_files/mixture-pl_0+a0_z0xy-sep_modified.stan",
    "stan_files/mixture-pl_0+a0y_z0xy-sep_modified.stan"
]:
    print(stan_file)
    model = CmdStanModel(stan_file=stan_file, cpp_options={ 'STAN_THREADS' : True }, compile='force')