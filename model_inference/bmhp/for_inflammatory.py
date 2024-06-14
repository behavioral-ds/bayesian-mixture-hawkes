from cmdstanpy import from_csv
import pandas as pd
import os

to_iterate = os.listdir("output/1/")

for t in to_iterate:
    a = from_csv(f"output/1/{t}/3")
    df = a.draws_pd()
    cols = [x for x in df.columns if 'gamma' in x and '[' in x and 'raw' not in x]
    
    relevant = df[cols]

    relevant.to_csv(f"for_inflammatory/{t}.csv", index=False)
