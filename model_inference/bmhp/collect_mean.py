import os, shutil
import numpy as np
for dir1 in os.listdir("output/"):
    dir1_ = "output/"+dir1+"/"
    for dir2 in os.listdir(dir1_):
        dir2_ = dir1_ + dir2 + "/"
        for dir3 in os.listdir(dir2_):
            dir3_ = dir2_ + dir3 + "/logliks/"
            print(dir1, dir1_, dir2, dir2_, dir3, dir3_)

            try:
                np.save(f"metrics_mean/logliks/{dir1}_{dir2}_{dir3}", np.load(dir3_+ "logliks.npy")[:1000,:].mean(axis=0))
                np.save(f"metrics_mean/ares/{dir1}_{dir2}_{dir3}", np.load(dir3_ + "ares.npy"))
            except:
                continue
