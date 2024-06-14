import os, shutil
for dir1 in os.listdir("output/"):
    dir1_ = "output/"+dir1+"/"
    for dir2 in os.listdir(dir1_):
        dir2_ = dir1_ + dir2 + "/"
        for dir3 in os.listdir(dir2_):
            dir3_ = dir2_ + dir3 + "/logliks/"
            print(dir1, dir1_, dir2, dir2_, dir3, dir3_)
            try:
                shutil.copyfile(dir3_+ "logliks.npy", f"metrics/logliks/{dir1}_{dir2}_{dir3}.p")
                shutil.copyfile(dir3_+ "per_cascade_logliks.npy", f"metrics/per_cascade_logliks/{dir1}_{dir2}_{dir3}.p")
                shutil.copyfile(dir3_+ "per_event_logliks.npy", f"metrics/per_event_logliks/{dir1}_{dir2}_{dir3}.p")
                shutil.copyfile(dir3_+ "pred_av_absrelerr.npy", f"metrics/pred_av_absrelerr/{dir1}_{dir2}_{dir3}.p")
            except:
                continue