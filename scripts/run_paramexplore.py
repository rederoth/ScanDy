import time
import numpy as np
import pandas as pd
from scipy import stats
import argparse
import os

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution

from scandy.models.ObjectModel import ObjectModel
from scandy.models.LocationModel import LocationModel
from scandy.utils.dataclass import Dataset
import scandy.utils.functions as uf


if __name__ == "__main__":

    starttime = time.time()

    # load VidCom dataset
    datadict = {
        # 'PATH' : '/mnt/raid/data/SCIoI/USCiLab/VidCom/',
        "PATH": "/scratch/nroth/VidCom/VidCom/",  # HPC
        "FPS": 30,
        "PX_TO_DVA": 0.06,
        "FRAMES_ALL_VIDS": 300,
        "gt_foveation_df": "VidCom_GT_fov_df.csv",
        "gt_fovframes_nss_df": "gt_fovframes_nss_df.csv",
        "trainset": [
            "dance01",
            "dance02",
            "garden06",
            "garden07",
            "park01",
            "park06",
            "road02",
            "road05",
            "room01",
            "walkway03",
        ],
        "testset": [
            "field03",
            "foutain02",
            "garden04",
            "garden09",
            "park09",
            "road04",
            "robarm01",
            "room02",
            "room03",
            "tommy02",
            "uscdog01",
            "walkway01",
            "walkway02",
        ],
    }
    VidCom = Dataset(datadict)

    runid = "obj_train_molin_64-32-50_2023-08-01-10H-44M-52S_22770892"
    #"loc_train_molin_64-32-50_2023-08-01-10H-46M-11S_22770898"
    # "obj_train_molin_64-32-50_2023-08-01-10H-44M-52S_22770892" 

    DILLNAME = f"{runid}.dill"
    evol = Evolution(lambda x: x, ParameterSpace(["mock"], [[0, 1]]))
    evol = evol.loadEvolution(f"results/{runid}/{DILLNAME}")
    df_evol = evol.dfEvolution(outputs=True).copy()
    df_top32 = df_evol.sort_values("score", ascending=False)[:32]

    if "obj_" in runid:
        model = ObjectModel(VidCom)
        parameters = ["ddm_thres", "ddm_sig", "att_dva", "ior_decay", "ior_inobj"]
    elif "loc_" in runid:
        model = LocationModel(VidCom)
        parameters = ["ddm_thres", "ddm_sig", "att_dva", "ior_decay", "ior_dva"]
    else:
        raise NotImplementedError("Only Object- and Location-based are implemented")

    if "molin" in runid:
        model.params["featuretype"] = "molin"
    elif "None" in runid:
        model.params["featuretype"] = "None"
    elif "TASEDnet" in runid:
        model.params["featuretype"] = "TASEDnet"
    else:
        raise NotImplementedError("Only molin, None, or TASEDnet are implemented")

    mean_pars = {}
    for par in parameters:
        mean_pars[par] = np.mean(df_top32[par])
        model.params[par] = mean_pars[par]
    print(mean_pars)

    # run once the mean parameters
    name = "res_df_mean"
    model.run("test", seeds=[s for s in range(1, 13)])
    model.evaluate_all_to_df()
    model.result_df.to_csv(f"results/{runid}/{name}.csv")

    # run for variations in certain factors
    relative_par_vals = [0.5, 0.75, 0.9, 1.1, 1.25, 2]
    for var_par in parameters:
        for factor in relative_par_vals:
            name = f"res_df_mean_{var_par}_{factor}"
            print(name)
            if os.path.isfile(f"results/{runid}/{name}.csv") == False:
                for par in parameters:
                    model.params[par] = mean_pars[par]
                # set var_par to ratio of fitted mean
                model.params[var_par] = factor * mean_pars[var_par]
                print(model.params)
                model.run("test", seeds=[s for s in range(1, 13)], overwrite_old=True)
                model.evaluate_all_to_df()
                model.result_df.to_csv(f"results/{runid}/{name}.csv")
            else:
                print("Already run!")
