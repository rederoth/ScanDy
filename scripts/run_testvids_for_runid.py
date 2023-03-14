import time
import numpy as np
import pandas as pd
from scipy import stats
import argparse
import os

# prepare logging
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
        "gt_foveation_df": "2021-12-04_VidCom_GT_fov_df",
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

    runid = "loc_train_molin_64-32-50_2023-03-09-01H-04M-33S_22332349"

    # obj_train_molin_64-32-50_2023-03-09-01H-04M-58S_22332350
    # obj_train_None_64-32-50_2023-03-09-01H-05M-43S_22332351
    # loc_train_molin_64-32-50_2023-03-09-01H-04M-33S_22332349
    # loc_train_TASEDnet_64-32-50_2023-03-09-01H-02M-20S_22332348

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

    for i in range(32):
        logging.info(
            f"{runid}\n############ {i} ############\nTime: {round((time.time()-starttime)/60,2)}min"
        )
        for par in parameters:
            model.params[par] = df_evol.sort_values("score", ascending=False).iloc[i][
                par
            ]

        if os.path.isfile(f"results/{runid}/testres_df_top{i}.csv") == False:
            model.run("test", seeds=[s for s in range(1, 13)], overwrite_old=True)
            model.evaluate_all_to_df()
            res_ratio = model.get_fovcat_ratio()
            # logging.info("GT" VidCom.get_fovcat_ratio(), "\nSIM", res_ratio)
            model.result_df.to_csv(f"results/{runid}/testres_df_top{i}.csv")
