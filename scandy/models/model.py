import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gif
from neurolib.utils.collections import dotdict
from abc import ABC, abstractmethod

from ..utils import functions as uf


class Model:
    """
    The Model class serves as base class for all scanpath models.

    It runs the model for a given set of videos and seeds, and stores the results in `result_dict`.
    The dictionary is then used to create a pandas dataframe, which is stored in the `result_df` attribute.
    The `result_df` is in the same format as the Dataset.gt_foveation_df, and can be used to evaluate the model.
    """

    def __init__(
        self, Dataset, params, preload_res_df=None  # integration, get_videodata,
    ):
        assert Dataset is not None, "No Dataset provided on which the model can run."
        self.Dataset = Dataset

        if hasattr(self, "name"):
            if self.name is not None:
                assert isinstance(self.name, str), "Model name is not a string."
        else:
            self.name = "Noname"

        # assert integration is not None, "Model integration function not given."
        # self.integration = integration

        # assert (
        #     get_videodata is not None
        # ), "No function for loading the data for each video."
        # self.get_videodata = get_videodata

        assert isinstance(params, dict), "Parameters must be a dictionary."
        self.params = params

        # # assert self.visual_vars not None:
        # assert hasattr(
        #     self, "visual_vars"
        # ), f"Model {self.name} has no attribute `visual_vars` (list of strings containing what to visualize)."
        # assert np.all(
        #     [type(s) is str for s in self.visual_vars]
        # ), "All entries in visual_vars must be strings."

        # video data will be loaded before running the model
        self.video_data = None

        # create output dictionary and result dataframe
        self.result_dict = {}
        if preload_res_df is None:
            self.result_df = pd.DataFrame()
        else:
            self.result_df = preload_res_df

    # @abstractmethod
    def load_videodata(self, videoname):
        """
        TODO: move (back?) to base class?!?!

        Provided the model parameters, load everything that is necessary to run
        the model for a given video.

        :param videoname: Single video for which the model should be run
        :type videoname: str
        :return: Parameter dictionary with all data neccessary to run the model for one video
        :rtype: dict
        """
        viddata = dotdict({})
        assert self.params is not None, "Model parameters not loaded"

        viddata.videoname = videoname
        # fetch info on featuretype and cb to load the feature map
        centerbias = self.params["centerbias"]
        featuretype = self.params["featuretype"]
        viddata.feature_maps = self.Dataset.load_featuremaps(
            videoname, featuretype, centerbias
        )
        if self.params["use_flow"]:
            viddata.flow_maps = self.Dataset.load_flowmaps(videoname)
        if self.params["use_objects"]:
            viddata.object_masks = self.Dataset.load_objectmasks(videoname)
        viddata.nframes = self.Dataset.video_frames[videoname]
        self.video_data = viddata  # save to class

    @abstractmethod
    def sgl_vid_run(self, viddata):
        pass

    def run(self, videos_to_run, seeds=[], overwrite_old=False):
        """
        Main interfacing function to run a model.
        :param videos_to_run: string, containing either a single video name, `test`, `train`, or `all`
        :param seeds: list of random seeds which will each result in a separate run/`trial` of the model (per video)
        """
        if (len(self.result_df.index) > 0) or (len(self.result_dict) > 0):
            assert (
                overwrite_old
            ), "There are already results stored, pass `overwrite_old=True` if you want to overwrite it"
            self.clear_model_outputs()
        # select videos to run, using keywords
        if videos_to_run == "all":
            videos = self.Dataset.used_videos
        elif videos_to_run == "train":
            assert hasattr(
                self.Dataset, "trainset"
            ), "Dataset needs an attribute called `trainingset`."
            videos = self.Dataset.trainset
        elif videos_to_run == "test":
            assert hasattr(
                self.Dataset, "testset"
            ), "Dataset needs an attribute called `testset`."
            videos = self.Dataset.testset
        elif videos_to_run in self.Dataset.used_videos:
            videos = [videos_to_run]
        else:
            raise Exception(
                f"The given `videos_to_run` is not a valid, must be a videoname in Dataset, `test`, `train`, or `all`."
            )

        assert len(seeds) > 0, f"Seeds given is {seeds}, must be list with len>0."
        assert len(videos) > 0, f"There are no videos in {videos_to_run}."

        # If only one video and one seed are given, return more details in the result_dict.
        # (This is first and foremost used for visualization purposes.)
        if len(videos) == len(seeds) == 1:
            self.params["sglrun_return"] = True
        else:
            self.params["sglrun_return"] = False

        # actually run the model for the given videos:
        for i, vid in enumerate(videos):
            video_res_dict = {}
            logging.info(
                f"Run video {i+1}/{len(videos)}: {vid} from videos_to_run {videos_to_run}..."
            )
            self.load_videodata(vid)
            for s in seeds:
                self.params["rs"] = s
                # Integration method from the specified model is used to write the result (dictionary) in result_dict
                video_res_dict[f"seed{s:03d}"] = self.sgl_vid_run(vid)
            self.result_dict[vid] = video_res_dict

        # check if there was a problem with the simulated data
        # self.checkOutputs()

    def save_model(self, filename):
        with open(f"{filename}.pkl", "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Model {self.name} is stored to {filename}.pkl")

    def clear_model_outputs(self):
        """Clears the model's results to create a fresh one"""
        self.result_dict = {}
        self.result_df = pd.DataFrame()

    #######
    ## Evaluation functions
    #######

    def select_videos(self, videos_to_eval="all"):
        """
        Function that selects the videos to be used for analysis.
        """
        if videos_to_eval == "all":
            videos = self.Dataset.used_videos
        elif videos_to_eval == "train":
            assert hasattr(
                self.Dataset, "trainset"
            ), "Dataset needs an attribute called `trainingset`."
            videos = self.Dataset.trainset
        elif videos_to_eval == "test":
            assert hasattr(
                self.Dataset, "testset"
            ), "Dataset needs an attribute called `testset`."
            videos = self.Dataset.testset
        elif videos_to_eval in self.Dataset.used_videos:
            videos = [videos_to_eval]
        else:
            raise Exception(
                f"The given `videos_to_eval` is not valid, must be a videoname in Dataset, `test`, `train`, or `all`."
            )
        return videos

    def evaluate_all_dur_amp(self):
        """
        Analog to evaluate_all_to_df, but only calculates foveation durations and saccade amplitudes.
        Hence, it is way more efficient and should be used when only these two statistics are of interest.
        """
        df_all = pd.DataFrame()
        for videoname in self.result_dict:
            for runname in self.result_dict[videoname]:
                run_dict = self.result_dict[videoname][runname]
                # if no saccades are made, diff will be empty and lead to an index error!
                if run_dict["f_sac"].size == 0:
                    # df = pd.DataFrame(columns=['dur_ms', 'amp_dva']) --> empty return will lead to error later!
                    fov_ends = np.array([0, self.Dataset.video_frames[videoname] - 1])
                else:
                    fov_ends = np.append(
                        run_dict["f_sac"], self.Dataset.video_frames[videoname] - 1
                    )

                df = pd.DataFrame()
                df["frame_start"] = [0] + [f + 1 for f in fov_ends[:-1]]
                df["frame_end"] = fov_ends
                df["dur_ms"] = (
                    (1 + df["frame_end"] - df["frame_start"])
                    * 1000.0
                    / self.Dataset.FPS
                )
                diff = np.array(
                    [
                        run_dict["gaze"][f + 1] - run_dict["gaze"][f]
                        for f in df["frame_end"][:-1]
                    ]
                )
                df["amp_dva"] = [np.nan] + list(
                    np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2) * self.Dataset.PX_TO_DVA
                )
                df_all = df_all.append(df)

        all_durations = df_all["dur_ms"].dropna().values
        all_amplitudes = df_all["amp_dva"].dropna().values

        return all_durations, all_amplitudes

    def evaluate_trial(self, videoname, runname, segmentation_masks=None):
        """
        Main evaluation function that returns a dataframe with all relevant foveation (and saccade) statistics.
        """
        assert (
            self.result_dict
        ), "`result_dict` is empty. You need to run the model before evaluating the results."
        # Runs in key error if they don't exist, maybe assert?
        run_dict = self.result_dict[videoname][runname]
        assert {"gaze", "f_sac"} <= set(
            run_dict
        ), "Integration method did not provide `gaze` and `f_sac`"
        df = pd.DataFrame()

        # option to pass it so it doesn't have to be loaded each time in the loop
        if segmentation_masks is None:
            segmentation_masks = self.Dataset.get_objectmasks(videoname)
        # get all foveated objects! should maybe be a seperate method?
        objects_per_frame = [
            uf.object_at_position(
                segmentation_masks[f],
                run_dict["gaze"][f][1],
                run_dict["gaze"][f][0],
                radius=self.Dataset.RADIUS_OBJ_GAZE,
            )
            for f in range(self.Dataset.video_frames[videoname])
        ]

        # no saccade in last frame possible, since loop runs only in range(frames-1)
        fov_ends = np.append(
            run_dict["f_sac"], self.Dataset.video_frames[videoname] - 1
        ).astype(int)
        N_fov = len(fov_ends)
        # dataframe has a row for each foveation
        df = pd.DataFrame()
        # this column allows to ignore_index=True without losing individual indexing per run
        df["nfov"] = [int(i) for i in range(N_fov)]
        # same for all foveations in this run
        df["video"] = videoname
        df["subject"] = runname
        df["frame_start"] = [0] + [f + 1 for f in fov_ends[:-1]]
        df["frame_end"] = fov_ends
        # add a `1+` such that even a single frame fov has a duration --> sums to duration!
        df["duration_ms"] = (
            (1 + df["frame_end"] - df["frame_start"]) * 1000.0 / self.Dataset.FPS
        )
        df["x_start"] = [run_dict["gaze"][f][1] for f in df["frame_start"]]
        df["y_start"] = [run_dict["gaze"][f][0] for f in df["frame_start"]]
        df["x_end"] = [run_dict["gaze"][f][1] for f in df["frame_end"]]
        df["y_end"] = [run_dict["gaze"][f][0] for f in df["frame_end"]]
        # check most common object between start and end frame
        df["object"] = [
            Counter(
                ", ".join(
                    objects_per_frame[
                        df["frame_start"].iloc[n] : df["frame_end"].iloc[n] + 1
                    ]
                ).split(", ")
            ).most_common(1)[0][0]
            for n in range(N_fov)
        ]

        # saccade properties depend on the end of the current fov and beginning of next one
        # TODO: check if diff has correct sign (otherwise angle is different...)
        diff = np.array(
            [
                run_dict["gaze"][f + 1] - run_dict["gaze"][f]
                for f in df["frame_end"][:-1]
            ]
        )
        # aviod error if no saccades are made
        if diff.size:
            # first foveation is excluded since no saccade preceedes it!
            df["sac_amp_dva"] = [np.nan] + list(
                np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2) * self.Dataset.PX_TO_DVA
            )
            df["sac_angle_h"] = [np.nan] + list(
                np.arctan2(diff[:, 0], diff[:, 1]) / np.pi * 180
            )
            # second entry of angle_p will also be nan since first angle_h is nan
            df["sac_angle_p"] = [np.nan] + [
                uf.angle_limits(df["sac_angle_h"][i + 1] - df["sac_angle_h"][i])
                for i in range(N_fov - 1)
            ]
        else:
            df["sac_amp_dva"] = [np.nan]
            df["sac_angle_h"] = [np.nan]
            df["sac_angle_p"] = [np.nan]

        fov_categories = []
        ret_times = np.zeros(N_fov) * np.nan
        for n in range(N_fov):
            obj = df["object"].iloc[n]
            if obj in ["Ground", ""]:
                fov_categories.append("B")
            elif (n > 0) and (df["object"].iloc[n - 1] == obj):
                fov_categories.append("I")
            else:
                prev_obj = df["object"].iloc[:n]
                if obj not in prev_obj.values:
                    fov_categories.append("D")
                else:
                    fov_categories.append("R")
                    return_frame = df["frame_end"][
                        prev_obj.where(prev_obj == obj).last_valid_index()
                    ]
                    # store time difference [in milliseconds] in array!
                    ret_times[n] = (
                        (df["frame_start"].iloc[n] - return_frame)
                        * 1000
                        / self.Dataset.FPS
                    )
        df["fov_category"] = fov_categories
        df["ret_times"] = ret_times
        # self.result_df = self.result_df.append(df), RETURN INSTEAD!
        return df

    def evaluate_all_to_df(self, overwrite_old=False):
        """
        Function that evaluates all trials and stores them to self.result_df.
        """
        assert (
            self.result_dict
        ), "`result_dict` is empty. You need to run the model before evaluating the results."
        if len(self.result_df.index) > 0:
            assert (
                overwrite_old
            ), "`result_df` is already filled, pass `overwrite_old=True` if you want to overwrite it"
            self.result_df = pd.DataFrame()
        for videoname in self.result_dict:
            # load masks outside of loop to be a bit more efficient
            segmentation_masks = self.Dataset.get_objectmasks(videoname)
            for runname in self.result_dict[videoname]:
                df_trial = self.evaluate_trial(videoname, runname, segmentation_masks)
                self.result_df = self.result_df.append(df_trial, ignore_index=True)

        return self.result_df

    def get_fovcat_ratio(self, videos_to_eval="all"):
        """
        Convenience function that returns the ratios as dictionary for the different categories.
        """
        videos = self.select_videos(videos_to_eval)
        eval_df = self.result_df[self.result_df["video"].isin(videos)]

        assert (
            len(eval_df) > 0
        ), f"`result_df` is empty for {videos_to_eval}, make sure to run `evaluate_all_to_df` first!"
        categories = ["B", "D", "I", "R"]
        ratios = {}
        full_dur = np.sum(eval_df.duration_ms)
        for cat in categories:
            ratio = (
                np.sum(eval_df[eval_df["fov_category"] == cat].duration_ms) / full_dur
            )
            ratios[cat] = ratio
        return ratios

    def functional_event_courses(self, videos_to_eval="all"):
        # for simulated data, the dimension is (#videos*#seeds, #frames)
        videos = self.select_videos(videos_to_eval)
        eval_df = self.result_df[self.result_df["video"].isin(videos)]
        n_scanpaths = len(videos) * len(set(eval_df.subject))
        eventcourse = np.chararray((n_scanpaths, self.Dataset.FRAMES_ALL_VIDS + 1))

        subcount = 0
        for vid in sorted(videos):
            seeds = sorted(set(eval_df[eval_df["video"] == vid].subject))
            for run in seeds:
                temp = eval_df[(eval_df["video"] == vid) & (eval_df["subject"] == run)]
                for index in temp.index:
                    f_start = temp.loc[index].frame_start
                    f_end = temp.loc[index].frame_end
                    # always add 1 to f_end to include the ending frame
                    eventcourse[subcount, f_start : f_end + 1] = temp.loc[
                        index
                    ].fov_category
                subcount += 1

        ground_f = np.zeros(self.Dataset.FRAMES_ALL_VIDS)
        inspection_f = np.zeros(self.Dataset.FRAMES_ALL_VIDS)
        detection_f = np.zeros(self.Dataset.FRAMES_ALL_VIDS)
        revisits_f = np.zeros(self.Dataset.FRAMES_ALL_VIDS)

        for f in range(self.Dataset.FRAMES_ALL_VIDS):
            array = eventcourse[:, f]
            uniques, counts = np.unique(array, return_counts=True)
            percentages = dict(zip(uniques, counts * 100 / len(array)))
            for key, value in percentages.items():
                if key == b"B":
                    ground_f[f] = value
                elif key == b"D":
                    detection_f[f] = value
                elif key == b"I":
                    inspection_f[f] = value
                elif key == b"R":
                    revisits_f[f] = value
        return [ground_f, detection_f, inspection_f, revisits_f]

    def evaluate_nss_scores(self):
        """
        Calculates the NSS score of the model for each video, averaged across time.
        Returns the mean across seeds of the NSS scores and its std.
        """
        assert (
            self.result_dict
        ), "`result_dict` is empty. You need to run the model before evaluating the results."
        nss_with_std = {}
        # go through all videos (keys) and load the GT
        for videoname in sorted(self.result_dict):
            # load masks outside of loop to be a bit more efficient
            gt_fovmaps = self.Dataset.get_nssmaps(videoname)
            nframes = gt_fovmaps.shape[0]
            vid_nss_scores = []
            for runname in self.result_dict[videoname]:
                run_dict = self.result_dict[videoname][runname]
                nss_frames = [
                    gt_fovmaps[f, run_dict["gaze"][f][0], run_dict["gaze"][f][0]]
                    for f in range(nframes)
                ]
                vid_nss_scores.append(np.mean(nss_frames))
            nss_with_std[videoname] = [np.mean(vid_nss_scores), np.std(vid_nss_scores)]

        return nss_with_std

    ######
    ## Evaluation visualization functions...
    ######

    def show_result_statistics(self, with_fovcat=True, title=True):
        """
        Show the summary statistics in fovdur, saccamp,
        TODO: throws error when category for one video is empty
        """
        assert (
            len(self.result_df) > 0
        ), "`result_df` is empty, make sure to run `evaluate_all_to_df` first!"
        if with_fovcat:
            hue = "fov_category"
            hue_order = ["B", "D", "I", "R"]
        else:
            hue = None
            hue_order = None
        # for the generation of titles
        dataname = ["GT", "SIM"]
        # log of foveation duration MEAN FOV DURATION AND MEDIAN
        fig1, ax1 = plt.subplots(1, 2, dpi=150, figsize=(10, 3))
        if not "log_duration" in self.Dataset.gt_foveation_df.columns:
            self.Dataset.gt_foveation_df["log_duration"] = np.log10(
                self.Dataset.gt_foveation_df["duration_ms"]
            )
        if not "log_duration" in self.result_df.columns:
            self.result_df["log_duration"] = np.log10(self.result_df["duration_ms"])
        sns.histplot(
            data=self.Dataset.gt_foveation_df,
            x="log_duration",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax1[0],
            multiple="dodge",
        )  # , bins=20)
        sns.histplot(
            data=self.result_df,
            x="log_duration",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax1[1],
            multiple="dodge",
        )  # , bins=20)
        if title:
            mean = [
                np.rint(np.mean(self.Dataset.gt_foveation_df["duration_ms"])),
                np.rint(np.mean(self.result_df["duration_ms"])),
            ]
            median = [
                np.rint(np.median(self.Dataset.gt_foveation_df["duration_ms"])),
                np.rint(np.median(self.result_df["duration_ms"])),
            ]
        for i in [0, 1]:
            if title:
                ax1[i].set_title(
                    f"{dataname[i]} mean dur: {mean[i]}ms (median: {median[i]}ms)"
                )
            ax1[i].set_xticks([1, 2, 3, 4])
            ax1[i].set_xticklabels([10, 100, 1000, 10000])
            ax1[i].set_xlabel("Foveation duration [ms]")

        # saccade amplitude MEAN AND MEDIAN SACC AMP
        fig2, ax2 = plt.subplots(1, 2, dpi=150, figsize=(10, 3))
        sns.histplot(
            data=self.Dataset.gt_foveation_df,
            x="sac_amp_dva",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax2[0],
            multiple="dodge",
        )  # , bins=20)
        sns.histplot(
            data=self.result_df,
            x="sac_amp_dva",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax2[1],
            multiple="dodge",
        )  # , bins=20)
        if title:
            mean = [
                np.round(np.mean(self.Dataset.gt_foveation_df["sac_amp_dva"]), 2),
                np.round(np.mean(self.result_df["sac_amp_dva"]), 2),
            ]
            median = [
                np.round(np.nanmedian(self.Dataset.gt_foveation_df["sac_amp_dva"]), 2),
                np.round(np.nanmedian(self.result_df["sac_amp_dva"]), 2),
            ]
        for i in [0, 1]:
            if title:
                ax2[i].set_title(
                    f"{dataname[i]} mean amp: {mean[i]}DVA (median: {median[i]}DVA)"
                )
            ax2[i].set_xlabel("Saccade amplitude [DVA]")
            ax2[i].set_xlim([0, 50])

        # saccade angle to horizontal RATIO OF I SAC
        fig3, ax3 = plt.subplots(1, 2, dpi=150, figsize=(10, 3))
        sns.histplot(
            data=self.Dataset.gt_foveation_df,
            x="sac_angle_h",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax3[0],
            multiple="dodge",
        )  # , bins=20)
        sns.histplot(
            data=self.result_df,
            x="sac_angle_h",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax3[1],
            multiple="dodge",
        )  # , bins=20)
        if title:
            ratio = [
                np.round(
                    100
                    * len(
                        self.Dataset.gt_foveation_df[
                            self.Dataset.gt_foveation_df["fov_category"] == "I"
                        ]
                    )
                    / len(self.Dataset.gt_foveation_df),
                    2,
                ),
                np.round(
                    100
                    * len(self.result_df[self.result_df["fov_category"] == "I"])
                    / len(self.result_df),
                    2,
                ),
            ]
        for i in [0, 1]:
            if title:
                ax3[i].set_title(
                    f"{dataname[i]} ratio of Inspection events: {ratio[i]}%"
                )
            ax3[i].set_xlabel("Saccade angle to horizontal [degree]")

        # saccade angle relative to previous saccade  RATIO OF B SAC
        fig4, ax4 = plt.subplots(1, 2, dpi=150, figsize=(10, 3))
        sns.histplot(
            data=self.Dataset.gt_foveation_df,
            x="sac_angle_p",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax4[0],
            multiple="dodge",
        )  # , bins=20)
        sns.histplot(
            data=self.result_df,
            x="sac_angle_p",
            hue=hue,
            hue_order=hue_order,
            kde=True,
            ax=ax4[1],
            multiple="dodge",
        )  # , bins=20)
        if title:
            ratio = [
                np.round(
                    100
                    * len(
                        self.Dataset.gt_foveation_df[
                            self.Dataset.gt_foveation_df["fov_category"] == "B"
                        ]
                    )
                    / len(self.Dataset.gt_foveation_df),
                    2,
                ),
                np.round(
                    100
                    * len(self.result_df[self.result_df["fov_category"] == "B"])
                    / len(self.result_df),
                    2,
                ),
            ]
        for i in [0, 1]:
            if title:
                ax4[i].set_title(
                    f"{dataname[i]} ratio of Background events: {ratio[i]}%"
                )
            ax4[i].set_xlabel("Saccade turning angle [degree]")

        # returntimes RATIO OF R SAC
        fig5, ax5 = plt.subplots(1, 2, dpi=150, figsize=(10, 3))
        sns.histplot(
            self.Dataset.gt_foveation_df.ret_times, kde=True, ax=ax5[0]
        )  # , bins=100)
        sns.histplot(self.result_df.ret_times, kde=True, ax=ax5[1])  # , bins=100)
        if title:
            ratioR = [
                np.round(
                    100
                    * len(
                        self.Dataset.gt_foveation_df[
                            self.Dataset.gt_foveation_df["fov_category"] == "R"
                        ]
                    )
                    / len(self.Dataset.gt_foveation_df),
                    2,
                ),
                np.round(
                    100
                    * len(self.result_df[self.result_df["fov_category"] == "R"])
                    / len(self.result_df),
                    2,
                ),
            ]
            ratioD = [
                np.round(
                    100
                    * len(
                        self.Dataset.gt_foveation_df[
                            self.Dataset.gt_foveation_df["fov_category"] == "D"
                        ]
                    )
                    / len(self.Dataset.gt_foveation_df),
                    2,
                ),
                np.round(
                    100
                    * len(self.result_df[self.result_df["fov_category"] == "D"])
                    / len(self.result_df),
                    2,
                ),
            ]
        for i in [0, 1]:
            if title:
                ax5[i].set_title(
                    f"{dataname[i]} ratio Revisits: {ratioR[i]}%, Detections: {ratioD[i]}%"
                )
            ax5[i].set_xlabel("Return time [ms]")

        plt.show()

    def show_fovdur_pervid(self):
        """
        TODO: Maybe kick this? can be very missleading...
        Function that plots the mean foveation duration per video (and the overall mean as line) for
        the simulated (x, --) and the ground truth (o, :) data.
        """
        assert (
            len(self.result_df) > 0
        ), "`result_df` is empty, make sure to run `evaluate_all_to_df` first!"
        videos = self.result_df.video.unique()
        categories = ["B", "D", "I", "R"]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(dpi=150)
        fovdurs_pervid_BDIR_sim = []
        fovdurs_pervid_BDIR_gt = []
        fovdurs_means_BDIR_sim = []
        fovdurs_means_BDIR_gt = []

        df_gt = self.Dataset.gt_foveation_df
        for cat in categories:
            fovdurs_pervid_BDIR_gt.append(
                [
                    np.mean(
                        df_gt.duration_ms[
                            (df_gt.video == vid) & (df_gt.fov_category == cat)
                        ].values
                    )
                    for vid in videos
                ]
            )
            fovdurs_means_BDIR_gt.append(
                np.mean(df_gt.duration_ms[df_gt.fov_category == cat].values)
            )
            fovdurs_pervid_BDIR_sim.append(
                [
                    np.mean(
                        self.result_df.duration_ms[
                            (self.result_df.video == vid)
                            & (self.result_df.fov_category == cat)
                        ].values
                    )
                    for vid in videos
                ]
            )
            fovdurs_means_BDIR_sim.append(
                np.mean(
                    self.result_df.duration_ms[
                        self.result_df.fov_category == cat
                    ].values
                )
            )

        overall_mean_gt = [
            np.mean(df_gt.duration_ms[df_gt.video == vid].values) for vid in videos
        ]
        overall_mean_sim = [
            np.mean(self.result_df.duration_ms[self.result_df.video == vid].values)
            for vid in videos
        ]

        for i, cat in enumerate(categories):
            ax.axhline(fovdurs_means_BDIR_gt[i], ls=":", color=colors[i], alpha=0.5)
            ax.axhline(fovdurs_means_BDIR_sim[i], ls="--", color=colors[i])
            ax.plot(
                fovdurs_pervid_BDIR_gt[i], ls="", marker="o", color=colors[i], alpha=0.5
            )
            ax.plot(
                fovdurs_pervid_BDIR_sim[i],
                ls="",
                marker="x",
                color=colors[i],
                label=cat,
            )
        ax.plot(overall_mean_sim, ls="", marker="x", color="k", label=r"$\mu$")
        ax.plot(overall_mean_gt, ls="", marker="o", color="k", alpha=0.5)
        ax.set_xticks(range(len(videos)))
        ax.set_xticklabels(videos, rotation=45, ha="right", rotation_mode="anchor")
        plt.legend()
        plt.show()

    def video_output_gif(
        self, videoname, storagename, interpolate=False, slowgif=False
    ):
        """
        General function that takes all the predicted scanpaths and plots it on top
        of the original video. No further details are visualized, hence it is model agnostic.
        """
        if hasattr(self.Dataset, "outputpath"):
            outputpath = self.Dataset.outputpath + storagename
        else:
            hasattr(self.Dataset, "PATH"), "Dataset has no defined PATH"
            outputpath = f"{self.Dataset.PATH}results/{storagename}"

        assert (
            videoname in self.result_dict
        ), f"No simulated scanpaths for {videoname} yet, first run the model!"
        res_dict = self.result_dict[videoname]
        vidlist = self.Dataset.get_videoframes(videoname)

        @gif.frame
        def frame(f):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(vidlist[f])
            # ax.imshow(gt_map[f,:,:], vmax=vmax_gt, cmap='gray', alpha=0.6)
            for key in res_dict.keys():
                ax.scatter(
                    res_dict[key]["gaze"][f][1],
                    res_dict[key]["gaze"][f][0],
                    s=150,
                    alpha=0.8,
                )
                if f > 0:
                    ax.plot(
                        [res_dict[key]["gaze"][f - 1][1], res_dict[key]["gaze"][f][1]],
                        [res_dict[key]["gaze"][f - 1][0], res_dict[key]["gaze"][f][0]],
                        linewidth=7,
                        ls=":",
                    )
            # ax.set_title(f"Ground truth vs prediction for DV $ \theta={dvthres}, \sigma={dvnoise}$, {name}, Frame: {f}")
            ax.set_axis_off()

        out = [frame(i) for i in range(len(vidlist))]
        if slowgif:
            gif.save(out, outputpath + "_slow.gif", duration=100)
            print(f"Saved to {outputpath}_slow.gif")
            # gif.save(out, f"videos/objvideo_{name}_{vidname}_thres_dv{thres_dv}_sig_dv{sig_dv}_ior_decay{ior_decay}_att_obj{att_obj}_att_dva{att_dva}_rs{rs}_slow.gif", duration=100)
        else:
            gif.save(out, outputpath + ".gif", duration=33)
            print(f"Saved to {outputpath}.gif")
            # gif.save(out, f"videos/objvideo_{name}_{vidname}_thres_dv{thres_dv}_sig_dv{sig_dv}_ior_decay{ior_decay}_att_obj{att_obj}_att_dva{att_dva}_rs{rs}.gif", duration=33)

        # raise Exception("Not yet implemented!")

    #######################################################################
    ##########                NOT YET IMPLEMENTED                ##########
    #######################################################################

    def store_model_with_results(self, storagename):
        """TODO: Method that stores the model with parameters and simulation results
        Right now, say explicitely what should be stored in pypet!
        """
        if hasattr(self.Dataset, "outputpath"):
            outputpath = self.Dataset.outputpath + storagename
        else:
            assert hasattr(self.Dataset, "PATH"), "Dataset has no defined PATH"
            outputpath = f"{self.Dataset.PATH}results/{storagename}"
        raise Exception("Not yet implemented!")
