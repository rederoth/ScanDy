import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import gif
from neurolib.utils.collections import dotdict
from abc import abstractmethod

from ..utils import functions as uf
from .objectfile import ObjectFile


class Model:
    """
    The Model class serves as base class for all scanpath models.

    Can be run for a given set of videos and seeds and stores the results in `result_dict`.
    The dictionary with a scanpath for each seed and video can then be evaluated
    analogously to human eye tracking data. is then used to create a pandas dataframe, which is stored in the `result_df` attribute.
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

        assert isinstance(params, dict), "Parameters must be a dictionary."
        self.params = params

        # video data will be loaded before running the model
        # self.video_data = None

        # Attributes that are updated when loading a video
        self.video_data = {}
        self._dt = 1000.0 / self.Dataset.FPS  # in ms, TODO for each video differently?
        self._t = 0.0
        # Attributes that are updated when running the model for a single video
        self._scanpath = []
        self._f_sac = []
        self._gaze_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        self._new_target = None
        self._prev_gaze_loc = np.zeros(2, int)  # self.params["startpos"].copy()
        # new due to saccades taking up time
        self._current_frame = 0
        self._cur_waiting_time = 0.0
        self._cur_fov_frac = 1.0

        # create output dictionary and result dataframe
        self.result_dict = {}
        if preload_res_df is None:
            self.result_df = pd.DataFrame()
        else:
            self.result_df = preload_res_df

    def load_videodata(self, videoname):
        """
        Provided the model parameters, load everything that is necessary to run
        the model for a given video into self.video_data.

        :param videoname: Single video for which the model should be run
        :type videoname: str
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
            if self.params["use_objectfiles"]:
                # Create list of object files, ASSUMPTION: masks have values from zero to nobj
                maxobj = np.max(viddata.object_masks)
                viddata.object_list = [
                    ObjectFile(obj_id, viddata.object_masks)
                    for obj_id in range(maxobj + 1)
                ]
        viddata.nframes = self.Dataset.video_frames[videoname]
        self.video_data = viddata  # save to class

    @abstractmethod
    def reinit_for_sgl_run():
        """Reinitialize all variables that are used in the model run."""
        pass

    @abstractmethod
    def update_features():
        """Module (I), defined in each model."""
        pass

    @abstractmethod
    def update_sensitivity():
        """Module (II), defined in each model."""
        pass

    @abstractmethod
    def update_ior():
        """Module (III), defined in each model."""
        pass

    @abstractmethod
    def update_decision():
        """Module (IV), defined in each model."""
        pass

    @abstractmethod
    def update_gaze():
        """Module (V), defined in each model."""
        pass

    def calc_sac_dur(self, dist_dva):
        """
        Calculate the saccade duration. We use literature values from
        Collewijn, H., Erkelens, C. J., & Steinman, R. M. (1988). Binocular co-ordination of human horizontal saccadic eye movements.
        :param dist_dva: saccade amplitude
        :type dist_dva: float
        """
        # sacdur = self.params["sacdur_slope"] * dist_dva + self.params["sacdur_intercept"]
        sacdur = 2.7 * dist_dva + 23.0
        return sacdur

    def sgl_vid_run(self, videoname, force_reload=False):
        """
        Run the model on a single video, depending on the implementation of the
        modules (I-V).

        :param videoname: Name of the video to run the model on
        :type videoname: str
        :param force_reload: Reload the video data (usually avoided), defaults to False
        :type force_reload: bool, optional
        :return: Result dictionary, containing the scanpath and saccade times
        :rtype: dict
        """
        assert self.params is not None, "Model parameters not loaded"
        # load the relevant data for videoname if not already loaded (or forced)
        if self.video_data is None or force_reload:
            self.load_videodata(videoname)
            print("Loaded video (None or reload) for", videoname)
        elif self.video_data["videoname"] != videoname:
            self.load_videodata(videoname)
            print("Loaded video (new name) for", videoname)
        assert self.video_data is not None, "Video data not loaded"

        # If provided, set random seed.
        if self.params.rs:
            np.random.seed(self.params.rs)

        # reinit all variables
        self.reinit_for_sgl_run()  # TODO!

        # set initial gaze location
        self._gaze_loc = self.params["startpos"].copy()
        self._scanpath.append(self._gaze_loc.copy())

        # Loop through all frames and run all modules
        # no new location in prediction in last frame => len(scanpath)=nframes
        self._t = 0.0
        nfov = 0
        dfov_list = []
        fov_start_f = 0
        fov_start_t = 0.0
        ongoing_sacdur = 0.0
        prev_sacdur = 0.0
        prev_sacamp = np.nan
        fov_start_loc = self._gaze_loc.copy()  # copy start position

        for f in range(self.video_data["nframes"] - 1):
            self._current_frame = f

            # do not accumulate evidence for the time the saccade was ongoing
            self._cur_fov_frac = np.clip(
                (self._cur_waiting_time + self._dt - ongoing_sacdur) / self._dt,
                0.0,
                1.0,
            )

            # if saccade takes up the whole frame, do not update decision variables
            if self._cur_fov_frac == 0.0:
                self._cur_waiting_time += self._dt
            else:
                self._cur_waiting_time = 0
                ongoing_sacdur = 0.0

            self.update_features()
            self.update_sensitivity()
            self.update_ior()
            self.update_decision()
            self.update_gaze()

            # update time
            self._t += self._dt

            # store when a saccade was made in list
            if self._new_target is not None:
                fov_dur = self._t - fov_start_t - self._cur_waiting_time
                # print(f"DEBUG: frame: {f}, time {self._t}, target {self._new_target}, fov_start_t {fov_start_t}, waiting time: {self._cur_waiting_time}, ongoing sacdur: {ongoing_sacdur}, cur fov frac: {self._cur_fov_frac}, fov_dur: {fov_dur}")
                # nfov, frame_start, frame_end, fov_dur, x_start, y_start, x_end, y_end, sac_dur
                dfov_list.append(
                    [
                        nfov,
                        fov_start_f,
                        f,
                        fov_dur,
                        fov_start_loc[0],
                        fov_start_loc[1],
                        self._prev_gaze_loc[0],
                        self._prev_gaze_loc[1],
                        prev_sacamp,
                        prev_sacdur,
                    ]
                )
                fov_start_loc = self._gaze_loc.copy()
                prev_sacamp = (
                    np.linalg.norm(self._gaze_loc - self._prev_gaze_loc)
                    * self.Dataset.PX_TO_DVA
                )
                ongoing_sacdur = self.calc_sac_dur(prev_sacamp)
                prev_sacdur = ongoing_sacdur.copy()
                fov_start_t = fov_start_t + fov_dur + prev_sacdur
                fov_start_f = int(fov_start_t // self._dt)

                self._f_sac.append(f)
                nfov += 1

            # add updated gaze location to scanpath in every frame
            self._scanpath.append(self._gaze_loc.copy())

        # last foveation ends with end of video
        dfov_list.append(
            [
                nfov,
                fov_start_f,
                self.video_data["nframes"] - 1,
                self._t - fov_start_t + self._dt,
                fov_start_loc[0],
                fov_start_loc[1],
                self._gaze_loc[0],
                self._gaze_loc[1],
                prev_sacamp,
                prev_sacdur,
            ]
        )

        # create dataframe out of dfov_list
        dfov = pd.DataFrame(
            dfov_list,
            columns=[
                "nfov",
                "frame_start",
                "frame_end",
                "duration_ms",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "sac_amp_dva",
                "sac_dur",
            ],
        )
        res_dict = {
            "gaze": np.asarray(self._scanpath),
            "f_sac": np.asarray(self._f_sac),
            "dfov": dfov,
        }
        return res_dict

    def run(self, videos_to_run, seeds=[], overwrite_old=False):
        """
        Main interfacing function to run a model.

        Results are then stored in the `result_dict` attribute.

        :param videos_to_run: Keyword for which videos to use, either a single video name, `test`, `train`, or `all`
        :type videos_to_run: str
        :param seeds: list of seeds which will each result in a separate run/`trial` of the model, defaults to []
        :type seeds: list, optional
        :param overwrite_old: If you want to run the same model again set this to True to overwrite the previous results, defaults to False
        :type overwrite_old: bool, optional
        :raises Exception: If the `videos_to_run` argument is not one of the allowed values
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

        # now run the model for the given videos & seeds:
        for i, vid in enumerate(videos):
            video_res_dict = {}
            logging.info(
                f"Run video {i+1}/{len(videos)}: {vid} from videos_to_run {videos_to_run}..."
            )
            self.load_videodata(vid)
            for s in seeds:
                self.params["rs"] = s
                # write the result (dictionary) in result_dict
                video_res_dict[f"seed{s:03d}"] = self.sgl_vid_run(vid)
            self.result_dict[vid] = video_res_dict

    def save_model(self, filename):
        """
        Save the model to a pickle file.

        :param filename: Include the full path and name of the model results, the extension
            will be appended by `.pkl`.
        :type filename: str
        """
        with open(f"{filename}.pkl", "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Model {self.name} is stored to {filename}.pkl")

    def clear_model_outputs(self):
        """
        Clears the model's results to create a fresh one
        """
        self.result_dict = {}
        self.result_df = pd.DataFrame()

    #######
    ## Evaluation functions
    #######

    def select_videos(self, videos_to_eval="all"):
        """
        Convenience function that selects the videos to be used for analysis.

        :param videos_to_eval: Keyword for which videos to use, either a single video name, `test`, `train`, or `all`, defaults to "all"
        :type videos_to_eval: str, optional
        :raises Exception: None of the allowed keywords were given.
        :return: List of strings with the name of the videos to be used for analysis
        :rtype: list
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
    
    def get_all_dur_amp(self):
        """
        Get the duration and amplitude of all saccades in all trials.
        
        :return: All durations and amplitudes of the model's results
        :rtype: np.array, np.array
        """
        # initialize empty np.arrays for durations and amplitudes
        durations = []
        amplitudes = []

        for vid in self.result_dict:
            for run in self.result_dict[vid]:
                durations.extend(self.result_dict[vid][run]["dfov"]["duration_ms"].dropna().values)
                amplitudes.extend(self.result_dict[vid][run]["dfov"]["sac_amp_dva"].dropna().values)
        
        if len(amplitudes) == 0:
            logging.warning("No saccades in model results.")
            amplitudes.append(0)
        
        return np.array(durations), np.array(amplitudes)

    def evaluate_trial(self, videoname, runname, segmentation_masks=None):
        """
        Evaluation function that returns a dataframe with all relevant foveation
        and saccade statistics for a single trial.

        :param videoname: Name of the video the model was run on
        :type videoname: str
        :param runname: Name of the run / seed the model was run for
        :type runname: str
        :param segmentation_masks: The segmentation masks of the video should be
            passed here, such that they dont have to be loaded for each trial, defaults to None
        :type segmentation_masks: np.array, optional
        :return: Dataframe with all relevant foveation and saccade statistics of this trial
        :rtype: pd.DataFrame
        """

        assert (
            self.result_dict
        ), "`result_dict` is empty. You need to run the model before evaluating the results."
        run_dict = self.result_dict[videoname][runname]
        assert {"gaze", "f_sac", "dfov"} <= set(
            run_dict
        ), "Integration method did not provide `gaze`, `f_sac`, and `dfov`."
        # option to pass masks so they dont have to be loaded each time in the loop
        if segmentation_masks is None:
            segmentation_masks = self.Dataset.load_objectmasks(videoname)

        # add additional evaluations to the existing foveation dataframe ()
        df = self.result_dict[videoname][runname]["dfov"]
        N_fov = len(df)
        # columns where the value is the same for all foveations in this run
        df.insert(1, "video", videoname)
        df.insert(2, "subject", runname)
        # check most common object between start and end frame
        df["object"] = [
            Counter(
                ", ".join(
                    [
                        # get all foveated objects in this foveation
                        # with tolerance of 1 dva (as for the human eye tracking data)
                        uf.object_at_position(
                            segmentation_masks[f],
                            run_dict["gaze"][f][0],
                            run_dict["gaze"][f][1],
                            radius=self.Dataset.RADIUS_OBJ_GAZE,
                        )
                        for f in range(
                            df["frame_start"].iloc[n], df["frame_end"].iloc[n] + 1
                        )
                    ]
                ).split(", ")
            ).most_common(1)[0][0]
            for n in range(N_fov)
        ]

        # calculate a number of saccade properties based on the gaze shift
        # depending on the end of the current fov and beginning of next one
        diff = np.array(
            [
                run_dict["gaze"][f + 1] - run_dict["gaze"][f]
                for f in df["frame_end"][:-1]
            ]
        )
        # aviod error if no saccades are made
        if diff.size:
            # first foveation is excluded since no saccade preceedes it!
            # df["sac_amp_eval"] = [np.nan] + list(
            #     np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2) * self.Dataset.PX_TO_DVA
            # )
            df["sac_angle_h"] = [np.nan] + list(
                np.arctan2(diff[:, 1], diff[:, 0]) / np.pi * 180
            )
            # second entry of angle_p will also be nan since first angle_h is nan
            df["sac_angle_p"] = [np.nan] + [
                uf.angle_limits(df["sac_angle_h"][i + 1] - df["sac_angle_h"][i])
                for i in range(N_fov - 1)
            ]
        else:
            df["sac_amp_eval"] = [np.nan]
            df["sac_angle_h"] = [np.nan]
            df["sac_angle_p"] = [np.nan]

        # add start and end time of each foveation
        df["fov_end"] = (df["duration_ms"] + df["sac_dur"]).cumsum()
        df["fov_start"] = df["fov_end"] - df["duration_ms"]

        # calculate the foveation categories (Background, Detection, Inspection, Revisit)
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
                    return_prev_t = df["fov_end"][
                        prev_obj.where(prev_obj == obj).last_valid_index()
                    ]
                    # store time difference [in milliseconds] in array!
                    ret_times[n] = (
                        (df["fov_start"].iloc[n] - return_prev_t)
                    )
        df["fov_category"] = fov_categories
        df["ret_times"] = ret_times

        # return the dataframes
        return df

    def evaluate_all_to_df(self, overwrite_old=False):
        """
        Evaluate all trials in `result_dict` and store the results in `result_df`.

        Runs `evaluate_trial` for each trial in `result_dict`.

        :param overwrite_old: Should not be repeated if its already been evaluated, defaults to False
        :type overwrite_old: bool, optional
        :return: Result dataframe of the model run
        :rtype: pd.DataFrame
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
            segmentation_masks = self.Dataset.load_objectmasks(videoname)
            for runname in self.result_dict[videoname]:
                df_trial = self.evaluate_trial(videoname, runname, segmentation_masks)
                self.result_df = pd.concat(
                    [self.result_df, df_trial], ignore_index=True
                )

        return self.result_df
    
    def evaluate_all_obj(self):
        """
        DIR evaluation based on individual objects

        :param overwrite_old: Should not be repeated if its already been evaluated, defaults to False
        :type overwrite_old: bool, optional
        :return: Result dataframe of the model run
        :rtype: pd.DataFrame       
        """        
        assert (
            self.result_df.empty == False
        ), "`result_df` is empty. You need to evaluate the foveations before the objects."
        # if len(self.result_obj.index) > 0:
        #     assert (
        #         overwrite_old
        #     ), "`result_df` is already filled, pass `overwrite_old=True` if you want to overwrite it"
        obj_list = []
        df = self.result_df
        for videoname in df["video"].unique():
            df_vid = df[df["video"] == videoname]
            # segmentation_masks = self.Dataset.load_objectmasks(videoname)
            for sub in df_vid["subject"].unique():
                df_trial = df_vid[df_vid["subject"] == sub]
        
                for obj_id in df_trial["object"].unique():
                    # dobj[f"Object {obj_id}"] = {}
                    # get all foveations where this object was foveated
                    dtemp = df_trial[df_trial["object"] == obj_id]
                    # add the overall time this object was foveated for each category
                    d_cat = {}
                    for fov_cat in ["D", "I", "R"]:
                        d_cat[fov_cat] = np.sum(
                            dtemp["duration_ms"][dtemp["fov_category"] == fov_cat]
                        )

                    obj_list.append(
                        {
                            "video": videoname,
                            "subject": sub,
                            "object": obj_id,
                            "D": d_cat["D"],
                            "I": d_cat["I"],
                            "R": d_cat["R"],
                        }
                    )
        # convert to dataframe
        dfobj = pd.DataFrame(obj_list)
        return dfobj



    def get_fovcat_ratio(self, videos_to_eval="all"):
        """
        Convenience function that returns the ratios as dictionary for the different categories.

        :param videos_to_eval: Keyword for videos, defaults to "all"
        :type videos_to_eval: str, optional
        :return: Dictionary with the ratios for the different categories
        :rtype: dict
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
            gt_fovmaps = self.Dataset.load_nssmaps(videoname)
            nframes = gt_fovmaps.shape[0]
            vid_nss_scores = []
            for runname in self.result_dict[videoname]:
                run_dict = self.result_dict[videoname][runname]
                nss_frames = [
                    gt_fovmaps[f, run_dict["gaze"][f][1], run_dict["gaze"][f][0]]
                    for f in range(nframes)
                ]
                vid_nss_scores.append(np.mean(nss_frames))
            nss_with_std[videoname] = [np.mean(vid_nss_scores), np.std(vid_nss_scores)]

        return nss_with_std

    def video_output_gif(self, videoname, storagename, slowgif=False, dpi=100):
        """
        General function that takes all the predicted scanpaths and plots it on top
        of the original video. No further details are visualized, hence it is model agnostic.

        :param storagename: Name of the file, will be appended to outputpath +".gif"
        :type storagename: str
        :param slowgif: If true, store it with 10fps, otherwise 30fps, defaults to False
        :type slowgif: bool, optional
        :param dpi: DPI for each frame when created and stored gif, defaults to 100
        :type dpi: int, optional
        """
        if hasattr(self.Dataset, "outputpath"):
            outputpath = self.Dataset.outputpath + storagename
        else:
            outputpath = f"{self.Dataset.PATH}results/{storagename}"

        assert (
            videoname in self.result_dict
        ), f"No simulated scanpaths for {videoname} yet, first run the model!"
        res_dict = self.result_dict[videoname]
        # load frames of the original video (never used by the model)
        vidlist = self.Dataset.load_videoframes(videoname)

        @gif.frame
        def frame(f):
            fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
            ax.imshow(vidlist[f])
            for key in res_dict.keys():
                ax.scatter(
                    res_dict[key]["gaze"][f][0],
                    res_dict[key]["gaze"][f][1],
                    s=150,
                    alpha=0.8,
                )
                if f > 0:
                    ax.plot(
                        [res_dict[key]["gaze"][f - 1][0], res_dict[key]["gaze"][f][0]],
                        [res_dict[key]["gaze"][f - 1][1], res_dict[key]["gaze"][f][1]],
                        linewidth=7,
                        ls=":",
                    )
            ax.set_axis_off()

        out = [frame(i) for i in range(len(vidlist))]
        if slowgif:
            gif.save(out, outputpath + "_slow.gif", duration=100)
            print(f"Saved to {outputpath}_slow.gif")
        else:
            gif.save(out, outputpath + ".gif", duration=33)
            print(f"Saved to {outputpath}.gif")
