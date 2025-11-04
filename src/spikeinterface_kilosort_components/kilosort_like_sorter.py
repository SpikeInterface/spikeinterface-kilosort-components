import importlib



from spikeinterface.core import (
    get_noise_levels,
    NumpySorting,
)

from spikeinterface.core.job_tools import fix_job_kwargs

from spikeinterface.preprocessing import bandpass_filter, common_reference, whiten
from spikeinterface.core.basesorting import minimum_spike_dtype

from spikeinterface.sorters.internal.si_based import ComponentsBasedSorter


import numpy as np


class Kilosort4LikeSorter(ComponentsBasedSorter):
    """
    This is a proof of concept of implementing the kilosort4 sorter using spikeinterface sorting components.

    Here the steps:
      * optional motion estimation/correction
      * preprocessing : filter + cmr + whitening
      * detect peak using "matched_filtering"
      * compute SVD
      * clustering with "kilosort-clustering" (re implented in this repo)
      * template matching with "kilosort-matching" (re implented in this repo)
      * final merge using the spikeinterface auto merge machinery.

    """

    sorter_name = "kilosort4like"

    _default_params = {
        "apply_motion_correction": False,
        "apply_preprocessing": True,
        "apply_whitening": True,
        "motion_correction": {"preset": "kilosort_like"},
        "filtering": {"freq_min": 150.0, "freq_max": 7000, "ftype": "bessel", "filter_order": 2, "margin_ms": 20},
        "waveforms": {
            "ms_before": 1.,
            "ms_after": 1.,
            "radius_um": 100.0,
        },
        "detection": {"peak_sign":"neg", "detect_threshold": 5},
        "selection": {"n_peaks_per_channel": 5000, "min_n_peaks": 100000, "select_per_channel": False},
        "clustering": {
            "peaks_svd" : dict(n_components=6),
            "verbose": False,
            "engine": "torch",
            "torch_device": "cuda",
            "cluster_downsampling": 1,
            "n_nearest_channels" : 10,
            "max_cluster_subset": 25000,
            "cluster_neighbors": 10,
            "dminx": 32,
            "min_cluster_size": 20,
        },
        "cleaning" : {"min_snr" : 3, "max_jitter_ms" : 0.1, "sparsify_threshold" : None},
        "matching": {
            "Th" : 10, # the real KS has 8 here but 10 seems better
            "max_iter" : 100,
            "engine" : "torch",
            "torch_device" : "cpu",
        },
        "apply_final_auto_merge": True,
        "job_kwargs": {},
        "save_array": True,
    }

    _params_description = {
        "apply_preprocessing": "Boolean to specify whether to preprocess the recording or not. If yes, then high_pass filtering + common\
                                                    median reference + whitening",
        "apply_motion_correction": "Apply motion correction or not",
        "motion_correction" : "Parameters for motion estimation/correction",
        "waveforms": "A dictonary containing waveforms params: ms_before, ms_after, radius_um",
        "filtering": "A dictonary containing filtering params: freq_min, freq_max",
        "detection": "A dictonary containing detection params: peak_sign, detect_threshold, exclude_sweep_ms, radius_um",
        "selection": "A dictonary containing selection params: n_peaks_per_channel, min_n_peaks",
        "clustering": "A dictonary containing clustering params: split_radius_um, merge_radius_um",
        "templates": "A dictonary containing waveforms params for peeler: ms_before, ms_after",
        "matching": "A dictonary containing matching params for matching: peak_shift_ms, radius_um",
        "job_kwargs": "A dictionary containing job kwargs",
        "save_array": "Save or not intermediate arrays",
    }

    handle_multi_segment = True

    @classmethod
    def get_sorter_version(cls):
        return "1."

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):

        from spikeinterface.sortingcomponents.tools import get_prototype_and_waveforms_from_recording
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering.main import find_clusters_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd
        from spikeinterface.preprocessing import correct_motion
        from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
        from spikeinterface.sortingcomponents.tools import clean_templates
        
        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = verbose

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        # preprocessing
        if params["apply_preprocessing"]:
            recording_f = recording_raw
            recording_f = bandpass_filter(recording_f, **params["filtering"], dtype="float32")
            recording_f = common_reference(recording_f)
        else:
            recording_f = recording_raw
            recording_f.annotate(is_filtered=True)

        if params["apply_whitening"]:
            recording_w = whiten(recording_f, dtype="float32", mode="local", radius_um=100.0)

        if params["apply_motion_correction"]:
            
            if verbose:
                print("Starting motion correction")

            _, motion_info = correct_motion(
                recording_w,
                folder=sorter_output_folder / "motion",
                output_motion_info=True,
                **params["motion_correction"],
            )

            interpolate_motion_kwargs = dict(
                border_mode="force_extrapolate",
                spatial_interpolation_method="kriging",
                sigma_um=20.0,
                p=2,
            )

            recording = InterpolateMotionRecording(
                recording_w,
                motion_info["motion"],
                **interpolate_motion_kwargs,
            )
        else:
            recording = recording_w

        # Save the preprocessed recording
        cache_folder = sorter_output_folder / "cache_preprocessing"
        recording = recording.save_to_folder(folder=cache_folder, **job_kwargs)

        noise_levels = get_noise_levels(recording, return_scaled=False, **job_kwargs)

        # this will be propagated over several methods
        ms_before = params['waveforms']['ms_before']
        ms_after = params['waveforms']['ms_after']

        ## Step : detection
        prototype, waveforms, _ = get_prototype_and_waveforms_from_recording(
            recording,
            n_peaks=10000,
            ms_before=ms_before,
            ms_after=ms_after,
            radius_um=params["waveforms"]["radius_um"] / 2,
            exclude_sweep_ms=max(ms_before, ms_after),
            noise_levels=noise_levels,
            job_kwargs=job_kwargs,
        )
        detection_params = params["detection"].copy()
        detection_params["prototype"] = prototype
        detection_params["ms_before"] = ms_before
        all_peaks = detect_peaks(recording, method="matched_filtering", method_kwargs=detection_params, job_kwargs=job_kwargs)

        if verbose:
            print(f"detect peaks: {len(all_peaks)} peaks found")

        ## Step : clustering

        # select a subset of peaks
        selection_params = params["selection"].copy()
        n_peaks = params["selection"]["n_peaks_per_channel"] * num_chans
        n_peaks = max(selection_params["min_n_peaks"], n_peaks)
        peaks = select_peaks(all_peaks, method="uniform", n_peaks=n_peaks)

        if verbose:
            print(f"select subset of peaks: {len(peaks)} peaks kept for clustering")

        clustering_kwargs = params["clustering"].copy()
        clustering_kwargs["peaks_svd"]["ms_before"] = ms_before
        clustering_kwargs["peaks_svd"]["ms_after"] = ms_after
        unit_ids, clustering_label, more_outs = find_clusters_from_peaks(
            recording, peaks, method="kilosort-clustering", method_kwargs=clustering_kwargs, extra_outputs=True, job_kwargs=job_kwargs,
        )

        mask = clustering_label >= 0
        sorting_pre_peeler = NumpySorting.from_samples_and_labels(
            peaks["sample_index"][mask],
            clustering_label[mask],
            sampling_frequency,
            unit_ids=unit_ids,
        )
        if verbose:
            print(f"find_clusters_from_peaks(): {sorting_pre_peeler.unit_ids.size} cluster found")

        # create th template from the median of SVD
        templates_dense, new_sparse_mask = get_templates_from_peaks_and_svd(
            recording,
            peaks,
            clustering_label,
            ms_before,
            ms_after,
            more_outs["svd_model"],
            more_outs["peaks_svd"],
            more_outs["peak_svd_sparse_mask"],
            operator="median",
        )

        templates = templates_dense.to_sparse(new_sparse_mask)

        # and clean small ones
        cleaning_kwargs = params.get("cleaning", {}).copy()
        cleaning_kwargs["noise_levels"] = noise_levels
        cleaning_kwargs["remove_empty"] = True
        templates = clean_templates(
            templates,
            **cleaning_kwargs
        )
        
        if verbose:
            print("Kept %d clean clusters" % len(templates.unit_ids))

        ## Step : template matching
        # peeler kilosort4 need temporal_components
        n_svd = clustering_kwargs["peaks_svd"]["n_components"]
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        wfs = waveforms / np.linalg.norm(waveforms, axis=1)[:, None]
        model = KMeans(n_clusters=n_svd, n_init=10).fit(wfs)
        temporal_components = model.cluster_centers_
        temporal_components = temporal_components / np.linalg.norm(temporal_components, axis=1)[:, None]
        temporal_components = temporal_components.astype(np.float32)
        model = TruncatedSVD(n_components=n_svd).fit(wfs)
        spatial_components = model.components_.astype(np.float32)
    

        matching_params = params["matching"].copy()
        matching_params["temporal_components"] = temporal_components
        matching_params["spatial_components"] = spatial_components

        spikes = find_spikes_from_templates(
            recording, templates, method="kilosort-matching", method_kwargs=matching_params, job_kwargs=job_kwargs
        )

        final_spikes = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        final_spikes["sample_index"] = spikes["sample_index"]
        final_spikes["unit_index"] = spikes["cluster_index"]
        final_spikes["segment_index"] = spikes["segment_index"]
        sorting = NumpySorting(final_spikes, sampling_frequency, templates.unit_ids)

        if verbose:
            print("Found %d spikes" % len(final_spikes))

        ## Step auto merge
        if params["apply_final_auto_merge"]:
            from spikeinterface.sorters.internal.spyking_circus2 import final_cleaning_circus

            # max_distance_um = merging_params.get("max_distance_um", 50)
            # merging_params["max_distance_um"] = max(max_distance_um, 2 * max_motion)

            analyzer_final =  final_cleaning_circus(
                recording,
                sorting,
                templates,
                similarity_kwargs={"method": "l1", "support": "union", "max_lag_ms": 0.1},
                sparsity_overlap=0.5,
                censor_ms=3.0,
                max_distance_um=50,
                template_diff_thresh=np.arange(0.05, 0.5, 0.05),
                debug_folder=None,
                job_kwargs=job_kwargs,
            )
            sorting = NumpySorting.from_sorting(analyzer_final.sorting)

            if verbose:
                print(f"Kept {len(sorting.unit_ids)} units after final merging")


        if params["save_array"]:
            sorting_pre_peeler = sorting_pre_peeler.save(folder=sorter_output_folder / "sorting_pre_peeler")
            np.save(sorter_output_folder / "noise_levels.npy", noise_levels)
            np.save(sorter_output_folder / "all_peaks.npy", all_peaks)
            np.save(sorter_output_folder / "peaks.npy", peaks)
            np.save(sorter_output_folder / "clustering_label.npy", clustering_label)
            np.save(sorter_output_folder / "spikes.npy", spikes)
            templates.to_zarr(sorter_output_folder / "templates.zarr")

        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        return sorting
