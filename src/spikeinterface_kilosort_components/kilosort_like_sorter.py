import importlib



from spikeinterface.core import (
    get_noise_levels,
    NumpySorting,
    estimate_templates_with_accumulator,
    Templates,
    compute_sparsity,
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
        "motion_correction": {"preset": "kilosort_like"},
        "filtering": {"freq_min": 150.0, "freq_max": 10000.0, "ftype":"bessel", "filter_order": 2,},
        "waveforms": {
            "ms_before": 2.,
            "ms_after": 2.,
            "radius_um": 80.0,
        },
        "detection": {"peak_sign":"neg", "detect_threshold": 6},
        "selection": {"n_peaks_per_channel": 5000, "min_n_peaks": 20000},
        "clustering": {
            "n_svd": 5,
            "verbose": False,
            "engine": "torch",
            # "torch_device": "cpu",
            "torch_device": "cuda",
            "cluster_downsampling": 20,
            "n_nearest_channels" : 10
        },
        "templates": {
            "sparsity_threshold": 1.5,
        },
        "matching": {
            "Th" : 8,
            "max_iter" : 100,
            "engine" : "torch",
            "torch_device" : "cpu",
        },
        "apply_final_auto_merge": True,
        "job_kwargs": {},
        "save_array": True,
    }

    _params_description = {
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
        from spikeinterface.sortingcomponents.clustering.main import find_cluster_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd
        from spikeinterface.sortingcomponents.tools import remove_empty_templates
        from spikeinterface.preprocessing import correct_motion
        from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

        

        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = verbose

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        # preprocessing
        if params["apply_motion_correction"]:
            rec_for_motion = recording_raw

            rec_for_motion = bandpass_filter(rec_for_motion, freq_min=300.0, freq_max=6000.0, dtype="float32")
            rec_for_motion = common_reference(rec_for_motion)
            if verbose:
                print("Start correct_motion()")
            _, motion_info = correct_motion(
                rec_for_motion,
                folder=sorter_output_folder / "motion",
                output_motion_info=True,
                **params["motion_correction"],
            )
            if verbose:
                print("Done correct_motion()")

        recording = bandpass_filter(recording_raw, **params["filtering"], dtype="float32")
        recording = common_reference(recording)

        if params["apply_motion_correction"]:
            interpolate_motion_kwargs = dict(
                border_mode="force_extrapolate",
                spatial_interpolation_method="kriging",
                sigma_um=20.0,
                p=2,
            )

            recording = InterpolateMotionRecording(
                recording,
                motion_info["motion"],
                **interpolate_motion_kwargs,
            )

        recording = whiten(recording, dtype="float32", mode="local", radius_um=100.0)

        # Save the preprocessed recording
        cache_folder = sorter_output_folder / "cache_preprocessing"
        recording = recording.save_to_folder(folder=cache_folder, **job_kwargs)

        noise_levels = get_noise_levels(recording, return_scaled=False, **job_kwargs)

        # this will be propagated over several methods
        ms_before = params['waveforms']['ms_before']
        ms_after = params['waveforms']['ms_after']

        # detection
        prototype, waveforms, _ = get_prototype_and_waveforms_from_recording(
            recording,
            n_peaks=10000,
            ms_before=ms_before,
            ms_after=ms_after,
            **job_kwargs,
        )
        detection_params = params["detection"].copy()
        detection_params["prototype"] = prototype
        detection_params["ms_before"] = ms_before
        all_peaks = detect_peaks(recording, method="matched_filtering", **detection_params, **job_kwargs)

        if verbose:
            print(f"detect_peaks(): {len(all_peaks)} peaks found")

        # selection
        selection_params = params["selection"].copy()
        n_peaks = params["selection"]["n_peaks_per_channel"] * num_chans
        n_peaks = max(selection_params["min_n_peaks"], n_peaks)
        peaks = select_peaks(all_peaks, method="uniform", n_peaks=n_peaks)

        if verbose:
            print(f"select_peaks(): {len(peaks)} peaks kept for clustering")

        clustering_kwargs = params["clustering"].copy()
        clustering_kwargs["ms_before"] = ms_before
        clustering_kwargs["ms_after"] = ms_after
        unit_ids, clustering_label, more_outs = find_cluster_from_peaks(
            recording, peaks, method="kilosort-clustering", method_kwargs=clustering_kwargs, extra_outputs=True, **job_kwargs
        )

        mask = clustering_label >= 0
        sorting_pre_peeler = NumpySorting.from_samples_and_labels(
            peaks["sample_index"][mask],
            clustering_label[mask],
            sampling_frequency,
            unit_ids=unit_ids,
        )
        if verbose:
            print(f"find_cluster_from_peaks(): {sorting_pre_peeler.unit_ids.size} cluster found")

        templates_dense, _ = get_templates_from_peaks_and_svd(
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
        # spasify to remove zeros
        sparsity = compute_sparsity(templates_dense, method="snr", noise_levels=noise_levels, threshold=0.1)
        templates = templates_dense.to_sparse(sparsity)
        templates = remove_empty_templates(templates)

        ## peeler kilosort4 need temporal_components
        n_svd = params["clustering"]["n_svd"]
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        wfs = waveforms / np.linalg.norm(waveforms, axis=1)[:, None]
        model = KMeans(n_clusters=n_svd, n_init=10).fit(wfs)
        temporal_components = model.cluster_centers_
        temporal_components = temporal_components / np.linalg.norm(temporal_components[:, None])
        temporal_components = temporal_components.astype(np.float32)
        model = TruncatedSVD(n_components=n_svd).fit(wfs)
        spatial_components = model.components_.astype(np.float32)
    

        matching_params = params["matching"].copy()
        matching_params["templates"] = templates
        matching_params["temporal_components"] = temporal_components
        matching_params["spatial_components"] = spatial_components

        spikes = find_spikes_from_templates(
            recording, method="kilosort-matching", method_kwargs=matching_params, **job_kwargs
        )

        final_spikes = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        final_spikes["sample_index"] = spikes["sample_index"]
        final_spikes["unit_index"] = spikes["cluster_index"]
        final_spikes["segment_index"] = spikes["segment_index"]
        sorting = NumpySorting(final_spikes, sampling_frequency, templates.unit_ids)


        ## DEBUG auto merge
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
                template_diff_thresh=np.arange(0.05, 0.4, 0.05),
                debug_folder=None,
                **job_kwargs,
            )
            sorting = NumpySorting.from_sorting(analyzer_final.sorting)


        if params["save_array"]:
            sorting_pre_peeler = sorting_pre_peeler.save(folder=sorter_output_folder / "sorting_pre_peeler")
            if params["apply_motion_correction"]:
                motion_info["motion"].save(sorter_output_folder / "motion")
            np.save(sorter_output_folder / "noise_levels.npy", noise_levels)
            np.save(sorter_output_folder / "all_peaks.npy", all_peaks)
            np.save(sorter_output_folder / "peaks.npy", peaks)
            np.save(sorter_output_folder / "clustering_label.npy", clustering_label)
            np.save(sorter_output_folder / "spikes.npy", spikes)
            templates.to_zarr(sorter_output_folder / "templates.zarr")

        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        return sorting
