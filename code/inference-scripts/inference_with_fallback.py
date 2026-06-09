"""
Inference runner with scene-level fallback support.

Matches best.ipynb's behavior:
  - Primary pipeline: LoMa + ALIKED ensemble
  - If COLMAP registers < fallback_min_ratio of images, re-run with
    ALIKED+LightGlue fallback pipeline
  - Transparent scenes use ensemble rank-fusion ordering (no SfM fallback)

Unlike inference.py, this runs reconstruction synchronously per scene so
that the fallback check can be performed inline.
"""

import argparse
import os
import pandas as pd
import shutil
import json
import gc
import torch
from pathlib import Path
from pipeline import Pipeline
from reconstruction import reconstruction


def parse_sample_submission(base_path, input_csv, target_datasets):
    data_dict = {}
    with open(input_csv, "r") as f:
        for i, l in enumerate(f):
            if i == 0:
                print("header:", l)
            if l and i > 0:
                image_path, dataset, scene, _, _ = l.strip().split(',')
                if target_datasets is not None and dataset not in target_datasets:
                    continue
                if not os.path.isfile(os.path.join(base_path, image_path)):
                    continue
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(Path(os.path.join(base_path, image_path)))

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")
    return data_dict


def run_pipeline_and_reconstruct(data_dict, dataset, scene, base_path,
                                  work_dir, pipeline_json, input_dir_root,
                                  colmap_mapper_options, device_id):
    """Run feature pipeline + COLMAP reconstruction for one scene. Returns submission_path."""
    with open(pipeline_json, "r") as f:
        pipeline_config = json.load(f)

    pipeline = Pipeline(
        data_dict[dataset][scene], work_dir,
        input_dir_root, pipeline_config, device_id
    )
    pipeline.exec()

    torch.cuda.empty_cache()
    gc.collect()

    submission_path = reconstruction(
        data_dict, dataset, scene, base_path,
        work_dir, colmap_mapper_options
    )
    return submission_path


def count_registered(submission_path, dataset, scene):
    """Count how many images got non-identity poses in the submission CSV."""
    if submission_path is None or not os.path.isfile(submission_path):
        return 0
    df = pd.read_csv(submission_path)
    # Identity rotation = "1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0"
    identity_rot = "1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0"
    registered = 0
    for _, row in df.iterrows():
        if str(row["rotation_matrix"]).strip() != identity_rot:
            registered += 1
    return registered


def run(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--target_datasets", nargs="*", required=False)
    parser.add_argument("--output_dir", required=False)
    args = parser.parse_args()

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.target_datasets:
        config.target_datasets = args.target_datasets

    if config.check_exist_dir and os.path.isdir(config.output_dir):
        raise Exception(f"{config.output_dir} already exists.")

    os.makedirs(config.output_dir, exist_ok=True)
    base_path = os.path.join(config.input_dir_root, "image-matching-challenge-2024")
    feature_dir = os.path.join(config.output_dir, "feature_outputs")
    shutil.copy(config.pipeline_json, config.output_dir)
    shutil.copy(config.transparent_pipeline_json, config.output_dir)

    # Load categories
    category_df = pd.read_csv(config.category_csv)
    categories = {}
    for _, row in category_df.iterrows():
        categories[row["scene"]] = row["categories"].split(";")

    data_dict = parse_sample_submission(base_path, config.input_csv, config.target_datasets)
    datasets = list(data_dict.keys())

    fallback_enabled = getattr(config, 'fallback_enabled', False)
    fallback_min_ratio = getattr(config, 'fallback_min_ratio', 0.3)
    fallback_pipeline_json = getattr(config, 'fallback_pipeline_json', None)

    submission_path_list = []

    for dataset in datasets:
        for scene in data_dict[dataset]:
            print(f"\n{'='*60}")
            print(f"[Scene] {dataset} / {scene}")
            print(f"{'='*60}")

            work_dir = Path(os.path.join(feature_dir, f"{dataset}_{scene}"))
            work_dir.mkdir(parents=True, exist_ok=True)

            is_transparent = "transparent" in categories.get(scene, [])

            # Select pipeline JSON
            if is_transparent:
                json_path = config.transparent_pipeline_json
            else:
                json_path = config.pipeline_json
            print(f"categories: {categories.get(scene, [])}")
            print(f"pipeline: {json_path}")

            # Run primary pipeline + reconstruction
            submission_path = run_pipeline_and_reconstruct(
                data_dict, dataset, scene, base_path,
                work_dir, json_path, config.input_dir_root,
                config.colmap_mapper_options, args.device_id
            )

            # --- Scene-level fallback (best.ipynb logic) ---
            if (
                fallback_enabled
                and not is_transparent
                and fallback_pipeline_json is not None
            ):
                total_images = len(data_dict[dataset][scene])
                registered = count_registered(submission_path, dataset, scene)
                ratio = registered / max(1, total_images)

                print(f"[Fallback check] registered={registered}/{total_images} "
                      f"({ratio:.3f}), threshold={fallback_min_ratio}")

                if ratio < fallback_min_ratio:
                    print(f"[Fallback] Ratio {ratio:.3f} < {fallback_min_ratio}; "
                          f"rerunning with ALIKED+LightGlue fallback pipeline.")

                    fallback_work_dir = Path(os.path.join(
                        feature_dir, f"{dataset}_{scene}_fallback"
                    ))
                    fallback_work_dir.mkdir(parents=True, exist_ok=True)

                    submission_path = run_pipeline_and_reconstruct(
                        data_dict, dataset, scene, base_path,
                        fallback_work_dir, fallback_pipeline_json,
                        config.input_dir_root,
                        config.colmap_mapper_options, args.device_id
                    )

                    fallback_registered = count_registered(submission_path, dataset, scene)
                    print(f"[Fallback result] registered={fallback_registered}/{total_images}")

                    # Use fallback result only if it's actually better
                    if fallback_registered <= registered:
                        print("[Fallback] Primary was better or equal; keeping primary.")
                        submission_path = os.path.join(work_dir, "submission.csv")

            if submission_path is not None:
                submission_path_list.append(submission_path)

    # Concat all submissions
    if len(submission_path_list) > 0:
        submission_df_list = [pd.read_csv(p) for p in submission_path_list]
        submission_df = pd.concat(submission_df_list).reset_index(drop=True)
    else:
        submission_df = pd.DataFrame.from_dict({
            "image_path": [],
            "dataset": [],
            "scene": [],
            "rotation_matrix": [],
            "translation_vector": []
        })
    output_path = os.path.join(config.output_dir, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"\nFinal submission: {output_path} ({len(submission_df)} rows)")


if __name__ == '__main__':
    cfg = Config
    run(cfg)
