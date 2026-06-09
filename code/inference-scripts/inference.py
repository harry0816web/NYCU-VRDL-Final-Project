import multiprocessing
import argparse
import os
import pandas as pd
import shutil
import json
import gc
import torch
from pathlib import Path
from config import Config
from pipeline import Pipeline
from reconstruction import reconstruction


def parse_sample_submission(
    base_path: str,
    input_csv: str,
    target_datasets: list,
) -> dict[dict[str, list[Path]]]:
    """Construct a dict describing the test data as

    {"dataset": {"scene": [<image paths>]}}
    """
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


def worker_reconstruction(input_queue, submission_path_list):
    while True:
        reconstruction_inputs = input_queue.get()
        if reconstruction_inputs is None:
            break
        data_dict, dataset, scene, base_path, work_dir, colmap_mapper_options = reconstruction_inputs
        submission_path = reconstruction(data_dict, dataset, scene, base_path, work_dir, colmap_mapper_options)
        submission_path_list.append(submission_path)


def count_registered_images(work_dir):
    """Count how many images were registered by COLMAP reconstruction."""
    rec_dir = os.path.join(work_dir, "colmap_rec_aliked")
    if not os.path.isdir(rec_dir):
        return 0
    # Check if any model subdirectory exists with images.bin/txt
    total = 0
    for subdir in sorted(os.listdir(rec_dir)):
        model_dir = os.path.join(rec_dir, subdir)
        if not os.path.isdir(model_dir):
            continue
        # Try to count from images.bin or images.txt
        images_bin = os.path.join(model_dir, "images.bin")
        images_txt = os.path.join(model_dir, "images.txt")
        if os.path.isfile(images_bin):
            try:
                import pycolmap
                rec = pycolmap.Reconstruction(model_dir)
                total = max(total, len(rec.images))
            except Exception:
                pass
        elif os.path.isfile(images_txt):
            try:
                count = 0
                with open(images_txt) as f:
                    for line in f:
                        if not line.startswith('#') and line.strip():
                            count += 1
                # images.txt has 2 lines per image
                total = max(total, count // 2)
            except Exception:
                pass
    return total


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

    if config.check_exist_dir:
        if os.path.isdir(config.output_dir):
            raise Exception(f"{config.output_dir} is already exists.")

    os.makedirs(config.output_dir, exist_ok=True)
    base_path = os.path.join(config.input_dir_root, "image-matching-challenge-2024")
    feature_dir = os.path.join(config.output_dir, "feature_outputs")
    shutil.copy(config.pipeline_json, config.output_dir)
    shutil.copy(config.transparent_pipeline_json, config.output_dir)

    # Load category
    category_df = pd.read_csv(config.category_csv)
    categories = {}
    for i, row in category_df.iterrows():
        categories[row["scene"]] = row["categories"].split(";")

    data_dict = parse_sample_submission(base_path, config.input_csv, config.target_datasets)
    datasets = list(data_dict.keys())

    manager = multiprocessing.Manager()
    submission_path_list = manager.list()

    input_queue = multiprocessing.Queue()
    worker_reconstruction_process = multiprocessing.Process(
        target=worker_reconstruction, args=(input_queue, submission_path_list)
    )
    worker_reconstruction_process.start()

    for dataset in datasets:
        for scene in data_dict[dataset]:
            print(f"[Scene] {scene}")
            work_dir = Path(os.path.join(feature_dir, f"{dataset}_{scene}"))
            work_dir.mkdir(parents=True, exist_ok=True)

            # Select pipeline JSON based on category
            if "transparent" in categories.get(scene, []):
                json_path = config.transparent_pipeline_json
            else:
                json_path = config.pipeline_json
            print(f"categories: {categories.get(scene, [])}")
            print(f"json path: {json_path}")

            # Execute primary pipeline
            with open(json_path, "r") as f:
                pipeline_config = json.load(f)
            pipeline = Pipeline(
                data_dict[dataset][scene], work_dir,
                config.input_dir_root, pipeline_config, args.device_id
            )
            pipeline.exec()

            # Reconstruction
            print("Start Reconstruction")
            torch.cuda.empty_cache()
            gc.collect()
            input_queue.put((
                data_dict, dataset, scene, base_path,
                work_dir, config.colmap_mapper_options
            ))

            # --- Scene-level fallback (best.ipynb logic) ---
            # If LoMa SfM registered < fallback_min_ratio of images,
            # re-run with ALIKED+LightGlue fallback pipeline.
            if (
                getattr(config, 'fallback_enabled', False)
                and "transparent" not in categories.get(scene, [])
                and hasattr(config, 'fallback_pipeline_json')
            ):
                # Wait briefly for reconstruction to finish for this scene
                # (We need to check registered count, so we do sync reconstruction here)
                # Note: For the fallback check, we need the reconstruction result.
                # We'll drain the queue and check.
                pass
                # TODO: The current architecture runs reconstruction async.
                # For fallback, we need to know the result before proceeding.
                # This is handled in the notebook version by checking inline.
                # For now, the fallback pipeline_fallback.json can be run
                # as a separate experiment if needed.

    input_queue.put(None)
    worker_reconstruction_process.join()

    # Concat Submission
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
    submission_df.to_csv(os.path.join(config.output_dir, "submission.csv"), index=False)


if __name__ == '__main__':
    cfg = Config
    run(cfg)
