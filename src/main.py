"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import argparse
import json
from datetime import datetime
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pathlib

sys.path.append("src")
sys.path.append("detr")
from engine import evaluate, train_one_epoch
from models import build_model
import util.misc as utils

import table_datasets as TD
from table_datasets import PDFTablesDataset, RandomMaxResize
from eval import eval_cocos


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root_dirs",
        required=True,
        help="Comma-separated root data directories for images and labels",
    )
    parser.add_argument(
        "--data_root_image_extensions",
        required=True,
        help="Comma separated image extensions, one per data root directory",
    )
    parser.add_argument(
        "--data_root_multiplicities",
        required=True,
        help="Oversampling factor, one per data root directory.",
    )
    parser.add_argument(
        "--config_file",
        required=True,
        help="Filepath to the config containing the args",
    )
    parser.add_argument("--backbone", default="resnet18", help="Backbone for the model")
    parser.add_argument(
        "--data_type",
        choices=["detection", "structure"],
        default="structure",
        help="toggle between structure recognition and table detection",
    )
    parser.add_argument("--model_load_path", help="The path to trained model")
    parser.add_argument("--load_weights_only", action="store_true")
    parser.add_argument(
        "--model_save_dir",
        help="The output directory for saving model params and checkpoints",
    )
    parser.add_argument(
        "--metrics_save_filepath", help="Filepath to save grits outputs", default=""
    )
    parser.add_argument(
        "--debug_save_dir", help="Filepath to save visualizations", default="debug"
    )
    # Example: '.*0\.[^.]+'
    parser.add_argument(
        "--debug_img_path_re_filter",
        help="Only debug images whose full path fully matches this regex.",
        default=".*",
    )
    parser.add_argument(
        "--table_words_dir", help="Folder containg the bboxes of table words"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "validate"],
        default="train",
        help="Modes: training (train) and evaluation (eval)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_drop", type=int)
    parser.add_argument("--lr_gamma", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint_freq", default=1, type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--train_max_size", type=int)
    parser.add_argument("--val_max_size", type=int)
    parser.add_argument("--test_max_size", type=int)
    parser.add_argument("--train_start_offset", type=int, default=0)
    parser.add_argument("--val_start_offset", type=int, default=0)
    parser.add_argument("--test_start_offset", type=int, default=0)
    parser.add_argument("--eval_pool_size", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--train_split_name", type=str, default="train")
    parser.add_argument("--train_xml_fileset", type=str)
    parser.add_argument("--test_split_name", type=str, default="test")

    parser.add_argument("--coco_eval_prefix", type=str)
    parser.add_argument(
        "--fused",
        action=argparse.BooleanOptionalAction,
        help="Use the experimental AdamW fused option.",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        help="Pin memory in data loader.",
    )
    parser.add_argument(
        "--enable_bounds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read bounds and transform to bounds otherwise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Used to randomly choose {train/val/test}_max_size the files if max_size < len(page_ids). Then also for the first epoch unless epoch_seeds[0] is specified.",
    )
    parser.add_argument("--epoch_seeds", type=int, nargs="*", default=[])
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument("--torch_printoptions", type=str, default="default")
    parser.add_argument(
        "--debug_samples",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--debug_losses",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--debug_gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--debug_model_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--debug_engine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the samples, losses and gradient for each iteration.",
    )
    parser.add_argument(
        "--debug_dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the page IDs as they are generated from PDFTablesDataset.",
    )
    parser.add_argument(
        "--debug_grits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug grits code.",
    )
    parser.add_argument(
        "--fast_grits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fast grits code.",
    )
    parser.add_argument(
        "--crop_around_midline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Crop around midline between hole and outer boundaries.",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int
    )
    parser.add_argument(
        "--wait_file_path",
        help="If this file exists, just wait.",
        type=pathlib.PosixPath,
    )
    return parser.parse_args()


def get_transform(data_type, image_set, *, enable_bounds, crop_around_midline):
    if data_type == "structure":
        return TD.get_structure_transform(image_set, enable_bounds=enable_bounds, crop_around_midline=crop_around_midline)
    else:
        return TD.get_detection_transform(image_set, enable_bounds=enable_bounds, crop_around_midline=crop_around_midline)


def get_class_map(data_type):
    if data_type == "structure":
        class_map = {
            "table": 0,
            "table column": 1,
            "table row": 2,
            "table column header": 3,
            "table projected row header": 4,
            "table spanning cell": 5,
            "no object": 6,
        }
    else:
        class_map = {"table": 0, "table rotated": 1, "no object": 2}
    return class_map


def get_data(args):
    """
    Based on the args, retrieves the necessary data to perform training,
    evaluation or GriTS metric evaluation
    """
    # Datasets
    print("loading data")
    class_map = get_class_map(args.data_type)

    if args.mode == "train":
        train_datasets = [
            PDFTablesDataset(
                os.path.join(data_root_dir, train_split_name),
                get_transform(args.data_type, "train", enable_bounds=args.enable_bounds, crop_around_midline=args.crop_around_midline),
                do_crop=False,
                max_size=args.train_max_size,
                include_eval=False,
                max_neg=0,
                make_coco=False,
                image_extension=data_root_image_extension,
                xml_fileset=args.train_xml_fileset
                or "{}_filelist.txt".format(train_split_name),
                class_map=class_map,
                start_offset=args.train_start_offset,
                enable_bounds=args.enable_bounds,
                debug_dataset=args.debug_dataset,
            )
            for train_split_name, data_root_dir, data_root_image_extension in zip(
                utils.split_by_comma(args.train_split_name),
                utils.split_by_comma(args.data_root_dirs),
                utils.split_by_comma(args.data_root_image_extensions),
                strict=True,
            )
        ]
        print(
            "Train dataset lengths: {}".format(
                [len(train_dataset) for train_dataset in train_datasets]
            )
        )
        dataset_train = ConcatDataset(
            sum(
                (
                    [
                        train_dataset,
                    ]
                    * int(data_root_dir_multiplicity)
                    for train_dataset, data_root_dir_multiplicity in zip(
                        train_datasets,
                        utils.split_by_comma(args.data_root_multiplicities),
                    )
                ),
                [],
            )
        )

        # Not sure why we still need bounds for validation, to investigate.
        val_during_training_enable_bounds = args.enable_bounds
        dataset_vals = [
            PDFTablesDataset(
                os.path.join(data_root_dir, "val"),
                get_transform(args.data_type, "val", enable_bounds=val_during_training_enable_bounds, crop_around_midline=args.crop_around_midline),
                do_crop=False,
                max_size=args.val_max_size,
                include_eval=False,
                make_coco=True,
                image_extension=data_root_image_extension,
                xml_fileset="val_filelist.txt",
                class_map=class_map,
                start_offset=args.val_start_offset,
                enable_bounds=val_during_training_enable_bounds,
                debug_dataset=args.debug_dataset,
            )
            for data_root_dir, data_root_image_extension in zip(
                utils.split_by_comma(args.data_root_dirs),
                utils.split_by_comma(args.data_root_image_extensions),
                strict=True,
            )
        ]

        # Does not generate any random numbers at init time:
        # https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py#L132
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True
        )

        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
        return (
            data_loader_train,
            [
                (
                    dataset_val,
                    DataLoader(
                        dataset_val,
                        args.val_batch_size or 2 * args.batch_size,
                        sampler=torch.utils.data.SequentialSampler(dataset_val),
                        drop_last=False,
                        collate_fn=utils.collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory,
                        prefetch_factor=args.prefetch_factor,
                    ),
                )
                for dataset_val in dataset_vals
            ],
            len(dataset_train),
        )

    elif args.mode in ["eval", "validate"]:
        dataset_tests = [
            PDFTablesDataset(
                os.path.join(data_root_dir, args.test_split_name),
                get_transform(args.data_type, "val", enable_bounds=args.enable_bounds, crop_around_midline=args.crop_around_midline),
                do_crop=False,
                max_size=args.test_max_size,
                include_eval=args.mode == "eval",
                make_coco=True,
                image_extension=data_root_image_extension,
                xml_fileset="{}_filelist.txt".format(args.test_split_name),
                class_map=class_map,
                start_offset=args.test_start_offset,
                enable_bounds=args.enable_bounds,
                debug_dataset=args.debug_dataset,
            )
            for data_root_dir, data_root_image_extension in zip(
                utils.split_by_comma(args.data_root_dirs),
                utils.split_by_comma(args.data_root_image_extensions),
                strict=True,
            )
        ]

        return [
            (
                dataset_test,
                DataLoader(
                    dataset_test,
                    args.test_batch_size or 2 * args.batch_size,
                    sampler=torch.utils.data.SequentialSampler(dataset_test),
                    drop_last=False,
                    collate_fn=utils.collate_fn,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    prefetch_factor=args.prefetch_factor,
                ),
            )
            for dataset_test in dataset_tests
        ]

    elif args.mode == "grits" or args.mode == "grits-all":
        dataset_tests = [
            PDFTablesDataset(
                os.path.join(data_root_dir, args.test_split_name),
                RandomMaxResize(1000, 1000, args.enable_bounds),
                include_original=True,
                max_size=args.max_test_size,
                make_coco=False,
                image_extension=data_root_image_extension,
                xml_fileset="{}_filelist.txt".format(args.test_split_name),
                class_map=class_map,
                start_offset=args.test_start_offset,
                enable_bounds=args.enable_bounds,
                debug_dataset=args.debug_dataset,
            )
            for data_root_dir, data_root_image_extension in zip(
                utils.split_by_comma(args.data_root_dirs),
                utils.split_by_comma(args.data_root_image_extensions),
                strict=True,
            )
        ]
        return ConcatDataset(dataset_tests)


# https://stackoverflow.com/posts/62764464/timeline
def numel(m: torch.nn.Module, only_trainable: bool = False):
    parameters = m.parameters()
    if only_trainable:
        parameters = (p for p in parameters if p.requires_grad)
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum((p.numel() for p in unique), 0)


def get_model(args, device):
    """
    Loads DETR model on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        print("loading model from checkpoint")
        loaded_state_dict = torch.load(args.model_load_path, map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    print("Trainable parameters: {}".format(numel(model, only_trainable=True)))
    print("All parameters: {}".format(numel(model, only_trainable=False)))
    return model, criterion, postprocessors


def set_seed(s):
    seed = s + utils.get_rank()
    print("Setting seed to: {}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def format_rng_state_dict(d, torch_printoptions):
    torch.set_printoptions(profile="full")
    repr_dict = {key: repr(t) for key, t in d.items()}
    torch.set_printoptions(torch_printoptions)
    return repr_dict


def get_rng_state_dict(device: torch.device):
    return {
        **{
            "cpu": torch.random.get_rng_state(),
        },
        **(
            {}
            if device.type == "cpu"
            else {device.type: torch.cuda.get_rng_state(device)}
        ),
    }

def evaluate_datasets(
    model,
    criterion,
    postprocessors,
    device,
    dataset_val_loaders,
    prefix):
    for i, (dataset_val, data_loader_val) in enumerate(dataset_val_loaders):
        if not len(dataset_val):
            continue
        stats, _ = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            dataset_val,
            device,
            None,
            "{}_valds_{}".format(
                prefix, i
            )
            if prefix
            else None,
        )
        print(
            "pubmed: AP50: {:.3f}, AP75: {:.3f}, AP: {:.3f}, AR: {:.3f} valds: {}".format(
                stats["coco_eval_bbox"][1],
                stats["coco_eval_bbox"][2],
                stats["coco_eval_bbox"][0],
                stats["coco_eval_bbox"][8],
                i
            )
        )

def move_rng_state_to_rng_state_dict(checkpoint):
    """Needed for /data/models/structure/pubmed/code/table-transformer/px/4/eb/enable_bounds/output/train_instant/202401150000000000/model.pth"""
    rng_state_dict = checkpoint.get('rng_state_dict')
    fixed_checkpoint = dict(checkpoint)
    bug = fixed_checkpoint.pop('rng_state', None)
    if bug and not rng_state_dict:
        fixed_checkpoint['rng_state_dict'] = bug
    return fixed_checkpoint

def train(args, model, criterion, postprocessors, device):
    """
    Training loop
    """

    print("loading data")
    dataloading_time = datetime.now()
    data_loader_train, dataset_val_loaders, train_len = get_data(args)
    print("finished loading data in :", datetime.now() - dataloading_time)

    max_batches_per_epoch = int(train_len / args.batch_size)
    print(
        "train_len: {} batch_size: {} args.fused: {}".format(
            train_len, args.batch_size, args.fused
        )
    )
    print("Max batches per epoch: {}".format(max_batches_per_epoch))

    print(
        "Starting with RNG state:\n{}".format(
            format_rng_state_dict(get_rng_state_dict(device), args.torch_printoptions)
        )
    )

    model_without_ddp = model
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay, fused=args.fused
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_drop, gamma=args.lr_gamma, verbose=True
    )

    resume_checkpoint = False
    if args.model_load_path:
        checkpoint = torch.load(args.model_load_path, map_location="cpu")
        # Fix for an older bug. Probably no longer needed.
        checkpoint = move_rng_state_to_rng_state_dict(checkpoint)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)

        if not args.load_weights_only and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Loading the scheduler step does not seem to make a difference. Probably because
            # we call step() without arguments, which is essentially stateless.
            # lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            resume_checkpoint = True
        elif args.load_weights_only:
            print(
                "*** WARNING: Resuming training and ignoring optimizer state. "
                "Training will resume with new initialized values. "
                "To use current optimizer state, remove the --load_weights_only flag."
            )
        else:
            print(
                "*** ERROR: Optimizer state of saved checkpoint not found. "
                "To resume training with new initialized values add the --load_weights_only flag."
            )
            raise Exception(
                "ERROR: Optimizer state of saved checkpoint not found. Must add --load_weights_only flag to resume training without."
            )

        print("checkpoint.keys(): {}".format(checkpoint.keys()))
        print("checkpoint.get('epoch']): {}".format(checkpoint.get("epoch")))
        if 'rng_state_dict' in checkpoint:
            print(
                "checkpoint.get('rng_state_dict'):\n{}".format(
                    format_rng_state_dict(checkpoint['rng_state_dict'], args.torch_printoptions)
                )
            )
        if not args.load_weights_only and "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        elif args.load_weights_only:
            print(
                "*** WARNING: Resuming training and ignoring previously saved epoch. "
                "To resume from previously saved epoch, remove the --load_weights_only flag."
            )
        else:
            print(
                "*** WARNING: Epoch of saved model not found. Starting at epoch {}.".format(
                    args.start_epoch
                )
            )
            raise Exception(
                "ERROR: Epoch of saved checkpoint not found. Must add --load_weights_only flag to resume training without."
            )
        if not args.load_weights_only and 'rng_state_dict' in checkpoint:
            rng_state_dict = checkpoint['rng_state_dict']
            torch.random.set_rng_state(rng_state_dict["cpu"])
            for device_type, value in rng_state_dict.items():
                if device_type != "cpu":
                    torch.cuda.set_rng_state(value, device=device_type)
        elif args.load_weights_only:
            print(
                "*** WARNING: Resuming training and ignoring previously saved RNG state. "
                "To resume from previously RNG state, remove the --load_weights_only flag."
            )
        else:
            print(
                "*** WARNING: RNG state of saved model not found. Starting at RNG state:\n{}".format(
                    format_rng_state_dict(get_rng_state_dict(device), args.torch_printoptions)
                )
            )
            raise Exception(
                "ERROR: RNG state of saved checkpoint not found. Must add --load_weights_only flag to resume training without."
            )

    # Use user-specified save directory, if specified
    if args.model_save_dir:
        output_directory = args.model_save_dir
    # If resuming from a checkpoint with optimizer state, save into same directory
    elif args.model_load_path and resume_checkpoint:
        output_directory = os.path.split(args.model_load_path)[0]
    # Create new save directory
    elif len(utils.split_by_comma(args.data_root_dirs)) == 1:
        run_date = datetime.now().strftime("%Y%m%d%H%M%S")
        output_directory = os.path.join(
            utils.split_by_comma(args.data_root_dirs)[0], "output", run_date
        )
    else:
        raise ValueError(
            "Need model_save_dir if multiple data root dirs and not resuming checkpoint."
        )

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Output directory: ", output_directory)
    model_save_path = os.path.join(output_directory, "model.pth")
    print("Output model path: ", model_save_path)
    if not resume_checkpoint and os.path.exists(model_save_path):
        print(
            "*** WARNING: Output model path exists but is not being used to resume training; training will overwrite it."
        )
        # raise ValueError("Will not continue to re-do the same work.")

    if args.start_epoch >= args.epochs:
        print(
            "*** WARNING: Starting epoch ({}) is greater or equal to the number of training epochs ({}).".format(
                args.start_epoch, args.epochs
            )
        )

    print("Start training")
    start_time = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        print("-" * 100)

        epoch_timing = datetime.now()
        if epoch < len(args.epoch_seeds):
            set_seed(args.epoch_seeds[epoch])
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            max_norm=args.clip_max_norm,
            max_batches_per_epoch=max_batches_per_epoch,
            print_freq=1000,
            debug_samples=args.debug_samples or args.debug_engine,
            debug_losses=args.debug_losses or args.debug_engine,
            debug_gradient=args.debug_gradient or args.debug_engine,
            debug_model_parameters=args.debug_model_parameters or args.debug_engine,
        )
        print("Epoch completed in ", datetime.now() - epoch_timing)
        print(train_stats)

        lr_scheduler.step()

        model_path_stem = "model_{}".format(epoch + 1)
        evaluate_datasets(
            model,
            criterion,
            postprocessors,
            device,
            dataset_val_loaders,
            "{}_model_path_stem_{}".format(
                args.coco_eval_prefix, model_path_stem
            ) if args.coco_eval_prefix else None
        )

        # Save current model training progress
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "scheduler_state_dict": lr_scheduler.state_dict(),
                'rng_state_dict': get_rng_state_dict(device),
            },
            model_save_path,
        )

        # Save checkpoint for evaluation
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save(model.state_dict(), os.path.join(
                output_directory, model_path_stem + ".pth"
            ))
            torch.save(get_rng_state_dict(device), os.path.join(
                output_directory, model_path_stem + "_rng_state_dict.pth"
            ))

    print(
        "Ending with RNG state:\n{}".format(
            format_rng_state_dict(get_rng_state_dict(device), args.torch_printoptions)
        )
    )
    print("Total training time: ", datetime.now() - start_time)


def main():
    print("Start: {}".format(datetime.now()))
    cmd_args = get_args().__dict__
    config_args = json.load(open(cmd_args["config_file"], "rb"))
    for key, value in cmd_args.items():
        if not key in config_args or not value is None:
            config_args[key] = value
    # config_args.update(cmd_args)
    args = type("Args", (object,), config_args)
    print(args.__dict__)
    print("-" * 100)

    if args.torch_num_threads:
        torch.set_num_threads(args.torch_num_threads)
    torch.set_printoptions(profile=args.torch_printoptions)

    # Check for debug mode
    if args.mode == "eval" and args.debug:
        print(
            "Running evaluation/inference in DEBUG mode, processing will take longer. Saving output to: {}.".format(
                args.debug_save_dir
            )
        )
        os.makedirs(args.debug_save_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        # Not sure if this helps at all with CPU.
        torch.use_deterministic_algorithms(True)

    print("loading model")
    model, criterion, postprocessors = get_model(args, device)

    if args.mode == "train":
        train(args, model, criterion, postprocessors, device)
    elif args.mode == "eval":
        eval_cocos(
            args,
            model,
            criterion,
            postprocessors,
            get_data(args),
            device,
            debug_samples=args.debug_samples or args.debug_engine,
            debug_losses=args.debug_losses or args.debug_engine,
        )
    elif args.mode == "validate":
        evaluate_datasets(
            model,
            criterion,
            postprocessors,
            device,
            get_data(args),
            "{}_model_path_stem_{}".format(
                args.coco_eval_prefix, "model"
            ) if args.coco_eval_prefix else None
        )
    print("End: {}".format(datetime.now()))

if __name__ == "__main__":
    main()
