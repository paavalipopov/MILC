from collections import deque
import datetime
from itertools import chain
import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.All_Architecture import combinedModel
from src.encoders_ICA import NatureCNN, NatureOneCNN
from src.lstm_attn import subjLSTM
from src.slstm_attn_catalyst import LSTMTrainer
from src.utils import get_argparser

from src.ts_data import load_dataset

import torch
import wandb
from sklearn.model_selection import StratifiedKFold, train_test_split


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):
    for k in range(5):
        for my_trial in range(10):
            wandb_logger = wandb.init(
                project=f"{args.prefix}-experiment-milc-{args.ds}",
                name=f"k_{k}-trial_{my_trial}",
                save_code=True,
            )

            start_time = time.time()
            # do stuff

            # ID = args.script_ID + 3
            ID = args.script_ID - 1

            JobID = args.job_ID
            print("ID = " + str(ID))
            print("exp = " + args.exp)
            print("pretraining = " + args.pre_training)
            currentDT = datetime.datetime.now()
            d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
            d2 = str(JobID) + "_" + str(ID)

            Name = args.exp + "_FBIRN_" + args.pre_training
            dir = "run-" + d1 + d2 + Name
            dir = dir + "-" + str(ID)
            output_path = "Output"
            opath = os.path.join(os.getcwd(), output_path)
            args.path = opath

            wdb1 = "wandb_new"
            wpath1 = os.path.join(os.getcwd(), wdb1)

            tfilename = str(JobID) + "outputFILE" + Name + str(ID)

            output_path = os.path.join(os.getcwd(), "Output")
            output_path = os.path.join(output_path, tfilename)

            ####### new
            features, labels = load_dataset(args.ds)
            #######
            # data_shape = features.shape

            ntrials = 1
            sample_x = 53
            sample_y = 20

            ###### old
            # subjects = 157
            # tc = 140
            ###### new
            subjects = features.shape[0]
            tc = features.shape[2]
            print(f"Subjects: {subjects}")
            print(f"TC: {tc}")
            ######

            samples_per_subject = int(tc / sample_y)
            window_shift = 10
            if torch.cuda.is_available():
                cudaID = str(torch.cuda.current_device())
                device = torch.device("cuda:" + cudaID)
            else:
                device = torch.device("cpu")
            if args.exp == "FPT":
                gain = [0.1, 0.05, 0.05]  # FPT
            elif args.exp == "UFPT":
                gain = [0.05, 0.45, 0.65]  # UFPT
            else:
                gain = [0.25, 0.35, 0.65]  # NPT

            current_gain = gain[ID]
            args.gain = current_gain

            data = features

            if args.fMRI_twoD:
                finalData = data
                finalData = torch.from_numpy(finalData).float()
                finalData = finalData.permute(0, 2, 1)
                finalData = finalData.reshape(
                    finalData.shape[0], finalData.shape[1], finalData.shape[2], 1
                )
            else:
                finalData = np.zeros(
                    (subjects, samples_per_subject, sample_x, sample_y)
                )
                for i in range(subjects):
                    for j in range(samples_per_subject):
                        finalData[i, j, :, :] = data[
                            i, :, (j * window_shift) : (j * window_shift) + sample_y
                        ]
                finalData = torch.from_numpy(finalData).float()

            print(finalData.shape)
            finalData2 = finalData

            all_labels = labels
            all_labels = torch.from_numpy(all_labels).int()

            ####### additional_test_datasets
            extra_test_eps = []
            extra_test_labels = []
            for dataset in args.test_ds:
                local_features, local_labels = load_dataset(dataset)

                local_subjects = local_features.shape[0]
                local_tc = local_features.shape[2]

                local_samples_per_subject = int(local_tc / sample_y)

                if args.fMRI_twoD:
                    finalData = local_features
                    finalData = torch.from_numpy(finalData).float()
                    finalData = finalData.permute(0, 2, 1)
                    finalData = finalData.reshape(
                        finalData.shape[0], finalData.shape[1], finalData.shape[2], 1
                    )
                else:
                    finalData = np.zeros(
                        (local_subjects, local_samples_per_subject, sample_x, sample_y)
                    )
                    for i in range(local_subjects):
                        for j in range(local_samples_per_subject):
                            finalData[i, j, :, :] = local_features[
                                i, :, (j * window_shift) : (j * window_shift) + sample_y
                            ]
                    finalData = torch.from_numpy(finalData).float()

                extra_test_eps.append(
                    {
                        "name": dataset,
                        "eps": finalData,
                    }
                )

                extra_test_labels.append(
                    {
                        "name": dataset,
                        "labels": torch.from_numpy(local_labels).int(),
                    }
                )
            #######

            results = torch.zeros(ntrials, 4)
            for trial in range(ntrials):
                print("trial = ", trial)
                output_text_file = open(output_path, "a+")
                output_text_file.write("Trial = %d gTrial = %d\r\n" % (trial, 1))
                output_text_file.close()

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                skf.get_n_splits(finalData2, all_labels)

                train_index, test_index = list(skf.split(finalData2, all_labels))[k]

                tr_eps, test_eps = finalData2[train_index], finalData2[test_index]
                tr_labels, test_labels = all_labels[train_index], all_labels[test_index]

                tr_eps, val_eps, tr_labels, val_labels = train_test_split(
                    tr_eps,
                    tr_labels,
                    test_size=tr_eps.shape[0] // 5,
                    random_state=42 + my_trial,
                    stratify=tr_labels,
                )

                # print(tr_eps.shape)
                # print(val_eps.shape)
                # print(test_eps.shape)
                # print(tr_labels.shape)
                # print(val_labels.shape)
                # print(test_labels.shape)

                observation_shape = finalData2.shape
                print(f"OBSERVATION SPACE: {observation_shape}")
                if args.encoder_type == "Nature":
                    encoder = NatureCNN(observation_shape[2], args)

                elif args.encoder_type == "NatureOne":
                    dir = ""
                    if args.pre_training == "basic":
                        dir = "PreTrainedEncoders/Basic_Encoder"
                    elif args.pre_training == "milc":
                        args.oldpath = wpath1 + "/PreTrainedEncoders/Milc"

                    encoder = NatureOneCNN(observation_shape[2], args)
                    lstm_model = subjLSTM(
                        device,
                        args.feature_size,
                        args.lstm_size,
                        num_layers=args.lstm_layers,
                        freeze_embeddings=True,
                        gain=current_gain,
                    )

                complete_model = combinedModel(
                    encoder,
                    lstm_model,
                    gain=current_gain,
                    PT=args.pre_training,
                    exp=args.exp,
                    device=device,
                    oldpath=args.oldpath,
                    complete_arc=args.complete_arc,
                )
                config = {}
                config.update(vars(args))
                config["obs_space"] = observation_shape

                if args.method == "sub-lstm":
                    trainer = LSTMTrainer(
                        complete_model,
                        config,
                        device=device,
                        tr_labels=tr_labels,
                        val_labels=val_labels,
                        test_labels=test_labels,
                        extra_test_labels=extra_test_labels,
                        wandb="wandb",
                        trial=str(trial),
                    )
                elif args.method == "sub-enc-lstm":
                    print("Change method to sub-lstm")
                else:
                    assert False, "method {} has no trainer".format(args.method)

                (
                    results[trial][0],
                    results[trial][1],
                    results[trial][2],
                    results[trial][3],
                    extra_tests,
                ) = trainer.pre_train(tr_eps, val_eps, test_eps, extra_test_eps)

                wandb_logger.log(
                    {
                        "test_accuracy": results[trial][0],
                        "test_score": results[trial][1],
                        "test_loss": results[trial][2],
                    },
                )

                for extra_test in extra_tests:
                    wandb_logger.log(
                        {
                            extra_test["name"]
                            + "_test_accuracy": extra_test["test_accuracy"],
                            extra_test["name"] + "_test_score": extra_test["test_auc"],
                            extra_test["name"] + "_test_loss": extra_test["test_loss"],
                        },
                    )
                wandb_logger.finish()

                np_results = results.numpy()
                tresult_csv = os.path.join(args.path, f"test_results_{args.ds}.csv")
                with open(tresult_csv, "a") as csvfile:
                    np.savetxt(csvfile, np_results, delimiter=",")
                elapsed = time.time() - start_time
                print("total time = ", elapsed)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ["pretraining-only"]
    config = {}
    config.update(vars(args))

    if args.test_ds is None:
        args.test_ds = []

    print(args.complete_arc)
    train_encoder(args)
