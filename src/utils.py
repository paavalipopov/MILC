# from a2c_ppo_acktr.envs import make_vec_envs
# from a2c_ppo_acktr.utils import get_vec_normalize
import argparse
from collections import defaultdict
import copy
import os
import subprocess

import numpy as np
from sklearn.metrics import f1_score as compute_f1_score
import torch

# methods that need encoder trained before
train_encoder_methods = ["sub-enc-lstm", "sub-lstm"]
probe_only_methods = [
    "supervised",
    "random-cnn",
    "majority",
    "pretrained-rl-agent",
]
pre_train_encoder_methods = ["basic", "milc"]

from src.settings import UTCNOW


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prefix",
        type=str,
        default=f"{UTCNOW}",
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    parser.add_argument(
        "--ds",
        type=str,
        choices=[
            "oasis",
            "adni",
            "abide",
            "abide_869",
            "abide_roi",
            "fbirn",
            "fbirn_100",
            "fbirn_200",
            "fbirn_400",
            "fbirn_1000",
            "cobre",
            "bsnip",
            "hcp",
            "hcp_roi",
            "ukb",
            "ukb_age_bins",
            "time_fbirn",
        ],
        required=True,
    )

    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=[
            "oasis",
            "adni",
            "abide",
            "abide_869",
            "abide_roi",
            "fbirn",
            "fbirn_100",
            "fbirn_200",
            "fbirn_400",
            "fbirn_1000",
            "cobre",
            "bsnip",
            "hcp",
            "hcp_roi",
            "ukb",
            "ukb_age_bins",
            "time_fbirn",
        ],
    )

    parser.add_argument(
        "--pre-training",
        type=str,
        default="milc",
        choices=pre_train_encoder_methods,
        help="Pre-Training Method to Use (default: Basic )",
    )
    parser.add_argument("--fMRI-twoD", action="store_true", default=False)
    parser.add_argument("--deep", action="store_true", default=False)
    parser.add_argument("--complete_arc", action="store_true", default=True)
    parser.add_argument("--no-complete_arc", dest="complete_arc", action="store_false")
    parser.add_argument(
        "--path",
        type=str,
        default="/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb",
        help="Path to store the encoder (default: )",
    )
    parser.add_argument(
        "--oldpath",
        type=str,
        default="/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb",
        help="Path to store the encoder (default: )",
    )
    parser.add_argument(
        "--fig-path",
        type=str,
        default="/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb",
        help="Path to store the encoder (default: )",
    )
    parser.add_argument(
        "--p-path",
        type=str,
        default="/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb",
        help="Path to store the encoder (default: )",
    )
    parser.add_argument(
        "--exp", type=str, default="UFPT", help="the exp to run (default:FPT )"
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.1,
        help="gain value for init (default:0.5 )",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Temperature for division of norms in pre training",
    )
    parser.add_argument(
        "--script-ID",
        type=int,
        default=1,
        help="Task Array ID, chooses the required subjects per sample and gain value",
    )
    parser.add_argument(
        "--teststart-ID", type=int, default=1, help="Task Set Start Index ID"
    )
    parser.add_argument("--job-ID", type=int, default=1, help="Job Array ID")
    parser.add_argument("--sample-number", type=int, default=0, help="Job Array ID")
    parser.add_argument(
        "--env-name",
        default="MontezumaRevengeNoFrameskip-v4",
        help="environment to train on (default: MontezumaRevengeNoFrameskip-v4)",
    )
    parser.add_argument(
        "--num-frame-stack",
        type=int,
        default=1,
        help="Number of frames to stack for a state",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        default=True,
        help="Whether to use a linear classifier",
    )
    parser.add_argument(
        "--pretraining-steps",
        type=int,
        default=100000,
        help="Number of steps to pretrain representations (default: 100000)",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=50000,
        help="Number of steps to train probes (default: 30000 )",
    )
    #     parser.add_argument('--probe-test-steps', type=int, default=15000,
    #                         help='Number of steps to train probes (default: 15000 )')
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of parallel environments to collect samples from (default: 8)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sub-lstm",
        choices=train_encoder_methods + probe_only_methods,
        help="Method to use for training representations (default: sub-lstm)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        default=True,
        help="Whether to use a linear classifier",
    )
    parser.add_argument(
        "--use_multiple_predictors",
        action="store_true",
        default=False,
        help="Whether to use multiple linear classifiers in the contrastive loss",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning Rate foe learning representations (default: 5e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-Batch Size (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3000,
        help="Number of epochs for  (default: 100)",
    )
    parser.add_argument("--cuda-id", type=int, default=1, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="NatureOne",
        choices=["Nature", "NatureOne"],
        help="Encoder type (Impala or Nature or NatureOne)",
    )
    parser.add_argument(
        "--feature-size", type=int, default=256, help="Size of features"
    )
    parser.add_argument(
        "--fMRI-feature-size", type=int, default=1024, help="Size of features"
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument("--color", action="store_true", default=False)
    parser.add_argument("--end-with-relu", action="store_true", default=False)
    parser.add_argument("--wandb-proj", type=str, default="curl-atari-neurips-scratch")
    parser.add_argument("--num_rew_evals", type=int, default=10)
    # rl-probe specific arguments
    parser.add_argument("--checkpoint-index", type=int, default=-1)

    # naff-specific arguments
    parser.add_argument(
        "--naff_fc_size",
        type=int,
        default=2048,
        help="fully connected layer width for naff",
    )
    parser.add_argument(
        "--pred_offset",
        type=int,
        default=1,
        help="how many steps in future to predict",
    )
    # CPC-specific arguments
    parser.add_argument(
        "--sequence_length", type=int, default=100, help="Sequence length."
    )
    parser.add_argument(
        "--steps_start",
        type=int,
        default=0,
        help="Number of immediate future steps to ignore.",
    )
    parser.add_argument(
        "--steps_end",
        type=int,
        default=99,
        help="Number of future steps to predict.",
    )
    parser.add_argument(
        "--steps_step",
        type=int,
        default=4,
        help="Skip every these many frames.",
    )
    parser.add_argument(
        "--gru_size",
        type=int,
        default=256,
        help="Hidden size of the GRU layers.",
    )
    parser.add_argument(
        "--lstm_size",
        type=int,
        default=200,
        help="Hidden size of the LSTM layers.",
    )
    parser.add_argument(
        "--fMRI-lstm_size",
        type=int,
        default=512,
        help="Hidden size of the LSTM layers.",
    )
    parser.add_argument(
        "--gru_layers", type=int, default=2, help="Number of GRU layers."
    )
    parser.add_argument(
        "--lstm_layers", type=int, default=1, help="Number of LSTM layers."
    )
    parser.add_argument(
        "--collect-mode",
        type=str,
        choices=["random_agent", "pretrained_ppo"],
        default="random_agent",
    )

    parser.add_argument("--beta", default=1.0)
    # probe arguments
    parser.add_argument("--weights-path", type=str, default="None")
    parser.add_argument("--train-encoder", action="store_true", default=True)
    parser.add_argument("--probe-lr", type=float, default=3e-6)
    parser.add_argument(
        "--probe-collect-mode",
        type=str,
        choices=["random_agent", "pretrained_ppo"],
        default="random_agent",
    )
    parser.add_argument("--num-runs", type=int, default=1)
    return parser


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calculate_accuracy(preds, y):
    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()
    return acc


def calculate_accuracy_by_labels(preds, y):
    acc = torch.sum(torch.eq(preds, y).float()) / len(y)
    return acc


def calculate_FP_Max(indices, ts_number):
    FP = 0.0
    N = len(ts_number)
    for i in range(len(indices)):
        x = indices[i]
        if ts_number[x] != ts_number[i]:
            FP += 1

    return FP / N


def calculate_FP(metrics, ts_number):
    FP = 0.0
    FP2 = 0.0
    N = 0
    unique = torch.unique(ts_number)
    for b in range(len(unique)):
        x = torch.sum(ts_number == unique[b].item()).item()
        y = torch.sum(ts_number != unique[b].item()).item()
        ml = x * y
        N += ml
    index = (metrics > 0.5).nonzero()
    cols = torch.unique(index[:, 0])
    unique_cols = torch.unique(cols)

    for i in range(len(unique_cols)):
        col_index = (index[:, 0] == unique_cols[i]).nonzero()
        col_val = index[col_index, 1]
        xt_tsindex = ts_number[unique_cols[i]].item()
        xrem_tsindex = ts_number[col_val]
        FP += torch.sum((xrem_tsindex != xt_tsindex)).item()
    # for a in range(len(index)):
    #     if ts_number[index[a][0].item()] != ts_number[index[a][1].item()]:
    #         FP = FP + 1
    # print('N = ', N)
    # print('FP = ', FP)
    # print('FP2 = ', FP2)
    FP = float(FP)
    FP = FP / N
    return torch.tensor(FP)


def calculate_multiclass_f1_score(preds, labels):
    preds = torch.argmax(preds, dim=1).detach().numpy()
    labels = labels.numpy()
    f1score = compute_f1_score(labels, preds, average="weighted")
    return f1score


def calculate_multiclass_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    acc = float(torch.sum(torch.eq(labels, preds)).data) / labels.size(0)
    return acc


class appendabledict(defaultdict):
    def __init__(self, type_=list, *args, **kwargs):
        self.type_ = type_
        super().__init__(type_, *args, **kwargs)

    #     def map_(self, func):
    #         for k, v in self.items():
    #             self.__setitem__(k, func(v))

    def subslice(self, slice_):
        """indexes every value in the dict according to a specified slice

        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.


        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}

        """
        sliced_dict = {}
        for k, v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict

    def append_update(self, other_dict):
        """appends current dict's values with values from other_dict

        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to append to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

        """
        for k, v in other_dict.items():
            self.__getitem__(k).append(v)


# Thanks Bjarten! (https://github.com/Bjarten/early-stopping-pytorch)
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        encoder_backup,
        lstm_backup,
        patience=7,
        verbose=False,
        wandb=None,
        name="",
        path="",
        trial="",
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.0
        self.val_min_loss = 0.0
        self.name = name
        self.wandb = wandb
        self.path = path
        self.trial = trial
        self.encoder_backup = ""
        self.lstm_backup = ""
        self.attn_backup = ""
        self.cone_backup = ""
        self.key_backup = ""
        self.value_backup = ""
        self.query_backup = ""
        self.mha_backup = ""
        self.threshold = 0.0001
        self.a = 0

    def __call__(
        self,
        val_loss,
        val_auc,
        model,
        lstm,
        attn,
        cone,
        save=0,
        key="",
        value="",
        query="",
        mha="",
    ):

        if save == 0:
            score = val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(
                    val_loss,
                    val_auc,
                    model,
                    lstm,
                    attn,
                    cone,
                    save,
                    key,
                    value,
                    query,
                    mha,
                )
            elif self.best_score - score <= self.threshold:
                self.counter += 1
                if self.counter >= 10:
                    print(
                        f"EarlyStopping for {self.name} counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
                    print(f"{self.name} has stopped")

            else:
                self.best_score = score
                self.save_checkpoint(
                    val_loss,
                    val_auc,
                    model,
                    lstm,
                    attn,
                    cone,
                    save,
                    key,
                    value,
                    query,
                    mha,
                )
                self.counter = 0
        else:
            self.save_checkpoint(
                val_loss,
                val_auc,
                model,
                lstm,
                attn,
                cone,
                save,
                key,
                value,
                query,
                mha,
            )

    def save_checkpoint(
        self,
        val_loss,
        val_auc,
        model,
        lstm,
        attn,
        cone,
        save,
        key,
        value,
        query,
        mha,
    ):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation accuracy increased for {self.name}  ({self.val_acc_max:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        if save == 1:
            # self.a = 0
            torch.save(
                self.encoder_backup,
                os.path.join(self.path, self.name + self.trial + ".pt"),
            )
            torch.save(
                self.lstm_backup,
                os.path.join(self.path, "lstm" + self.trial + ".pt"),
            )
            torch.save(
                self.attn_backup,
                os.path.join(self.path, "attn" + self.trial + ".pt"),
            )
            torch.save(
                self.cone_backup,
                os.path.join(self.path, "cone" + self.trial + ".pt"),
            )
            # torch.save(self.key_backup, os.path.join(self.path, 'key' + self.trial + '.pt'))
            # torch.save(self.value_backup, os.path.join(self.path, 'value' + self.trial + '.pt'))
            # torch.save(self.query_backup, os.path.join(self.path, 'query' + self.trial + '.pt'))
            # torch.save(self.mha_backup, os.path.join(self.path, 'mha' + self.trial + '.pt'))
        else:
            encoder_state = model.state_dict()
            lstm_state = lstm.state_dict()
            attn_state = attn.state_dict()
            cone_state = cone.state_dict()
            # key_state = key.state_dict()
            # value_state = value.state_dict()
            # query_state = query.state_dict()
            # mha_state = mha.state_dict()

            self.lstm_backup = copy.deepcopy(lstm_state)
            self.encoder_backup = copy.deepcopy(encoder_state)
            self.attn_backup = copy.deepcopy(attn_state)
            self.cone_backup = copy.deepcopy(cone_state)
            # self.key_backup = copy.deepcopy(key_state)
            # self.value_backup = copy.deepcopy(value_state)
            # self.query_backup = copy.deepcopy(query_state)
            # self.mha_backup = copy.deepcopy(mha_state)

            self.val_min_loss = val_loss
            self.val_acc_max = val_auc


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
