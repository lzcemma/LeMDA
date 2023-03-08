import os
import argparse
import json
import time
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from autogluon.text.automm import AutoMMPredictor
from autogluon.text.automm.utils import make_exp_dir
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    BINARY,
    MULTICLASS,
)
from autogluon.core.utils.loaders import load_zip

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="petfinder", type=str)
    parser.add_argument("--exp-dir", default="exp", type=str)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--config-dir", default=None, type=str)
    parser.add_argument("--model-config-name", default="fusion_mlp_image_text_tabular", type=str)
    parser.add_argument("--data-config-name", default="default", type=str)
    parser.add_argument("--optim-config-name", default="adamw", type=str)
    parser.add_argument("--env-config-name", default="default", type=str)
    parser.add_argument("--max-img-num-per-col", default=1, type=int)
    parser.add_argument("--ckpt-path", default=None, type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "overrides", metavar="N", type=str, nargs="*", help="Additional flags to overwrite the configuration"
    )
    parser.add_argument("--verbosity", default=3, type=int)
    parser.add_argument("--time-limit", default=None, type=int)
    parser.add_argument("--dataset-id", default=None, type=int)
    return parser


def main(args):
    dataset_path = 'data/petfinder_processed'
    train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
    test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)
    valid_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)
    label_column = 'AdoptionSpeed'



    image_col = 'Images'
    train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0]) # Use the first image for a quick tutorial
    test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])
    valid_data[image_col] = valid_data[image_col].apply(lambda ele: ele.split(';')[0])


    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    valid_data[image_col] = valid_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

    print(f"train sample num: {len(train_data)}")
    print(f"valid sample num: {len(valid_data)}")
    print(f"test sample num: {len(test_data)}")

    predictor = AutoMMPredictor(
        label="AdoptionSpeed",
        problem_type="multiclass",
        eval_metric="quadratic_kappa",
        verbosity=3,
    )

    save_path = os.path.join(args.exp_dir, "petfiner")
    save_path = make_exp_dir(
        root_path=save_path,
        job_name="petfiner",
        create=False,
    )
    predictor.fit(
        train_data=train_data,
        tuning_data=valid_data,
        time_limit=args.time_limit,
        save_path=save_path,
    )

    scores, y_pred = predictor.evaluate(
        data=test_data,
        metrics=["quadratic_kappa"],
        return_pred=True,
    )
    print(scores)



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
