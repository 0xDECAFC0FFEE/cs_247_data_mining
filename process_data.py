import os
import pandas as pd
import random
import numpy as np
from utils import *
from itertools import product
import re
from ast import literal_eval
from pathlib import Path
import pickle

def filename_from_args(dataset_path, domain, traintest, phase, filetype, extension):
    """gets folder structure filename from arguments

    Arguments:
        dataset_path -- root path
        domain {[type]} -- [description]
        traintest {[type]} -- [description]
        phase {[type]} -- [description]
        filetype {[type]} -- [description]

    Raises:
        Exception: [description]

    Returns:
        [type] -- [description]
    """
    folder = Path(f"{dataset_path}/{domain}/underexpose_{traintest}")

    if filetype in ["click", "qtime"]:
        filename = f"underexpose_{traintest}_{filetype}-{phase}.{extension}"
    elif filetype in ["user", "item"]:
        filename = f"underexpose_{filetype}_feat.{extension}"
    else:
        raise Exception("filetype not recognized")
        
    return folder, filename

def get_raw_dataset(domain="international", traintest="train", phase="0", filetype="click"):
    """gets dataset associated with arguments and does some cleaning

    Keyword Arguments:
        domain {str} -- "international" or "china" (default: {"international"})
        traintest {str} -- "train" or "test" (default: {"train"})
        phase {str} -- number between 0 and 6. (default: {"0"})
        filetype {str} -- one of "click", "item", "user", "qtime" (default: {"click"})
        
    if filetype is click, user or qtime, returns cleaned dataframe
    if filetype is item, returns the item_ids, text_vecs, and img_vecs
    """

    folder, filename= filename_from_args("dataset", domain, traintest, phase, filetype, "csv")

    if filetype == "item":
        header = ["item_id","text_vec","img_vec"]
        with open(folder/filename) as handle:
            lines = handle.readlines()
            item_ids = np.zeros(len(lines))
            text_vec = np.zeros((len(lines), 128))
            img_vec = np.zeros((len(lines), 128))
            for i, line in tqdm(list(enumerate(lines))):
                line = literal_eval(line)[1]
                item_ids[i] = line[0]
                text_vec[i] = line[1]
                img_vec[i] = line[2]
        return {"item_id": item_ids, "text_vec": text_vec, "img_vec": img_vec}
    
    if filetype == "click":
        header = ["user_id","item_id","time"]
        dataframe = pd.read_csv(folder/filename, sep=",", names=header)
        return dataframe
    elif filetype == "qtime":
        header = ["user_id", "time"]
        dataframe = pd.read_csv(folder/filename, sep=",", names=header)
        return dataframe
    elif filetype == "user":
        header = ["user_id","user_age_level","user_gender","user_city_level"]
        dataframe = pd.read_csv(folder/filename, sep=",", names=header)

        average_city_level = dataframe["user_city_level"].mean()
        average_city_level_map = {k:k for k in range(20)}
        average_city_level_map[None] = average_city_level
        dataframe["user_city_level"] = dataframe["user_city_level"].map(average_city_level_map)
        
        average_age = dataframe["user_age_level"].mean()
        average_age_map = {k:k for k in range(20)}
        average_age_map[None] = average_age
        dataframe["user_age_level"] = dataframe["user_age_level"].map(average_age_map)
        dataframe["user_gender"] = dataframe["user_gender"].map({"M": 1, "F":-1, None:0})

        dataset = {}
        dataset["user_id"] = np.array(dataframe["user_id"])
        dataset["user_age_level"] = np.array(dataframe["user_age_level"])
        dataset["user_gender"] = np.array(dataframe["user_gender"])
        dataset["user_city_level"] = np.array(dataframe["user_city_level"])

        return dataset

traintest_vals = ["train", "test"]
phases = [str(i) for i in range(7)]

def fix_timestamps(domain, dataset):
    """
        fixes timestamps for the dataframes such that 1 = 1 hour from the first timestamp
    """
    min_timestamp = 999
    length_of_day = 0.0000547 # got by staring at insights.ipynb
    length_of_hour = length_of_day/24

    print("normalizing timestamps")
    for traintest, phase, filetype in tqdm(product(traintest_vals, phases, ["click", "qtime"])):
        if traintest != "train" or filetype != "qtime":
            df = dataset[(traintest, phase, filetype)]
            min_timestamp = min(min_timestamp, min(df["time"]))

    # save timestamps to adjusted values
    for traintest, phase, filetype in tqdm(product(traintest_vals, phases, ["click", "qtime"])):
        if traintest != "train" or filetype != "qtime":
            df = dataset[(traintest, phase, filetype)]
            df["time"] = (df["time"]-min_timestamp)/length_of_hour

    return dataset

def build_contiguized_keymap(raw_ids, keymap=None, next_key=None):
    if keymap == None:
        keymap = {}

    initial_keymap_size = len(keymap)
    new_items = set(raw_ids) - set(keymap.keys())
    for new_item in new_items:
        if next_key == None:
            new_id = build_contiguized_keymap.next_id
            build_contiguized_keymap.next_id += 1
        else:
            new_id = next_key
            next_key += 1
        keymap[new_item] = new_id

    if 0 in new_items and new_items == set(range(len(new_items))):
        print("warning: dataset already contiguous")

    if 1 in new_items and  new_items == set(range(1, 1+len(new_items))):
        print("warning: dataset already contiguous starting at 1")

    contiguized_ids = [keymap[raw_id] for raw_id in raw_ids]

    return contiguized_ids, keymap
build_contiguized_keymap.next_id = 0

# def contiguize_keys(dataset, column, allow_missing=False):
#     for key, df in dataset.items():
#         if column not in df.columns:
#             if not allow_missing:
#                 raise Exception(f"column {column} not in dataframe {key}")
#             else:
#                 print(f"warning: column {column} not in dataframe {key}")

#     keymap = build_contiguized_keymap([df[column] for df in dataset.values() if column in df.columns], next_key=0)

#     for key, df in dataset.items():
#         if column in df.columns:
#             dataset[key][column] = df[column].map(keymap)

#     keymap = {v: k for k, v in keymap.items()}

#     return dataset, keymap

def contiguize_dataset_keys(domain, dataset):
    """
        contiguizing item and user keymaps such that each user_id and item_id could be used as an array index

        Arguments:
            domain {"china" or "international"}
            dataset {dict of dataframes}
    """
    item_keymap, next_item_key = {}, 0
    user_keymap, next_user_key = {}, 0

    for phase in phases:
        for traintest in ["train", "test"]:
            filetype = "click"
            df = dataset[(traintest, phase, filetype)]
            df["item_id"], item_keymap = build_contiguized_keymap(df["item_id"], item_keymap, next_key=next_item_key)
            df["user_id"], user_keymap = build_contiguized_keymap(df["user_id"], user_keymap, next_key=next_user_key)
        traintest = "test"
        filetype = "qtime"
        df = dataset[(traintest, phase, filetype)]
        df["user_id"], user_keymap = build_contiguized_keymap(df["user_id"], user_keymap, next_key=next_user_key)

    df = dataset["user"]
    df["user_id"], user_keymap = build_contiguized_keymap(df["user_id"], user_keymap, next_key=next_user_key)

    df = dataset["item"]
    df["item_id"], item_keymap = build_contiguized_keymap(df["item_id"], item_keymap, next_key=next_item_key)

    os.makedirs(Path("processed_data")/domain, exist_ok=True)
    with open(Path("processed_data")/domain/"user_keymap.pkl", "wb+") as handle:
        pickle.dump(user_keymap, handle)

    with open(Path("processed_data")/domain/"item_keymap.pkl", "wb+") as handle:
        pickle.dump(item_keymap, handle)

    return dataset

def group_by_user(dataset):
    print("grouping_by_user")
    output_dataset = {}
    for phase in tqdm(phases):
        for traintest in ["train", "test"]:
            filetype = "click"
            key = (traintest, phase, filetype)
            df = dataset[key]
            output_dataset[key] = {}
            for user_id, row in df.groupby("user_id"):
                row = row.sort_values(by="time")
                output_dataset[key][user_id] = np.array(row["item_id"]), np.array(row["time"])

        traintest = "test"
        filetype = "qtime"
        key = (traintest, phase, filetype)
        df = dataset[key]
        output_dataset[key] = {user_id: time for user_id, time in zip(df["user_id"], df["time"])}

    output_dataset["item"] = dataset["item"]
    output_dataset["user"] = dataset["user"]

    return output_dataset

def save_preprocessed_dataset(domain, dataset):
    """
        saving dataset

        Arguments:
            domain {"china" or "international"}
            dataset {dict of dataframes}
    """
    for traintest, phase, filetype in tqdm(product(traintest_vals, phases, ["click", "qtime"])):
        if traintest != "train" or filetype != "qtime":
            folder, new_filename = filename_from_args("processed_data", domain, traintest, phase, filetype, "pkl")
            os.makedirs(folder, exist_ok=True)
            with open(folder/new_filename, "wb+") as handle:
                pickle.dump(dataset[(traintest, phase, filetype)], handle)

    with open(Path("processed_data")/domain/"underexpose_train"/"underexpose_item_feat.pkl", "wb+") as handle:
        pickle.dump(get_raw_dataset(domain, "train", filetype="item"), handle)

    with open(Path("processed_data")/domain/"underexpose_train"/"underexpose_user_feat.pkl", "wb+") as handle:
        pickle.dump(get_raw_dataset(domain, "train", filetype="user"), handle)

def load_processed_dataset(domain):
    outputs = {}
    for filetype in ["item", "user"]:
        with open(Path("processed_data")/domain/"underexpose_test"/f"underexpose_{filetype}_feat.pkl") as handle:
            outputs[filetype] = pickle.load(handle)

    for filetype in ["click", "qtime"]:
        for phase in phases:
            for traintest in traintest_vals:
                folder, new_filename = filename_from_args("processed_data", domain, traintest, phase, filetype, "pkl")
                with open(folder/new_filename) as handle:
                    outputs[filetype] = pickle.load(handle)

    with open(Path("processed_data")/domain/"user_keymap.pkl", "rb") as handle:
        user_keymap = pickle.load(handle)

    with open(Path("processed_data")/domain/"item_keymap.pkl", "rb") as handle:
        item_keymap = pickle.load(handle)

    return outputs, (user_keymap, item_keymap)

if __name__ == "__main__":
    for domain in ["international", "china"]:
        print(f"starting preprocessing for domain {domain}")
        
        dataset = {}
        for traintest, phase, filetype in tqdm(product(traintest_vals, phases, ["click", "qtime"])):
            if traintest != "train" or filetype != "qtime":
                df = get_raw_dataset(domain=domain, traintest=traintest, phase=phase, filetype=filetype)
                dataset[(traintest, phase, filetype)] = df
        dataset["item"] = get_raw_dataset(domain=domain, traintest="train", filetype="item")
        dataset["user"] = get_raw_dataset(domain=domain, traintest="train", filetype="user")

        dataset = fix_timestamps(domain, dataset)

        dataset = contiguize_dataset_keys(domain, dataset)
        dataset = group_by_user(dataset)
        print("====================== train click ======================")
        print(dataset[("train", "0", "click")])
        print("====================== user ======================")
        print(dataset["user"])
        print("====================== item ======================")
        print(dataset["item"])
        print("====================== test qtime ======================")
        print(dataset["test", "0", "qtime"])
        save_preprocessed_dataset(domain, dataset)


# notes on data processing
    # 1. cleans nulls with average values. replaces strings with appropriate values.
    # 2. fixes timestamps
    # 3. normalzie user and item information