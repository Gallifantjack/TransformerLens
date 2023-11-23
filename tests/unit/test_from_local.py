import os
import pickle as pkl

import datasets
import pytest
import torch
from torch.utils.data import dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from transformer_lens import HookedTransformer, HookedTransformerConfig

# from transformer_lens.HookedTransformer import set_tokenizer
from transformer_lens.train import HookedTransformerTrainConfig, train
from transformer_lens.utils import tokenize_and_concatenate


# dummy weight conversion function
def convert_weights(state_dict, config):
    print(f"State dict type: {type(state_dict)}, Config type: {type(config)}")
    return state_dict

# create a pkl the same as created when pickle_dump= True in train function
def create_temp_pkl(model= "NeelNanda/Attn_Only_1L512W_C4_Code"):
    # create a temp pickle file
    ref_attn_only = HookedTransformer.from_pretrained(model)
    pickle_path = "temp_{model}.pkl"
    with open(pickle_path, "wb") as f:
        pkl.dump(ref_attn_only, f)
    return pickle_path

# create a state_dict the same as in current training at checkpoints
def create_temp_state_dict(model):
    torch.save(model.state_dict(), "temp_state_dict.pt")
    return "temp_state_dict.pt"

# Check if dumping model to pickle permits reloading of model
def test_model_reloading_from_pickle():
    pickle_file_path = create_temp_pkl()
    
    assert os.path.exists(
        pickle_file_path
    ), "Pickle file not found for the trained model"

    with open(pickle_file_path, "rb") as f:
        pkl_model = pkl.load(f)

    assert (
        pkl_model.generate("Barack Obama ") is not None
    ), "Model failed to generate text after reloading from pickle"

# check if from_local function and weight conversion function works
def test_from_local_functionality():
    pickle_file_path = create_temp_pkl()
    
    with open(pickle_file_path, "rb") as f:
        pkl_model = pkl.load(f)
    pkl_config = pkl_model.cfg
    
    model_state_dict_path = create_temp_state_dict(pkl_model)
    
    local_model = HookedTransformer.from_local(
        model_state_dict_path, pkl_config, convert_weights
    )

    assert local_model is not None
    assert (
        local_model.generate("Barack Obama") is not None
    ), "Model failed to generate text after loading from local"
