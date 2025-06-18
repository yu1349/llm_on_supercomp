# 汎用
import argparse
import pickle
from util import read_jsonl

# ファイル関係
import os
import json
import glob

# 機械学習関係
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import infer_auto_device_map

# メタ情報抽出データ関係
from metainfo_extract import metainfo_generator_fn, create_inst
from datasets import Dataset

def intialize_tokenizer_and_model(model_name_or_path:str):
    try:
        # トークナイザーの初期化
        ## DecoderはPADがないため、EOSで代用
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side='right'
        # モデルの初期化
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
            )
    except:
        print("Tokenizer & Model Initialize Error")
        return
    # 学習をしない
    model.eval()
    return tokenizer, model

def main(args):
    print("Start")

    # device_map="auto"の挙動を確認
    tokenizer, model = intialize_tokenizer_and_model(args.model_name_or_path)

    device_map = infer_auto_device_map(
        model, 
        max_memory={0: "32GiB", 1: "32GiB", 2: "32GiB", 3: "32GiB"}
        )
    # ちなみに出力のdevice_mapをモデルのロードに渡しても良かったはず
    print(device_map)

    print("Finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--input_data_path', type=str)

    parser.add_argument('--output_dir', type=str)

    args, unk = parser.parse_known_args()

    main(args)