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

    # 入力データの読み込み
    jsonl_data = read_jsonl(args.input_data_path)
    print("input_data_size:::", len(jsonl_data))
    print("input_sample:::", jsonl_data[0])

    # モデルとトークナイザーの初期化
    ## 本実装では読み込みの際はCPUにロードする -> pipe作成時にGPUへ
    tokenizer, model = intialize_tokenizer_and_model(args.model_name_or_path)

    # プロンプトの作成
    input_texts = create_inst(jsonl_data, tokenizer)
    input_dataset = Dataset.from_generator(lambda: metainfo_generator_fn(input_texts))
    print("input_prompt:::", input_dataset.__getitem__(0))

    # パイプラインの作成
    pipe = pipeline(
        task="text-generation",                 # LLMはtext-generationで基本的にOK
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,                        # 貪欲法 (temeperature=0)
        device=0,                               # GPU1枚なのでcuda:0を指定
        )
    
    # 推論
    outs = pipe(input_dataset['text'], truncation=True, padding=True, max_length=512)

    # 標準出力
    for out in outs:
        print(out)

    # 出力の保存
    model_name = os.path.basename(args.model_name_or_path)
    output_save_dir = f"../output"
    os.makedirs(output_save_dir, exist_ok=True)
    print(output_save_dir)
    with open(f"{output_save_dir}/{model_name}.pkl", 'wb') as output_file:
        pickle.dump(outs, output_file)

    print("Finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--input_data_path', type=str)

    parser.add_argument('--output_dir', type=str)

    args, unk = parser.parse_known_args()

    main(args)