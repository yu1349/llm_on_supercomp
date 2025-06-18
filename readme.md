# 名古屋大学スパコン「不老」でのLLMの利用
## 概要
このリポジトリは「不老」上での動作を想定しています。研究室GPUサーバでの動作確認はできていません。ごめんなさい。

## 目次
- [ディレクトリ構成](#ディレクトリ構成)
- [環境設定](#環境設定)
- [コードの実行](#コードの実行)

## ディレクトリ構成
```
.
|-data   :実験データ
  |-metainfo_your_sample.jsonl
|-env    :環境
  |-llm_on_supercomp.yml
|-exp    :実験の実装
  |-inference_on_your_setting.py
  |-inference_on_your_setting.sh
|-output :実験での出力
  |-your_model.pkl
```

## 環境設定
デバイス系のモジュールはスパコンで、Python系のライブラリはcondaで構築します。
```
# anaconda
## スパコンでのcondaの有効化
eval "$(~/miniconda3/bin/conda shell.bash hook)"
## conda環境のロード
conda env create -n your_env -f llm_on_supercomp.yml
conda activate your_env

# デバイス系
module load gcc/11.3.0 cuda/12.4.1 openmpi_cuda/4.0.5 nccl/2.19.3
```
環境ファイルを使用したconda環境の構築はエラーを吐きやすいので、失敗したら教えてください。

## コードの実行
*.pyはpythonスクリプトなので、*.shのバッチジョブスクリプトを実行してください。
```
pjsub inference_on_your_setting.sh
```