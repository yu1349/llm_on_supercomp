import json
import torch

class MetaInfoDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
def metainfo_generator_fn(texts):
    for text in texts:
        yield {"text": text}

def create_inst(jsonl_data: list[dict], tokenizer) -> list[str]:
    texts: list[str] = []

    # 自分でinstを作ってください
    task_def = "Your task is to extract meta-information about research data cited by the given URL from section title, body text and footnote/reference."

    # ループでchat_apply_template適応
    for test_id, json_data in enumerate(jsonl_data):    
        messages = []

        # 各要素の取得
        ## ここ自分のデータセットと併せてください
        ## メタ情報抽出データセットの場合、URL, SectionTitle, BodyText, FootnoteReferenceが必要
        url_txt = f"Given URL: {json_data['url']}"
        sec_title_txt = f"Section Title: {json_data['text'].split('[SectionTitle]')[-1].split('[Body]')[0].strip()}"
        body_txt = f"Body Text: {json_data['text'].split('[Body]')[-1].split('[FootnoteReference]')[0].strip()}"
        foot_ref_txt = f"Footnote or Reference Text: {json_data['text'].split('[FootnoteReference]')[-1].strip()}"

        # 文字列として結合
        user_txt = task_def+'\n'+url_txt+'\n'+sec_title_txt+'\n'+body_txt+'\n'+foot_ref_txt
        
        # 対話形式に格納
        ## ローカルLLMはsystemないモデルもあるので、気を付けて
        messages.append({"role": "user", "content": user_txt})

        # 各モデルのテンプレートを適応
        ## toolsとかfunctionとかでjson出力が指定できる
        ## 引数がモデルによって違うので、自分でchat_templateを読んでください
        instruction = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            # tools=[tool_to_dict()], 
            add_generation_prompt=True,
            # enable_thinking=False
            )
        
        texts.append(instruction)

    return texts