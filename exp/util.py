import json

def read_jsonl(file_path):
    '''
    関数: jsonlファイルを読み込む関数
    引数: ファイルパス
    '''
    jsonl_data = []
    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            jsonl_data.append(json.loads(line))
    return jsonl_data

def read_ids(file_path, check_type):
    '''
    関数: \nのファイルを読み込む関数
    引数: ファイルパス, ファイル中身の型
    '''
    split_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        split_ids = file.readlines()
    if check_type == int:
        split_ids = [int(split_id) for split_id in split_ids]
    return split_ids

def writeout_jsonl(output_path, json_data):
    '''
    関数: list[json]のデータをファイルに書き込む関数
    引数: ファイルパス, list[json]のデータ
    '''
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for row_data in json_data:
            json.dump(row_data, output_file)
            output_file.write('\n')