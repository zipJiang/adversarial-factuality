"""
"""
import json


with open('/brtx/603-nvme2/jzhan237/temp/dev_DUP3_mistral-inst.json', 'r', encoding='utf-8') as file_:
    data = json.load(file_)
    print(data[135])['parsed']