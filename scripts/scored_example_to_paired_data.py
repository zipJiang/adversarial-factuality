'''
convert a set up (prompt, response, score) pair to paired data of (prompt, chosen response, rejected response) pairs by finding responses that have the same prompt but different scores (response with higher score is chosen response, response with lower score is rejected response).
'''

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to the input file')
    parser.add_argument('-o', '--output', type=str, help='path to the output file')
    parser.add_argument('-f', '--full_prompt_func', type=str, default="<s>[INST] Tell me a bio of {}. [/INST]", help='e.g. "<s>[INST] Tell me a bio of {}. [/INST]", the {} will be replaced with the original prompt')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    paired_data = []
    for idx, item in enumerate(data):
        prompt = item['topic']
        full_prompt = args.full_prompt_func.replace("{}", prompt)
        response = item['output']['parsed']
        score = item['score']
        for jdx, jtem in enumerate(data):
            if jdx <= idx:
                continue
            if jtem['topic'] == prompt:
                if jtem['score'] > score:
                    paired_data.append({
                        "full_prompt": full_prompt,
                        "gold": ["N/A"],
                        "chosen": jtem['output']['parsed'],
                        "rejected": response,
                        "chosen_score": jtem['score'],
                        "rejected_score": score
                    })
                elif jtem['score'] < score:
                    paired_data.append({
                        "full_prompt": full_prompt,
                        "gold": ["N/A"],
                        "chosen": response,
                        "rejected": jtem['output']['parsed'],
                        "chosen_score": score,
                        "rejected_score": jtem['score']
                    })

    j = {
        'args': vars(args),
        'data': paired_data
    }

    with open(args.output, "w") as f:
        json.dump(j, f, indent=4)
