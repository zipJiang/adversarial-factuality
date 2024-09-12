"""
"""

import ujson as json
from scipy.stats import pearsonr, spearmanr


def main():
    """
    """

    paths = [
        # "/brtx/606-nvme2/zpjiang/adversarial-factuality/data/scores/nli_comparison/DeBERTa-v3-base-mnli-fever-anli.json",
        "/brtx/606-nvme2/zpjiang/adversarial-factuality/data/scores/nli_comparison/DeBERTa-v3-large-mnli-fever-anli-ling-wanli.json",
        "/brtx/606-nvme2/zpjiang/adversarial-factuality/data/scores/nli_comparison/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli.json"
    ]

    scores = []
    
    for path in paths:
        with open(path, "r") as f:
            scores.append([item['score'] for item in json.load(f)])
            
    print(pearsonr(scores[0], scores[1])[0])
    print(spearmanr(scores[0], scores[1])[0])
    
    
if __name__ == "__main__":
    main()
