"""This script attempt to do basic callings
of wikidata api for feature verification.
"""

import requests

def check_if_human(entity_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={entity_id}&property=P31&format=json"
    response = requests.get(url)
    data = response.json()
    
    # Extracting the claim data
    claims = data.get('claims', {}).get('P31', [])
    
    for claim in claims:
        if claim['mainsnak']['datavalue']['value']['id'] == 'Q5':
            return True
    return False

# Example usage
is_human = check_if_human('Q6247345')  # Albert Einstein
print("Is human:", is_human)