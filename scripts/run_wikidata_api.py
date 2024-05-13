"""This script attempt to do basic callings
of wikidata api for feature verification.
"""

import requests
from SPARQLWrapper import SPARQLWrapper, JSON


def get_nationalities(entity_ids):

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    values_clause = " ".join(f"wd:{id}" for id in entity_ids)
    query = f"""
    SELECT ?entity ?entityLabel ?nationalityLabel ?continentLabel WHERE {{
        VALUES ?entity {{ {values_clause} }}
        ?entity wdt:P27 ?nationality.
        ?nationality wdt:P30 ?continent.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    print(results)

    nationalities = []
    for result in results["results"]["bindings"]:
        entity = result["entity"]["value"]
        entity_label = result["entityLabel"]["value"]
        nationality = result["nationalityLabel"]["value"]
        continent = result["continentLabel"]["value"]
        nationalities.append({"entity": entity, "nationality": nationality, "continent": continent})

    return nationalities

# Example usage for multiple entities
print(get_nationalities(['Q42', 'Q8023']))  # Replace with desired Wikidata entity IDs