"""
"""

from typing import Text, Dict, Any, Tuple, List
import os
import sqlite3
import pickle
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import logging
import numpy as np
from overrides import overrides
from .task import Task


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


__INTERNAL_QUERY_BATCH_SIZE__ = 100
__INTERNAL_QUERY_BATCH_SIZE_WIKI__ = 50


def fetch_wikidata_ids(dbpedia_url):
    # SPARQL endpoint for DBpedia
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    wikidata_ids = []

    query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?wikidataID WHERE {{
        <{dbpedia_url}> owl:sameAs ?wikidataID .
        FILTER(strstarts(str(?wikidataID), "http://www.wikidata.org/entity/"))
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Extracting the Wikidata ID from the results
    for result in results["results"]["bindings"]:
        wikidata_url = result["wikidataID"]["value"]
        wikidata_id = wikidata_url.split("/")[-1]  # Extract ID from URL
        wikidata_ids.append(wikidata_id)

    return wikidata_ids


def fetch_wikidata_ids_in_batch(dbpedia_urls):

    values_clause = (
        "VALUES ?dbpediaResource { "
        + " ".join([f"<{resource}>" for resource in dbpedia_urls])
        + " }"
    )
    query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?dbpediaResource ?wikidataID WHERE {{
        {values_clause}
        ?dbpediaResource owl:sameAs ?wikidataID .
        FILTER(strstarts(str(?wikidataID), "http://www.wikidata.org/entity/"))
    }}
    """

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    resource_to_wikidata = {}

    for result in results["results"]["bindings"]:
        dbpedia_url = result["dbpediaResource"]["value"]
        wikidata_url = result["wikidataID"]["value"]
        wikidata_id = wikidata_url.split("/")[-1]

        if dbpedia_url not in resource_to_wikidata:
            resource_to_wikidata[dbpedia_url] = []

        resource_to_wikidata[dbpedia_url].append(wikidata_id)

    return [resource_to_wikidata.get(resource, []) for resource in dbpedia_urls]


def process_dpentity_dump(entry):
    """ """
    possible_entries = fetch_wikidata_ids(entry)
    if len(possible_entries) != 1:
        raise ValueError(f"Multiple entries found for {entry}")

    return {
        "entity_name": entry,
        "entity_id": possible_entries[0],
    }


def process_dpentity_dump_batch(entries):

    possible_entries_list = fetch_wikidata_ids_in_batch(entries)

    filtered = []
    for pidx in range(len(entries)):
        if len(possible_entries_list[pidx]) != 1:
            logger.info(f"Multiple entries found for {entries[pidx]}")
        else:
            filtered.append(
                {
                    "entity_name": entries[pidx],
                    "entity_id": possible_entries_list[pidx][0],
                }
            )

    return filtered


def check_if_human(entity_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={entity_id}&property=P31&format=json"
    response = requests.get(url)
    data = response.json()

    # Extracting the claim data
    claims = data.get("claims", {}).get("P31", [])

    for claim in claims:
        if claim["mainsnak"]["datavalue"]["value"]["id"] == "Q5":
            return entity_id, True
    return entity_id, False


def check_if_human_in_batch(entity_ids):
    """ """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    values_clauses = " ".join([f"wd:{entity_id}" for entity_id in entity_ids])

    query = f"""
    SELECT ?item ?itemLabel WHERE {{
        VALUES ?item {{ {values_clauses} }}
        ?item wdt:P31 wd:Q5.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    result_entities = {
        result["item"]["value"].split("/")[-1]: True
        for result in results["results"]["bindings"]
    }
    response = {
        entity_id: result_entities.get(entity_id, False) for entity_id in entity_ids
    }

    return [entity_id for entity_id, is_human in response.items() if is_human]


def get_nationality_in_batch(entity_ids):
    """
    """
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    values_clausese = " ".join([f"wd:{entity_id}" for entity_id in entity_ids])
    
    query = f"""
    SELECT ?item ?itemLabel ?nationalityLabel ?continentLabel WHERE {{
        VALUES ?item {{ {values_clausese} }}
        ?item wdt:P27 ?nationality.
        ?nationality wdt:P30 ?continent.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    nationalities = {}
    
    for result in results['results']['bindings']:
        entity_id = result["item"]["value"].split("/")[-1]
        nationality = result['nationalityLabel']['value']
        continent = result['continentLabel']['value']
        nationalities[entity_id] = {
            "nationality": nationality,
            "continent": continent,
        }
        
    return  {entity_id: nationalities.get(entity_id, {"nationality": None, "continent": None}) for entity_id in entity_ids}


def get_wikipedia_pages_in_batch(entity_ids):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Prepare the SPARQL query
    entity_list = " ".join("wd:{}".format(entity) for entity in entity_ids)
    query = f"""
    SELECT ?item ?article WHERE {{
        VALUES ?item {{ {entity_list} }}
        ?article schema:about ?item .
        ?article schema:isPartOf <https://en.wikipedia.org/> .
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Process the results
    pages = {}
    for result in results["results"]["bindings"]:
        pages[result["item"]["value"].split("/")[-1]] = result["article"]["value"]

    for entity_id in entity_ids:
        if entity_id not in pages:
            pages[entity_id] = None

    return pages


def query_for_normalized_titles_in_batch(titles):
    """We can extract unnormalized titles from
    the Wikipedia URL, but we need the api to
    get the normalized titles.
    """
    time.sleep(0.1)

    # Join page names into a single string separated by '|'
    titles_query = "|".join(titles)

    # Base URL for the Wikipedia API
    base_url = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API call
    params = {
        "action": "query",
        "titles": titles_query,
        "format": "json",
        "prop": "info",  # Requesting basic page info; you can adjust this as needed
    }

    # Making the API call
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # return response.json()
        item_conversion_dict = {
            item["from"]: item["to"] for item in response.json()["query"]["normalized"]
        }
        results = []
        for title in titles:
            if title not in item_conversion_dict:
                results.append(title.replace("_", " "))
            else:
                results.append(item_conversion_dict[title])
        return results
    else:
        raise ValueError(f"Failed to fetch data, status code: {response.status_code}")


@Task.register("filter-required-entities")
class FilterRequiredEntities(Task):
    def __init__(
        self,
        popqa_path: Text,
        wikientity_path: Text,
        wikipedia_dump_path: Text,
        protected_entity_path: Text,
        output_dir: Text,
    ):
        super().__init__()
        self._popqa_path = popqa_path
        self._wikientity_path = wikientity_path
        self._wikipedia_dump_path = wikipedia_dump_path
        self._protected_entity_path = protected_entity_path

        self._output_dir = output_dir

    @overrides
    def run(self):

        # first let us load the wikidata entities

        def _filter_is_human(items: List[Text]):
            """ """
            results = set()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(
                        check_if_human_in_batch,
                        items[vidx : vidx + __INTERNAL_QUERY_BATCH_SIZE__],
                    )
                    for vidx in range(0, len(items), __INTERNAL_QUERY_BATCH_SIZE__)
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Checking if is human.",
                ):
                    try:
                        res = future.result()
                        for r in res:
                            results.add(r)
                    except Exception as e:
                        logger.error(f"Error: {e}")

            logger.info(f"Number of entities after is_human filtering: {len(results)}")

            return results

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        if not os.path.exists(
            os.path.join(self._output_dir, "processed_popqa_entities.pkl")
        ):
            popqa_entities = self._process_popqa()
            with open(
                os.path.join(self._output_dir, "processed_popqa_entities.pkl"), "wb"
            ) as f:
                pickle.dump(popqa_entities, f)
        else:
            with open(
                os.path.join(self._output_dir, "processed_popqa_entities.pkl"), "rb"
            ) as f:
                popqa_entities = pickle.load(f)

        if not os.path.exists(
            os.path.join(self._output_dir, "filtered_popqa_entities.pkl")
        ):
            filtered_keys = _filter_is_human(
                [item["entity_id"] for item in popqa_entities]
            )
            filtered_popqa_entities = [
                item for item in popqa_entities if item["entity_id"] in filtered_keys
            ]
            with open(
                os.path.join(self._output_dir, "filtered_popqa_entities.pkl"), "wb"
            ) as f:
                pickle.dump(filtered_popqa_entities, f)
        else:
            with open(
                os.path.join(self._output_dir, "filtered_popqa_entities.pkl"), "rb"
            ) as f:
                filtered_popqa_entities = pickle.load(f)

        if not os.path.exists(
            os.path.join(self._output_dir, "processed_wikidata_entities.pkl")
        ):
            wdump_dict, wikidata_entities = self._process_wikientity_dump()
            with open(
                os.path.join(self._output_dir, "processed_wikidata_entities.pkl"), "wb"
            ) as f:
                pickle.dump(wikidata_entities, f)
        else:
            with open(
                os.path.join(self._output_dir, "processed_wikidata_entities.pkl"), "rb"
            ) as f:
                wikidata_entities = pickle.load(f)
                wdump_dict = np.load(self._wikientity_path)

        if not os.path.exists(
            os.path.join(self._output_dir, "filtered_wikidata_entities.pkl")
        ):
            filtered_keys = _filter_is_human(
                [item["entity_id"] for item in wikidata_entities]
            )
            filtered_wikidata_entities = [
                item for item in wikidata_entities if item["entity_id"] in filtered_keys
            ]

            with open(
                os.path.join(self._output_dir, "filtered_wikidata_entities.pkl"), "wb"
            ) as f:
                pickle.dump(filtered_wikidata_entities, f)

        else:
            with open(
                os.path.join(self._output_dir, "filtered_wikidata_entities.pkl"), "rb"
            ) as f:
                filtered_wikidata_entities = pickle.load(f)

        # extract all filtered_wikidata_entities_id

        if not os.path.exists(
            os.path.join(self._output_dir, "dedup_wikidata_entities_id.pkl")
        ):
            wikidata_entity_id_to_freq = {
                item["entity_id"]: wdump_dict[item["entity_name"]].size
                for item in tqdm(filtered_wikidata_entities)
            }
            with open(
                os.path.join(self._output_dir, "dedup_wikidata_entities_id.pkl"),
                "wb",
            ) as f:
                pickle.dump(wikidata_entity_id_to_freq, f)

        else:
            with open(
                os.path.join(self._output_dir, "dedup_wikidata_entities_id.pkl"),
                "rb",
            ) as f:
                wikidata_entity_id_to_freq = pickle.load(f)

        del wdump_dict

        # combining frequency information
        if not os.path.exists(
            os.path.join(self._output_dir, "dedup_popqa_entities_id.pkl")
        ):
            popqa_entity_to_freq = {
                item["entity_id"]: item for item in filtered_popqa_entities
            }
            with open(
                os.path.join(self._output_dir, "dedup_popqa_entities_id.pkl"), "wb"
            ) as f:
                pickle.dump(popqa_entity_to_freq, f)
        else:
            with open(
                os.path.join(self._output_dir, "dedup_popqa_entities_id.pkl"), "rb"
            ) as f:
                popqa_entity_to_freq = pickle.load(f)

        # combine these two

        if not os.path.exists(os.path.join(self._output_dir, "raw_stats.pkl")):
            stats = {
                key: {
                    "entity_id": key,
                    "wikidata_freq": val,
                    "popqa_freq": (
                        popqa_entity_to_freq[key]["freq"]
                        if key in popqa_entity_to_freq
                        else -1
                    ),
                    "popqa_entity_name": (
                        popqa_entity_to_freq[key]["entity_name"]
                        if key in popqa_entity_to_freq
                        else "[N/A]"
                    ),
                }
                for key, val in tqdm(wikidata_entity_id_to_freq.items())
            }

            # also add popqa entities that are not in wikidata
            in_stats_count = 0
            not_in_stats_count = 0
            for key, val in popqa_entity_to_freq.items():
                if key not in stats:
                    stats[key] = {
                        "entity_id": key,
                        "wikidata_freq": -1,
                        "popqa_freq": val["freq"],
                        "popqa_entity_name": val["entity_name"],
                    }
                    not_in_stats_count += 1
                else:
                    in_stats_count += 1

            logger.info(f"Number of popqa entities in stats: {in_stats_count}")
            logger.info(f"Number of popqa entities not in stats: {not_in_stats_count}")

            with open(os.path.join(self._output_dir, "raw_stats.pkl"), "wb") as f:
                pickle.dump(stats, f)
        else:
            with open(os.path.join(self._output_dir, "raw_stats.pkl"), "rb") as f:
                stats = pickle.load(f)

        def _attach_wikipedia_page(stats):
            entity_ids = list(stats.keys())

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(
                        get_wikipedia_pages_in_batch,
                        entity_ids[kidx : kidx + __INTERNAL_QUERY_BATCH_SIZE__],
                    )
                    for kidx in range(0, len(entity_ids), __INTERNAL_QUERY_BATCH_SIZE__)
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Fetching wikipedia pages.",
                ):
                    try:
                        pages = future.result()
                        for entity_id, page in pages.items():
                            stats[entity_id]["wikipedia_page"] = page
                    except Exception as e:
                        logger.error(f"Error: {e}")

            return stats

        # attach wikipedia page to the stats
        if not os.path.exists(
            os.path.join(self._output_dir, "stats_with_wikipedia_pages.pkl")
        ):
            stats = _attach_wikipedia_page(stats)
            with open(
                os.path.join(self._output_dir, "stats_with_wikipedia_pages.pkl"), "wb"
            ) as f:
                pickle.dump(stats, f)

        else:
            with open(
                os.path.join(self._output_dir, "stats_with_wikipedia_pages.pkl"), "rb"
            ) as f:
                stats = pickle.load(f)

        # finally, find the title of the wikipedia page using the wikipedia api
        def _attach_wikipedia_titles(stats):
            # There's further filtering that items without wikipedia page is removed.
            stats = {
                key: val
                for key, val in stats.items()
                if "wikipedia_page" in val and val["wikipedia_page"] is not None
            }
            entity_names = [
                (val["entity_id"], val["wikipedia_page"].split("/")[-1])
                for val in stats.values()
            ]

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures_to_entity_ids = {
                    executor.submit(
                        query_for_normalized_titles_in_batch,
                        [
                            e[1]
                            for e in entity_names[
                                kidx : kidx + __INTERNAL_QUERY_BATCH_SIZE_WIKI__
                            ]
                        ],
                    ): [
                        e[0]
                        for e in entity_names[
                            kidx : kidx + __INTERNAL_QUERY_BATCH_SIZE_WIKI__
                        ]
                    ]
                    for kidx in range(
                        0, len(entity_names), __INTERNAL_QUERY_BATCH_SIZE_WIKI__
                    )
                }

                for future in tqdm(
                    as_completed(futures_to_entity_ids),
                    total=len(futures_to_entity_ids),
                    desc="Fetching wikipedia titles.",
                ):
                    try:
                        titles = future.result()
                        # print(titles)
                        for entity_id, title in zip(
                            futures_to_entity_ids[future], titles
                        ):
                            stats[entity_id]["wikipedia_title"] = title
                    except Exception as e:
                        logger.error(f"Error: {e}")

            return stats

        if not os.path.exists(
            os.path.join(self._output_dir, "stats_with_wikipedia_titles.pkl")
        ):
            stats = _attach_wikipedia_titles(stats)
            with open(
                os.path.join(self._output_dir, "stats_with_wikipedia_titles.pkl"), "wb"
            ) as f:
                pickle.dump(stats, f)

        else:
            with open(
                os.path.join(self._output_dir, "stats_with_wikipedia_titles.pkl"), "rb"
            ) as f:
                stats = pickle.load(f)
                
        # Now try to attach nationality and part of items in the stats
        
        def _attach_nationalities(stats):
            """
            """
            entity_ids = list(stats.keys())
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(
                        get_nationality_in_batch,
                        entity_ids[kidx : kidx + __INTERNAL_QUERY_BATCH_SIZE__],
                    )
                    for kidx in range(0, len(entity_ids), __INTERNAL_QUERY_BATCH_SIZE__)
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Attaching nationalities.",
                ):
                    try:
                        pages = future.result()
                        for entity_id, nationality_dict in pages.items():
                            stats[entity_id].update(nationality_dict)
                    except Exception as e:
                        logger.error(f"Error: {e}")

            return stats
        
        if not os.path.exists(
            os.path.join(self._output_dir, "stats_with_nationalities.pkl")
        ):
            stats = _attach_nationalities(stats)
            with open(os.path.join(
                self._output_dir, "stats_with_nationalities.pkl"), "wb"
            ) as f:
                pickle.dump(stats, f)

        else:
            with open(os.path.join(
                self._output_dir, "stats_with_nationalities.pkl"), "rb"
            ) as f:
                stats = pickle.load(f)
        

        # Now try to find the entities that are findable in the wikipedia dump
        # if not os.path.exists(os.path.join(self._output_dir), "stats_with_wikipedia_dump_availabilities.pkl"):
        #     self._check_against_dump(stats)
        # else:
        #     with open(os.path.join(self._output_dir, "stats_with_wikipedia_dump_availabilities.pkl"), "rb") as f:
        #         stats = pickle.load(f)
        if not os.path.exists(
            os.path.join(
                self._output_dir,
                "stats_with_wikipedia_dump_availabilities.pkl",
            )
        ):
            stats = self._check_against_dump(stats)
            with open(
                os.path.join(
                    self._output_dir,
                    "stats_with_wikipedia_dump_availabilities.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(stats, f)
        else:
            with open(
                os.path.join(
                    self._output_dir, "stats_with_wikipedia_dump_availabilities.pkl"
                ),
                "rb",
            ) as f:
                stats = pickle.load(f)

        # final step, check against the protected entities (already presented in the dump to avaoid leakage)
        if not os.path.exists(
            os.path.join(
                self._output_dir, "stats_checked_against_protected_entities.pkl"
            )
        ):
            with open(self._protected_entity_path, "r", encoding="utf-8") as f:
                protected_entities = set(
                    [item.strip().lower() for item in f.read().split("\n")]
                )

                already_selected_count = 0
                for key, val in stats.items():
                    val["adjusted_freq"] = (
                        max(val["wikidata_freq"], val["popqa_freq"])
                        if val["wikidata_freq"] != -1 and val["popqa_freq"] != -1
                        else -1
                    )
                    val["adjusted_freq_source"] = (
                        (
                            "wikidata"
                            if val["wikidata_freq"] > val["popqa_freq"]
                            else "popqa"
                        )
                        if val["wikidata_freq"] != -1 and val["popqa_freq"] != -1
                        else ("wikidata" if val["wikidata_freq"] != -1 else "popqa")
                    )
                    if val["wikipedia_title"].strip().lower() in protected_entities:
                        val["already_selected"] = True
                        already_selected_count += 1
                    else:
                        val["already_selected"] = False
                        
                stats = list(stats.values())

                logger.info(
                    f"Number of already selected entities: {already_selected_count}"
                )

            with open(
                os.path.join(
                    self._output_dir, "stats_checked_against_protected_entities.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(stats, f)

        else:
            with open(
                os.path.join(
                    self._output_dir, "stats_checked_against_protected_entities.pkl"
                ),
                "rb",
            ) as f:
                stats = pickle.load(f)

    def _check_against_dump(self, stats):

        connection = sqlite3.connect(self._wikipedia_dump_path, check_same_thread=False)
        cursor = connection.cursor()
        results = cursor.execute("SELECT title FROM documents ;")
        results = cursor.fetchall()

        set = {str(r[0]) for r in tqdm(results, desc="Building Title set.")}

        in_dump_count = 0
        in_dump_and_popQA_count = 0
        for sk in tqdm(stats.keys(), desc="Checking against dump."):
            stats[sk]["is_in_dump"] = stats[sk]["wikipedia_title"] in set
            if stats[sk]["is_in_dump"]:
                in_dump_count += 1
                if stats[sk]["popqa_freq"] != -1:
                    in_dump_and_popQA_count += 1

        logger.info(f"Number of entities in dump: {in_dump_count}")
        logger.info(f"Number of entities in dump and popqa: {in_dump_and_popQA_count}")

        return stats

    def _process_wikientity_dump(self):
        """ """

        def _filter_non_ambiguous(item_dict: Dict[Text, Any]):
            results = []

            item_urls = list(item_dict.keys())

            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [
                    executor.submit(
                        process_dpentity_dump_batch,
                        item_urls[kidx : kidx + __INTERNAL_QUERY_BATCH_SIZE__],
                    )
                    for kidx in range(0, len(item_urls), __INTERNAL_QUERY_BATCH_SIZE__)
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Checking if only one entries.",
                ):
                    try:
                        processed = future.result()
                        results.extend(processed)
                    except Exception as e:
                        logger.error(f"Error: {e}")

            logger.info(f"Number of entities after valid filtering: {len(results)}")
            return results

        wdump_dict = np.load(self._wikientity_path)

        return wdump_dict, _filter_non_ambiguous(wdump_dict)

    def _process_popqa(self):

        popqa_entities = []
        checked_entites = set()

        with open(self._popqa_path, "r", encoding="utf-8") as f:
            popqa = csv.DictReader(f, delimiter="\t")

            for row in tqdm(popqa):
                subj = row["subj"]
                subj_id = row["s_uri"].split("/")[-1]
                if subj_id not in checked_entites:
                    checked_entites.add(subj_id)
                    popqa_entities.append(
                        {
                            "freq": int(row["s_pop"]),
                            "entity_name": subj,
                            "entity_id": subj_id,
                        }
                    )

                obj = row["obj"]
                obj_id = row["o_uri"].split("/")[-1]
                if obj_id not in popqa_entities:
                    popqa_entities.append(
                        {
                            "freq": int(row["o_pop"]),
                            "entity_name": obj,
                            "entity_id": obj_id,
                        }
                    )

        logger.info(f"Number of entities in PopQA: {len(popqa_entities)}")

        return popqa_entities
