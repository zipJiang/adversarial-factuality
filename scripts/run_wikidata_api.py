"""This script attempt to do basic callings
of wikidata api for feature verification.
"""

import requests


def fetch_wikipedia_titles(page_names):
    """
    Fetches information for multiple Wikipedia pages in a single API call.

    Args:
    page_names (list of str): A list of page titles to query.

    Returns:
    dict: A dictionary with page titles and their information.
    """
    # Join page names into a single string separated by '|'
    titles = '|'.join(page_names)

    # Base URL for the Wikipedia API
    base_url = 'https://en.wikipedia.org/w/api.php'

    # Parameters for the API call
    params = {
        'action': 'query',
        'titles': titles,
        'format': 'json',
        'prop': 'info',  # Requesting basic page info; you can adjust this as needed
    }

    # Making the API call
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': 'Failed to fetch data, status code: {}'.format(response.status_code)}

# Example usage
page_names = ['Albert_einstein', 'Artificial_intelligence', 'Python_(programming_language)', 'Machine_learning']
result = fetch_wikipedia_titles(page_names)
print(result)