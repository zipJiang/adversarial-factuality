"""
"""


import os
import sqlite3
from unittest import TestCase
from src.utils.query_serper import CachedSerperAPI


class TestCachedSerper(TestCase):
    def setUp(self):
        self.serper_api = CachedSerperAPI(
            serper_api_key=os.environ['SERPER_API_KEY'],
            cache_path=".cache/serper/test_serper_cache.db",
            k=3,
        )
        
    def test_single_api_query(self):
        """
        """

        result_3 = self.serper_api("Kalki Koechlin acting credits and biography")
        print(result_3)
        
        self.serper_api.k = 5
        result_5 = self.serper_api("Kalki Koechlin acting credits and biography")
        print(result_5)
        
        conn = sqlite3.connect(".cache/serper/serper_cache.db")
        cursor = conn.cursor()
        cursor.execute("SELECT result FROM cache WHERE query = ? AND k = ?", ("Kalki Koechlin acting credits and biography", 3))
        row = cursor.fetchone()
        
        self.assertEqual(result_3, row[0])
        
        cursor.execute("SELECT result FROM cache WHERE query = ? AND k = ?", ("Kalki Koechlin acting credits and biography", 5))
        row = cursor.fetchone()
        self.assertEqual(result_5, row[0])