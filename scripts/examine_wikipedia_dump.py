"""There could be more data in the sqlite dataset than
the one already presented in the file. We may want to
sample from there and generate.
"""

import sqlite3


db_path = "db/enwiki-20230401.db"

con = sqlite3.connect(db_path, check_same_thread=False)
cursor = con.cursor()

# print all fieldnames from documents
res = cursor.execute("PRAGMA table_info(documents);").fetchall()
print(res)
# res = cursor.execute("SELECT title FROM documents;").fetchall()