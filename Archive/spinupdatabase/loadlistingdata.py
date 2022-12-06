#import the libraries

import pandas
import sqlite3

listing = pandas.read_excel("dataset/airbnblisting.xlsx",sheet_name="listingreviews", header=0)

db_conn = sqlite3.connect("airbnblisting.db")

listing.to_sql("airbnblisting", db_conn, if_exists='append', index=False)

austinlisting = pandas.read_sql("SELECT * FROM airbnblisting where listing_id = 5456 LIMIT 5", db_conn)

print(austinlisting)

db_conn.close()