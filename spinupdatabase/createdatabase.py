#import the libraries
import sqlite3

db_conn = sqlite3.connect("airbnblisting.db")

c = db_conn.cursor()

c.execute(
    """CREATE TABLE listing (
        id INTEGER,
        listing_id INTEGER, 
        listing_name TEXT,
        listing_location TEXT,
        listing_neighbourhood TEXT,
        listing_latitude REAL,
        listing_longitude REAL,
        listing_price REAL,
        listing_review_scores REAL,
        listing_comments TEXT,
        PRIMARY KEY(id)
        );"""
)

db_conn.close()