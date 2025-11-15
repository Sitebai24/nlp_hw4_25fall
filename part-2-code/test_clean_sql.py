from load_data import custom_clean_sql

samples = [
    "SELECT * FROM flights WHERE 1=1 AND city='Boston'",
    "SELECT * FROM airport , flight",
    "SELECT * FROM flight WHERE AND(airport='JFK')",
    "SELECT name FROM airline WHERE 1 = 1 AND 1 = 1"
]

for s in samples:
    cleaned = custom_clean_sql(s)
    print("Before:", s)
    print("After: ", cleaned)
    print("-" * 50)