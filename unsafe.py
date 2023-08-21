import sqlite3

def unsafe_query(user_input):
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    query = "SELECT * FROM users WHERE username = '" + user_input + "';"
    cursor.execute(query)
    return cursor.fetchall(result)
