import os
from dotenv import load_dotenv

import mysql.connector
from mysql.connector import Error
from typing import Optional

class DatabaseConnection:
    def __init__(self):
        load_dotenv()  # .env 파일 로드
        self.config = {
            "host": os.getenv("DB_HOST"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                return True
        except Error as e:
            print(f"Error: {e}")
            return False
        
    def get_connection(self):
        if not self.connection or not self.connection.is_connected():
            self.connect()
        return self.connection
    
    def execute_query(self, query: str, params: Optional[tuple] = None, structured: bool = False):
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            cursor.close()
            if structured:
                return {"success": True, "data": result}
            return result

        except Error as e:
            if structured:
                return {"success": False, "error": str(e)}
            return None
    
    def execute_update(self, query: str, params: Optional[tuple] = None):
        try:
            connection = self.get_connection()
            cursor = connection.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            connection.commit()
            last_id = cursor.lastrowid
            cursor.close()
            return last_id
        
        except Error as e:
            print(f"Error: {e}")
            if connection:
                connection.rollback()
            return None

db = DatabaseConnection()