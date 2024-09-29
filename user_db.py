import os

import bcrypt
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(url, key)
table = "users"

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def store_user(username: str, password: str, role: str):
    hashed_password = hash_password(password)
    rows_to_insert = [
        {"username": username, "hashed_password": hashed_password, "role": role},
    ]

    try:
        response = (
            supabase.table(table)
            .insert(rows_to_insert)
            .execute()
        )
        print(f"User '{username}' has been added.")
    
    except Exception as e:
        print("Error adding user:", e)
        
def delete_user(username: str):
    try:
        response = (
            supabase.table(table)
            .delete()
            .eq("username", username)
            .execute()
        )
        print(f"User '{username}' has been deleted.")
    
    except Exception as e:
        print("Error deleting user:", e)

def fetch_user_from_database(username: str):
    try:
        response = (
            supabase.table(table)
            .select("hashed_password, role")
            .eq("username", username)
            .execute()
        )
        row = response.data[0]
        return row
    
    except Exception as e:
        print("Error fetching user:", e)