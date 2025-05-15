#!/usr/bin/env python
# update_database.py - Script to update the database schema

from database import Base, engine, DATABASE_URL

def update_database_schema():
    """Update the database schema by recreating tables."""
    print(f"Updating database schema at {DATABASE_URL}...")
    
    # Option 1: Drop all tables and recreate (destructive)
    # This will erase all existing data
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    print("Database schema updated successfully.")
    print("NOTE: All previous data has been erased.")
    print("You will need to reprocess your audio files to populate the database again.")

if __name__ == "__main__":
    response = input("This will erase all data in the database. Continue? (y/n): ")
    if response.lower() == "y":
        update_database_schema()
    else:
        print("Update cancelled.")
