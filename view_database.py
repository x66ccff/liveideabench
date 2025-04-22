import os
import sqlite3
import pandas as pd
import json

# Define the database file path
DB_PATH = './data/ideabench.db'

def load_and_display_database():
    """
    Load the SQLite database and display its content as a DataFrame
    """
    # Check if the database file exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_PATH}' does not exist")
        return

    # Create a connection to the database
    conn = sqlite3.connect(DB_PATH)

    try:
        # Query all results
        query = "SELECT * FROM results ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)

        # Process JSON fields
        for json_col in ['parsed_scores', 'parsed_reasoning']:
            if json_col in df.columns:
                df[json_col] = df[json_col].apply(
                    lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else x
                )

        # Print basic statistics
        print(f"Total records in the database: {len(df)}")
        print("\nBasic Statistics:")
        print(f"Unique keywords: {df['keywords'].nunique()}")
        print(f"Number of idea model types: {df['idea_model'].nunique()}")
        print(f"Number of critic model types: {df['critic_model'].nunique()}")

        # Display an overview of the DataFrame
        print("\nData Preview:")
        # Select more meaningful columns for display
        display_columns = [
            'id', 'timestamp', 'keywords', 'idea_model', 'critic_model',
            'parsed_scores', 'first_was_rejected'
        ]
        preview_df = df[display_columns].head(10)

        return df

    except sqlite3.Error as e:
        print(f"Error reading the database: {e}")

    finally:
        # Close the connection
        conn.close()

# Execute the main function
if __name__ == "__main__":
    print("Loading IdeaBench database...")
    df = load_and_display_database()
    df.to_csv('./csvs/view.csv')

    if df is not None:
        # More analysis code can be added here
        print("\n🎉Database loaded successfully! \n💡You can now run `stats.ipynb` to generate `data/data.parquet` which serves as input for the subsequent analysis notebooks....")

        # If interactive analysis is needed, the df variable can be kept
        # For example, you can uncomment the lines below to enable interactive analysis
        # import code
        # code.interact(local=locals())