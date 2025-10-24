import json
import pandas as pd
import requests
import os

def download_statsbomb_data(competition_id, season_id, output_dir="data"):
    """
    Downloads StatsBomb event data for a given competition and season.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get match IDs for the specified competition and season
    competitions_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    competitions = requests.get(competitions_url).json()
    
    match_ids = []
    for comp in competitions:
        if comp['competition_id'] == competition_id and comp['season_id'] == season_id:
            matches_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{competition_id}/{season_id}.json"
            matches = requests.get(matches_url).json()
            match_ids = [match['match_id'] for match in matches]
            break
    
    if not match_ids:
        print(f"No matches found for competition_id {competition_id} and season_id {season_id}")
        return

    print(f"Downloading {len(match_ids)} matches...")
    for match_id in match_ids:
        event_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
        events = requests.get(event_url).json()
        
        with open(os.path.join(output_dir, f"events_{match_id}.json"), "w") as f:
            json.dump(events, f)
        print(f"Downloaded events for match {match_id}")

def load_and_preprocess_events(file_path):
    """
    Loads a single StatsBomb event JSON file and performs basic preprocessing.
    """
    with open(file_path, 'r') as f:
        events = json.load(f)
    
    df = pd.json_normalize(events, sep='_')
    
    # Example preprocessing: convert timestamp to seconds
    if 'timestamp' in df.columns:
        df['timestamp_seconds'] = df['timestamp'].apply(lambda x: sum(float(i) * 60**j for j, i in enumerate(x.split(':')[::-1])))
    
    return df

if __name__ == "__main__":
    # Example usage: Download data for a specific competition and season
    # You can find competition_id and season_id from https://github.com/statsbomb/open-data/tree/master/data/competitions.json
    # For example, La Liga 2015/2016: competition_id=11, season_id=27
    
    # download_statsbomb_data(competition_id=11, season_id=27, output_dir="data/raw_events")

    # Example usage: Load and preprocess a downloaded event file
    # Assuming you have a file like 'data/raw_events/events_xxxx.json'
    # event_df = load_and_preprocess_events("data/raw_events/events_xxxx.json")
    # print(event_df.head())
    pass