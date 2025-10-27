import pandas as pd

def load_user_profiles(profile_path="data/User_Preference_Profiles.csv"):
    return pd.read_csv(profile_path)