import pandas as pd

def get_efa_block(hour):
    if hour >= 23 or hour < 3:
        return 1
    elif 3 <= hour < 7:
        return 2
    elif 7 <= hour < 11:
        return 3
    elif 11 <= hour < 15:
        return 4
    elif 15 <= hour < 19:
        return 5
    elif 19 <= hour < 23:
        return 6
    
def get_efa_block2(hour):
    if hour > 23 or hour <= 3:
        return 1
    elif 3 < hour <= 7:
        return 2
    elif 7 < hour <= 11:
        return 3
    elif 11 < hour <= 15:
        return 4
    elif 15 < hour <= 19:
        return 5
    elif 19 < hour <= 23:
        return 6
    
def shift_efa(time):
    if time.hour == 23:
        return time + pd.Timedelta(hours=1)
    else:
        return time
