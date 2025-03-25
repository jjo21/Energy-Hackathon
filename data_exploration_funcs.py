def get_efa_block(hour):
    if hour >= 23 or hour < 3:
        return 'EFA 1'
    elif 3 <= hour < 7:
        return 'EFA 2'
    elif 7 <= hour < 11:
        return 'EFA 3'
    elif 11 <= hour < 15:
        return 'EFA 4'
    elif 15 <= hour < 19:
        return 'EFA 5'
    elif 19 <= hour < 23:
        return 'EFA 6'
