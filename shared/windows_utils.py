import re
from datetime import datetime


def parse_dotnet_date(date_str):
    match = re.match(r'/Date\((\d+)([+-]\d{4})\)/', date_str)
    if match:
        timestamp = int(match.group(1)) / 1000
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    return date_str
