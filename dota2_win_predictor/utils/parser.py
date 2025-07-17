import requests
import time
import json
import os

SAVE_PATH = "../data/raw/matches.jsonl"
NUM_MATCHES = 10000
MATCH_IDS_SEEN = set()

# Загружаем уже сохранённые match_id
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "match_id" in data:
                    MATCH_IDS_SEEN.add(data["match_id"])
            except json.JSONDecodeError:
                continue

def fetch_public_matches():
    url = "https://api.opendota.com/api/publicMatches"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print("Ошибка при запросе:", e)
    return []

def save_matches(matches, path):
    global MATCH_IDS_SEEN
    new_count = 0
    with open(path, "a", encoding="utf-8") as f:
        for match in matches:
            match_id = match.get("match_id")
            if (
                match_id not in MATCH_IDS_SEEN and
                isinstance(match.get("radiant_team"), list) and
                isinstance(match.get("dire_team"), list) and
                len(match["radiant_team"]) == 5 and
                len(match["dire_team"]) == 5 and
                "radiant_win" in match
            ):
                data = {
                    "match_id": match_id,
                    "radiant": match["radiant_team"],
                    "dire": match["dire_team"],
                    "radiant_win": match["radiant_win"]
                }
                f.write(json.dumps(data) + "\n")
                MATCH_IDS_SEEN.add(match_id)
                new_count += 1
    return new_count

if __name__ == "__main__":
    total = len(MATCH_IDS_SEEN)
    while total < NUM_MATCHES:
        print(f"Загружаю... {total}/{NUM_MATCHES}")
        matches = fetch_public_matches()
        if matches:
            added = save_matches(matches, SAVE_PATH)
            total += added
            print(f"Добавлено {added} новых матчей.")
        else:
            print("Получены пустые данные.")
        time.sleep(120)
