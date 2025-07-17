import requests
import json

def build_mapping(output_path="../data/raw/hero_mapping.json"):
    url = "https://api.opendota.com/api/heroes"
    resp = requests.get(url)
    resp.raise_for_status()
    heroes = resp.json()

    mapping = {}
    for h in heroes:
        name = h["localized_name"].lower()
        mapping[name] = h["id"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(mapping)} героев в {output_path}")

if __name__ == "__main__":
    build_mapping()