import json
from pathlib import Path

# Настройки
match_path = Path("../data/raw/matches.jsonl")
output_path = Path("../data/processed/matches_indexed.jsonl")
hero_data_path = Path("../data/raw/hero_mapping.json")  # hero_name -> hero_id

# Загружаем героев (имя -> id)
with hero_data_path.open("r", encoding="utf-8") as f:
    heroes = json.load(f)  # dict: hero_name -> hero_id

# Создаем обратный словарь id -> имя
id_to_name = {v: k for k, v in heroes.items()}

# Читаем матчи и собираем все уникальные hero_id
hero_ids = set()
all_matches = []
with match_path.open("r", encoding="utf-8") as f:
    for line in f:
        match = json.loads(line)
        all_matches.append(match)
        hero_ids.update(match["radiant"])
        hero_ids.update(match["dire"])

sorted_ids = sorted(hero_ids)
# Создаем индексирование по hero_id
hero_id_to_index = {hid: idx for idx, hid in enumerate(sorted_ids)}

# Фильтруем id_to_name только по тем id, что встречаются в матчах
id_to_name_filtered = {hid: id_to_name.get(hid, f"unknown_{hid}") for hid in sorted_ids}

# Создаем hero_name -> hero_index (через hero_id_to_index)
hero_name_to_index = {name: hero_id_to_index[hid] for hid, name in id_to_name_filtered.items() if not name.startswith("unknown")}
# Создаем обратный hero_index -> hero_name
hero_index_to_name = {idx: name for name, idx in hero_name_to_index.items()}

# Сохраняем словари
with open("../data/raw/hero_id_to_index.json", "w", encoding="utf-8") as f:
    json.dump(hero_id_to_index, f, indent=2, ensure_ascii=False)

with open("../data/raw/hero_name_to_index.json", "w", encoding="utf-8") as f:
    json.dump(hero_name_to_index, f, indent=2, ensure_ascii=False)

with open("../data/raw/hero_index_to_name.json", "w", encoding="utf-8") as f:
    json.dump(hero_index_to_name, f, indent=2, ensure_ascii=False)

# Сохраняем обновлённый матч-файл с индексированными героями
with output_path.open("w", encoding="utf-8") as f:
    for match in all_matches:
        radiant_idx = [hero_id_to_index[hid] for hid in match["radiant"]]
        dire_idx = [hero_id_to_index[hid] for hid in match["dire"]]
        indexed_match = {
            "match_id": match["match_id"],
            "radiant": radiant_idx,
            "dire": dire_idx,
            "radiant_win": match["radiant_win"]
        }
        f.write(json.dumps(indexed_match, ensure_ascii=False) + "\n")

print(f"Создано {len(sorted_ids)} уникальных героев.")
print("Словари и датасет сохранены.")
