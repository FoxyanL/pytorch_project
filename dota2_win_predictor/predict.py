import torch
import json

# параметры
HERO_COUNT = 126
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# имя -> индекс героев
with open("data/raw/hero_name_to_index.json", "r", encoding="utf-8") as f:
    hero_name_to_index = json.load(f)

def heroes_to_onehot(radiant_names, dire_names):
    x = torch.zeros(HERO_COUNT * 2, dtype=torch.float32)

    for hero in radiant_names:
        idx = hero_name_to_index.get(hero.lower())
        if idx is not None and 0 <= idx < HERO_COUNT:
            x[idx] = 1.0

    for hero in dire_names:
        idx = hero_name_to_index.get(hero.lower())
        if idx is not None and 0 <= idx < HERO_COUNT:
            x[idx + HERO_COUNT] = 1.0

    return x.unsqueeze(0)

# Пример входа
radiant = ['magnus', 'doom', 'windranger', 'sven', 'lion']
dire = ['hoodwink', 'huskar', 'io', 'gyrocopter', 'witch doctor']

# Загрузка модели
from models.predictor import Dota2MLP

model = Dota2MLP(hero_count=HERO_COUNT)
checkpoint = torch.load("checkpoints/model_epoch1.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Подготовка входа
x = heroes_to_onehot(radiant, dire).to(DEVICE)

# Предсказание
with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits).item()

print(f"Вероятность победы Radiant: {prob:.4f}")
if prob >= 0.5:
    print("Прогноз: Победа Radiant")
else:
    print("Прогноз: Победа Dire")
