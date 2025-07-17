import torch
from torch.utils.data import Dataset
import json

class Dota2OneHotDataset(Dataset):
    def __init__(self, jsonl_path: str, hero_count: int = 126):
        self.hero_count = hero_count
        self.input_dim = hero_count * 2  # Radiant + Dire
        self.samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                radiant = data['radiant']
                dire = data['dire']
                radiant_win = data['radiant_win']

                # One-hot вектор: 126 для Radiant, 126 для Dire
                x = torch.zeros(self.input_dim, dtype=torch.float32)

                for hero_id in radiant:
                    if 0 <= hero_id < hero_count:
                        x[hero_id] = 1.0

                for hero_id in dire:
                    if 0 <= hero_id < hero_count:
                        x[hero_id + hero_count] = 1.0

                y = torch.tensor(1.0 if radiant_win else 0.0, dtype=torch.float32)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
