import torch
from torch.utils.data import Dataset


class PeopleDataset(Dataset):
    """
      PeopleDataset - кастомный датасет для загрузки изображений людей

  Атрибуты:
      data: tensor, содержит изображения людей
      target: list, содержит метки классов для изображений

  Методы:
      init(self, X, y) - конструктор класса, принимает на вход изображения и метки классов
      len(self) - возвращает длину датасета (число изображений)
      getitem(self, idx) - возвращает idx-ое изображение из датасета и соответствующую ему метку класса
  """

    def __init__(self, X, y):
        self.data = torch.tensor(X)
        self.target = y

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx].permute(2, 0, 1), self.target[idx]
        # перестановка местами, для того, чтобы число каналов было сначала.