from torch import nn


class LFW_CNN(nn.Module):

  def __init__(self):
    super(LFW_CNN, self).__init__()

    # первый слой
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1, stride=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2)
    self.bn1 = nn.BatchNorm2d(num_features=32)
    self.dropout1 = nn.Dropout(0.4)
    self.act1 = nn.ReLU()

    #второй слой
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    self.bn2 = nn.BatchNorm2d(num_features=64)
    self.dropout2 = nn.Dropout(0.4)
    self.act2 = nn.ReLU()

    #третий слой
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
    self.pool3 = nn.MaxPool2d(kernel_size=2)
    self.bn3 = nn.BatchNorm2d(num_features=128)
    self.dropout3 = nn.Dropout(0.4)
    self.act3 = nn.ReLU()

    #четвертый слой

    self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
    self.pool4 = nn.MaxPool2d(kernel_size=2)
    self.bn4 = nn.BatchNorm2d(num_features=256)
    self.dropout4 = nn.Dropout(0.4)
    self.act4 = nn.ReLU()

    #после трех сверток применим полносвязный слой
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(in_features=1536, out_features=1024)
    self.bn5 = nn.BatchNorm1d(num_features=1024)
    self.dropout5 = nn.Dropout(0.4)
    self.act5 = nn.ReLU()

    self.linear2 = nn.Linear(in_features=1024, out_features=512)
    self.bn6 = nn.BatchNorm1d(num_features=512)
    self.dropout6 = nn.Dropout(0.4)
    self.act6 = nn.ReLU()

    self.linear3 = nn.Linear(in_features=512, out_features=256)
    self.bn7 = nn.BatchNorm1d(num_features=256)
    self.dropout7 = nn.Dropout(0.4)
    self.act7 = nn.ReLU()

    self.linear4 = nn.Linear(in_features=256, out_features=256)
    self.bn8 = nn.BatchNorm1d(num_features=256)
    self.dropout8 = nn.Dropout(0.4)
    self.act8 = nn.ReLU()

    self.linear5 = nn.Linear(in_features=256, out_features=n_classes)

  def forward(self, x):
    #применяем сверточные слои
    layer1 = self.act1(self.dropout1(self.bn1(self.pool1(self.conv1(x)))))
    layer2 = self.act2(self.dropout2(self.bn2(self.pool2(self.conv2(layer1)))))
    layer3 = self.act3(self.dropout3(self.bn3(self.pool3(self.conv3(layer2)))))
    layer4 = self.act4(self.dropout4(self.bn4(self.pool4(self.conv4(layer3)))))

    #применяем полносвязные слои
    dense1 = self.flatten(layer4)
    dense1 = self.act5(self.dropout5(self.bn5(self.linear1(dense1))))
    dense2 = self.act6(self.dropout6(self.bn6(self.linear2(dense1))))
    dense2 = self.act7(self.dropout7(self.bn7(self.linear3(dense2))))
    dense2 = self.act8(self.dropout8(self.bn8(self.linear4(dense2))))

    #выход
    out = self.linear5(dense2)
    return out