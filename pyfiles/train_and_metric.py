from tqdm.notebook import tqdm
from collections import defaultdict
from IPython.display import clear_output


def train(model, criterion, optimizer, trainloader, testloader, num_epochs=30):
    """
    model --     модель
    criterion -- лосс функция
    optimizer -- оптимизатор
    trainloader -- генератор батчей тренировочного датасета
    testloader -- генератор батчет тестового датасета
    num_epochs -- число эпох

    return:

    model -- обученная модель
    history -- история обучения
     """
    history = defaultdict(lambda: defaultdict(list))

    for epoch in range(num_epochs):
        train_loss = 0
        test_loss = 0

        # устанавливаем режим тренировки
        model.train(True)

        for X_batch, y_batch in tqdm(trainloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.cpu().detach().numpy()

        train_loss = train_loss / len(trainloader)

        history['loss']['train'].append(train_loss)

        # устанавливаем валидацию
        model.eval()

        with torch.no_grad():
            for X_batch, y_batch in tqdm(testloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                test_loss += loss.cpu().detach().numpy()

        test_loss = test_loss / len(testloader)

        history['loss']['test'].append(test_loss)

        clear_output()
        print(f'Epoch {epoch}')
        print(f'train_loss = {train_loss}')
        print(f'test_loss = {test_loss}')

    return model, history




def get_roc_auc_score(subset, model):
    """
       get_roc_auc_score - функция, вычисляющая метрику ROC AUC для переданного подмножества и модели

   Аргументы:

       subset: Dataset, подмножество датасета, на котором нужно вычислить метрику
       model: nn.Module, модель, для которой нужно вычислить метрику

   Возвращает:

       test_roc_auc_score: float, значение метрики ROC AUC
   """

    test_roc_auc_score = 0
    testloader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(testloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            softmax = nn.Softmax(dim=1)
            y_proba = softmax(logits.cpu().detach())  # для предсказания вероятности классов нужно сделать softmax

            test_roc_auc_score += roc_auc_score(y_batch.cpu().detach().numpy(), y_proba.numpy(), multi_class='ovo',
                                                average='macro')

    test_roc_auc_score /= len(testloader)

    return test_roc_auc_score