import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Utils:
  def __init__(self):
    pass

  def prepare_data(self):

    train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    return train_data, test_data

  def load_data(self, train_data, test_data):
    batch_size = 512
    kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

  def visualize_data(self, train_loader):
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure()

    for i in range(12):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])

  def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

  def train(self, model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()

      # Predict
      pred = model(data)

      # Calculate loss
      loss = criterion(pred, target)
      train_loss+=loss.item()

      # Backpropagation
      loss.backward()
      optimizer.step()

      correct += self.GetCorrectPredCount(pred, target)
      processed += len(data)

      pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

  def test(self, model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction = 'sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



  def train_and_evaluate_model(self):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if self.use_cuda else "cpu")
    model = Net().to(self.device)
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    criterion = F.cross_entropy
    num_epochs = 20

    for epoch in range(1, num_epochs+1):
      print(f'Epoch {epoch}')
      self.train(model, device, train_loader, optimizer, criterion)
      self.test(model, device, test_loader, criterion)
      scheduler.step()

    return train_losses, test_losses, train_acc, test_acc

  def plot_losses_and_accuracy(self, train_losses, train_acc, test_losses, test_acc):
      fig, axs = plt.subplots(2,2,figsize=(15,10))
      axs[0, 0].plot(train_losses)
      axs[0, 0].set_title("Training Loss")
      axs[1, 0].plot(train_acc)
      axs[1, 0].set_title("Training Accuracy")
      axs[0, 1].plot(test_losses)
      axs[0, 1].set_title("Test Loss")
      axs[1, 1].plot(test_acc)
      axs[1, 1].set_title("Test Accuracy")

  def runMnistModel(self):
    train_data , test_data = self.prepare_data()
    train_loader, test_loader = self.load_data(train_data, test_data)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    train_losses, train_acc, test_losses, test_acc = self.train_and_evaluate_model()
    self.plot_losses_and_accuracy(train_losses, train_acc, test_losses, test_acc)
