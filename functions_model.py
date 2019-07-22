from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Get directory and loader
def get_dir_loader(directory):

    data_dir = directory
    train_dir = 'flowers/train'
    
    if (data_dir == 'flowers/' or data_dir == 'flowers/valid'):
        if data_dir == 'flowers/':
            valid_dir = data_dir + 'valid'
        else:
            valid_dir = data_dir
        test_dir = 'flowers/test'
    elif data_dir == 'flowers/test':
        valid_dir = 'flowers/valid'
        test_dir = data_dir

    
    mean_normalize = [0.485, 0.456, 0.406]
    std_dev_normalize = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean_normalize,
                                                                std_dev_normalize)])

    validate_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean_normalize,
                                                                   std_dev_normalize)])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_normalize,
                                                               std_dev_normalize)])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)                                     
    validate_data = datasets.ImageFolder(valid_dir, transform = validate_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size = 32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
    if (data_dir == 'flowers/' or data_dir == 'flowers/valid'):
        loader = validate_loader
    elif data_dir == 'flowers/test':
        loader = test_loader
    
    return train_loader, loader, train_data

# Initialize model
def init_model(arch):
    if (arch == None or arch == 'vgg'):
        model = models.vgg16(pretrained = True)   
    elif arch == 'densenet':
        model = models.densenet121(pretrained = True)
   
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model

# Hidden units check
def hidden_units_check(arch):
    if arch == 'vgg':
        hidden_units = 1024
    elif arch == 'densenet':
        hidden_units = 500
    
    return hidden_units

# Create classifier
def create_classifier(arch, hidden_units):
    
    if arch == 'vgg':
        classifier = nn.Sequential(OrderedDict([
                                              ('fc1', nn.Linear(25088, hidden_units)),
                                              ('relu1', nn.ReLU()),
                                              ('drop1', nn.Dropout()),
                                              ('fc2', nn.Linear(hidden_units, 102)),
                                              ('output', nn.LogSoftmax(dim=1))
                                              ]))
    elif arch == 'densenet':
        classifier = nn.Sequential(OrderedDict([
                                              ('fc1', nn.Linear(1024, hidden_units)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(hidden_units, 102)),
                                              ('output', nn.LogSoftmax(dim=1))
                                              ]))
    return classifier

# Pass data through network
def run_network(model, train_loader, loader, epoch, criterion, optimizer, device):
    
    # Train the network
    epochs = epoch
    print_every = 40
    steps = 0

    model.to(device)
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        
        for inputs, labels in train_loader:
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
            
                with torch.no_grad():
                    test_loss, accuracy = validation(model, loader, criterion, device)
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(loader)))
            
                running_loss = 0
            
                model.train()
                
# Implement a function for the validation pass
def validation(model, loader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model.forward(inputs)
        test_loss = criterion(outputs, labels).item()
        
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim = 1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy  

# Get criterion and optimizer
def get_crit_opt(model_classifier, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_classifier.parameters(), lr = learning_rate) #0.01
    
    return criterion, optimizer