from functions_model import get_dir_loader, init_model, create_classifier, run_network, get_crit_opt, hidden_units_check
from utils import get_input_args
import torch

in_arg = get_input_args()

# Retrieve and load data
train_loader, loader, train_data = get_dir_loader(in_arg.dir)

# Initialize model and freeze parameters
model = init_model(in_arg.arch)

if in_arg.hidden_units == None:
    hidden_units = hidden_units_check(in_arg.arch)
else:
    hidden_units = in_arg.hidden_units

# Create classifier
model.classifier = create_classifier(in_arg.arch, hidden_units)

# Retrieve criterion and optimizer
criterion, optimizer = get_crit_opt(model.classifier, in_arg.learning_rate)

# Train network
run_network(model, train_loader, loader, in_arg.epochs, criterion, optimizer, in_arg.gpu)

# Save model to checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {'arch': in_arg.arch,
              'epochs': in_arg.epochs,
              'optimizer': optimizer.state_dict,
              'criterion': criterion.state_dict,
              'class_to_idx': model.class_to_idx,
              'classifier': model.classifier,
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')