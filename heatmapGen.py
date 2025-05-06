
import torch
from torch import nn
from torch.utils.data import DataLoader
from file_reading import load_challenge_data, connvert_to_difference_data
from plotting_test import base_2, surface_plot
from scipy.signal import iirnotch, filtfilt, butter
from test import FetalQRSWindowDetector, load_data, train_test_cycle,test
from fvcore.nn import FlopCountAnalysis


batch_size = 64
epochs = 60
# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset, test_dataset = load_data(challenge=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

z = []
x = []
y = []



train_dataset, test_dataset = load_data(challenge=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for hp in range(2,64,4):    #(1,50,3):
    x.append(hp)

    print(f"{hp}")


    model = FetalQRSWindowDetector(in_channels=4,hidden_width=hp).to(device)
    acc = train_test_cycle((train_loader,test_loader), model, batch_size=batch_size,lr=0.005,epochs=epochs,threshold=0.3)
    y.append(acc)

base_2(x,y)


# for hp in range(2,64,4):    #(1,50,3):
#     x.append(hp)
#     temp = []
#     for lp in range(1,40,2):
#         print(f"{hp},  {lp}")
#         if len(x) == 1:
#             y.append(lp/10)
#         train_dataset, test_dataset = load_data(challenge=True,lp=lp/10,hp=hp)

#         # Create DataLoaders
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#         criterion = nn.BCELoss()
#         test_loss, accuracy, f1_score = test(test_loader, model, criterion, 0.3)
#         temp.append(f1_score)





# z = []
# x = []
# y = []
# for lr in range(1,50,3):    #(1,50,3):
#     x.append(lr/10000)
#     temp = []
#     for cutoff in range(1,60,5):
#         if len(x) == 1:
#             y.append(cutoff/100)
#         # Initialize model, loss, optimizer
#         model = FetalQRSWindowDetector(in_channels=4).to(device)

#         # input_tensor = torch.randn(1,4, 150)  # Batch size of 64
#         # flops = FlopCountAnalysis(model, input_tensor)
#         # print(f"FLOPs: {flops.total()}")  # Total FLOPs for the forward pass
#         acc = train_test_cycle((train_loader,test_loader), model, batch_size=batch_size,lr=lr/10000,epochs=epochs,threshold=cutoff/100)
#         temp.append(acc)
#         print(acc)
    
#     z.append(temp)

surface_plot(x,y,z)