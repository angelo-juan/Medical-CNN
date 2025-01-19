#pycharm preference python interpreter install pip package
#pycharm terminal
#install pytorch : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
#安裝完成後，在 Python 環境中執行以下程式碼來驗證是否成功使用 Apple Silicon Metal 加速：
import torch
print(torch.backends.mps.is_available()) # 應該輸出 True
print(torch.device('mps'))  # 應該輸出 mps:0

#1. set the environment

#install monai packages for medical CNN :pip install -U gdown monai timm

import os
import shutil
from tqdm.notebook import tqdm
import yaml
import gdown
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import torch
import torchvision
import timm

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, CropForegroundd, EnsureChannelFirstd,
    EnsureTyped, Lambdad, LoadImaged,
    RandAffined, RandFlipd, RepeatChanneld,
    Resized, ScaleIntensityd, ScaleIntensityRanged
)
from monai.utils.misc import set_determinism
# Set seed for reproducibility
seed = 42
set_determinism(seed)
print('1-1-2')

#2. download data from Google Drive and untarred in specific folder

group_num = 15 #@param {type:"slider", min: 1, max: 15}
print('2-1-1')

file_ids = [
    '1k0P1gvmFuAEaY5Xdwx3sEdEGXPWtuvGG', '1KBZ9zH75YOr8tKyXuOmRKSlSu7LWT8hz',
    '1ZUK-YGq_PFPbZKIY4vOQuHNW4tQ665fY', '1Hpbh-3zXcyzV15ppICD5-QJ4PvimSaXv',
    '1efrMEe_9Ib6TK9H7jifAWkAjfoApfl7T', '1LaySKm3H-4adrU_9Id4GSamFL8eBkEMo',
    '1_Nrh5BqTkuM42NHRSFyQajs67pRhG13H', '1zkEp14iSY5NC158CcT9CucBtLGHT3Ela',
    '1ZSwFWKvzwScHQW24rwMU4kRodKpOmVrV', '1Ahlm78bJY4eJRkJQs0QkmO6fr1zyOdBd',
    '136jWKAL18EOkNeyiavp-CcsYdSyiSDib', '1sdUx1bR1mqKo9796nGScUf5PDir0QMam',
    '1jG8CUVA9dpqrB1eviZTp5C3U0BDPVM0b', '1ngwbBFmfvag_3Vo_3dDNr8NkFeXYUmLp',
    '1j2j8CY-o5Ls0uXAQgO6EPjWgv6unGBj4'
]
tar_file = 'Atelectasis.tar.gz'
# 指定下載路徑
download_path = '/'
folder = 'Atelectasis'
# 確保路徑存在，如果不存在則創建
if not os.path.exists(tar_file):
    id = file_ids[(group_num-1)]
    print(f"id = {id}, tar_file = {tar_file}")
    gdown.download(id=id, output=tar_file)
else:
    print("Data already exist!")
print('2-1-2')

# unzip the file
if not os.path.exists(folder) and os.path.exists(tar_file):
    print("Data untarring")
    cmd = f'tar -zxf {tar_file}'
    ret_code = os.system(cmd)
    print(f'{cmd} completed with return code {ret_code}')

if not os.path.exists(folder):
    print("Data directory doesn't exist.")
else:
    print("Data successfully downloaded and untarred")
print('2-1-3')

#3. Check dataset status

#3-1. Set the path and load the csv
folder_path = '/Users/user/PythonProject/CNN/Atelectasis'
with open(f'{folder_path}/data_list.yaml', 'r') as fp:
    data_dict = yaml.safe_load(fp)
data_list = data_dict['data']
print('3-1')

#3-2. split data

#3-2-1 number of data for train, valid, test
train_percentage = 0.6 # @param {type: "number"}
valid_percentage = 0.2 #@param {type: "number"}
test_percentage = 0.2 #@param {type: "number"}
print('3-2-1')

#3-2-2 split
label_list = [a["label"] for a in data_list] # 提取所有標籤
 # 第一次分割：分割出訓練集和剩餘部分 (驗證集 + 測試集)
train_list, valid_test_list = train_test_split(
    data_list,
    train_size=train_percentage,
    test_size=valid_percentage+test_percentage, # 驗證集和測試集比例加總
    stratify=label_list, # 分層抽樣
    random_state=seed, # 設定隨機種子
    shuffle=True # 打亂資料順序
)

valid_test_label_list = [a["label"] for a in valid_test_list] # 提取剩餘部分的標籤
 # 第二次分割：將剩餘部分分割成驗證集和測試集
valid_list, test_list = train_test_split(
    valid_test_list,
    train_size=valid_percentage/(valid_percentage+test_percentage), # 計算驗證集在剩餘部分中的比例
    test_size=test_percentage/(valid_percentage+test_percentage), # 計算測試集在剩餘部分中的比例
    stratify=valid_test_label_list, # 分層抽樣
    random_state=seed, # 設定隨機種子
    shuffle=True # 打亂資料順序
)
print(f'{len(train_list)} data for training')
print(f'{len(valid_list)} data for validation')
print(f'{len(test_list)} data for testing')
print('3-2-2')

#3-3 Data preprocessing and data visualization

#3-3-1
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    Resized,
    RepeatChanneld,
    RandAffined
)
# 定義影像的空間大小，用於調整影像尺寸
spatial_size = [224, 224]  # 將影像調整為 224x224 像素
# 定義前處理步驟 (在所有資料集上執行的步驟)
preprocess = [
    Resized(keys='image', spatial_size=spatial_size) # 將影像調整為指定大小
]
# 定義增強步驟 (僅在訓練集上執行的步驟)
augmentation = [
    RandAffined( # 隨機仿射變換
        keys='image', # 作用於 'image' 鍵對應的影像
        rotate_range=0, # 旋轉角度範圍 (此處設為 0，表示不旋轉)
        shear_range=0, # 剪切角度範圍 (此處設為 0，表示不剪切)
        translate_range=0, # 平移範圍 (此處設為 0，表示不平移)
        scale_range=2, # 縮放範圍 (此處設為 2，表示在 0.5 到 2 倍之間隨機縮放)
        prob=1, # 應用此變換的機率 (1 表示總是應用)
        padding_mode='border' # 邊界填充模式
    )
]
# 設定快取率 (0 表示不使用快取)
cache_rate = 0 #通常在資料量非常大時，為了加速訓練會使用快取，這裡設定為0表示不使用
# 定義載入步驟 (所有資料集都需要載入)
load = [
    LoadImaged(keys='image', reader='pilreader', ensure_channel_first=True), # 使用 PIL 讀取影像，並確保通道優先 (C, H, W)
    EnsureTyped(keys=['image', 'label']) # 確保 'image' 和 'label' 為正確的資料型別 (例如 Tensor)
]
# 確保影像有三個通道 (例如 RGB 影像)。如果原始影像是灰階影像，則會重複通道。
preprocess += [RepeatChanneld(keys='image', repeats=3)] # 將單通道影像轉換為三通道影像
 # 定義訓練、驗證和測試集的轉換
train_transform = Compose(load + preprocess + augmentation) # 訓練集：載入 + 前處理 + 增強
valid_transform = Compose(load + preprocess) # 驗證集：載入 + 前處理 (不使用增強)
test_transform = Compose(load + preprocess) # 測試集：載入 + 前處理 (不使用增強)
# 總結：
# 這段程式碼定義了影像資料的前處理和增強流程。
# - load: 負責載入影像和標籤，並轉換為正確的資料型別。
# - preprocess: 負責調整影像大小和通道數。
# - augmentation: 負責對訓練影像進行隨機仿射變換，以增加資料的多樣性，防止過擬合。
# 訓練集使用所有步驟 (載入、前處理、增強)，而驗證集和測試集只使用載入和前處理步驟。
print('3-3-1')

#3-3-2
train_set = CacheDataset(train_list, train_transform, cache_rate=cache_rate)
valid_set = CacheDataset(valid_list, valid_transform, cache_rate=cache_rate)
test_set = CacheDataset(test_list, test_transform, cache_rate=cache_rate)
print('3-3-2')

#3-3-3
data = train_set[99]
print(data['image'].size())
print(data['image'].max())
print(data['image'].min())
print('3-3-3')

#3-3-4
# check label distribution
def check_dist(dataset):
    positive = 0
    negative = 0
    for data in dataset:
        if data['label'].item() == 1:
            positive += 1
        else:
            negative += 1
    print(
        f'number of positive = {positive:.0f}, '
        f'number of negative = {negative:.0f}, '
        f'number of total data = {len(dataset)}'
    )
check_dist(train_set)
check_dist(valid_set)
check_dist(test_set)
print('3-3-4')

# 3-3-5
# helper function for plotting image
def show_data_with_histogram(data):
    column_cnt = 2
    fig, axs = plt.subplots(1, 2, figsize=(5*column_cnt, 4))

    image = data['image']
    image = image.permute(2, 1, 0)
    title = f"Label: {data['label'].item()}"

    axs[0].imshow(image)
    axs[0].set_title(title)

    counts, bins = np.histogram(image, bins=256)
    axs[1].hist(bins[:-1], bins=256, weights=counts, log=True)
    plt.show()
    print('3-3-5')

#3-3-6
random_index = np.random.randint(0, len(train_set), 10)
for index in random_index:
    data = train_set[int(index)]
    show_data_with_histogram(data)
    print('3-3-6')

#4

#4-1
#4-1-1
# Set the hyperparameters
num_epoch = 100#@param {type:"integer"}
batch_size = 150#@param {type: "integer"}
lr = 1e-3#@param {type: "number"}
print('4-1-1')

#4-1-2
import timm
# Set the model
model_name = 'resnet50' # 例如，可以选择任何有效的模型名称
model = timm.create_model(model_name, pretrained=True, num_classes=1)
#print(model)  # 检查模型结构
print('4-1-2')

#4-1-3
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_state = model.state_dict().copy()
model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print('4-1-3')

#4-2
#4-2-1
import torch
torch.cuda.empty_cache()
print('4-2-1')

#4-2-2
record = {'train': [], 'valid': []}
model.load_state_dict(init_state)
best_valid_loss = np.inf
for epoch in tqdm(range(num_epoch)):
    train_loss = 0.0
    valid_loss = 0.0
    # training
    model.train()
    for data in train_loader:
        images = data['image'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad() # zero out the optimizer gradient

        preds = model(images) # model prediction
        loss = criterion(preds, labels) # calculate loss
        loss.backward() # calculate gradient of loss w.r.t. model weights
        optimizer.step() # weights update
        train_loss += loss.item()

    # validation
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images = data['image'].to(device)
            labels = data['label'].to(device)

            preds = model(images)
            loss = criterion(preds, labels)
            valid_loss += loss.item()


    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    record['train'].append(train_loss)
    record['valid'].append(valid_loss)

    # save model with lowest validation loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_state = model.state_dict().copy()
        torch.save(model.state_dict(), "/content/best_weights.pth")

    print(
        f'[{epoch+1}/{num_epoch}] '
        f'Train loss: {train_loss:3.3f}, '
        f'Valid loss: {valid_loss:3.3f} '
    )
print('4-2-2')

#4-2-3
# plot loss
fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.plot(record['train'])
axs.plot(record['valid'])
axs.set_xticks(range(0, num_epoch+1, 5))
axs.set_ylabel('Loss')
axs.set_xlabel('Epoch')
axs.legend(['train', 'valid'], loc='lower left')
plt.show()
print('4-2-3')

#4-2-4
# helper function for inference and plotting
def infer(model, data_loader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data['image'].to(device)
            labels = data['label'].to(device)

            preds = sigmoid(model(images))

            for pred, label in zip(preds, labels):
                y_pred.append(pred.item())
                y_true.append(label.item())

    return y_true, y_pred
print('4-2-4')


#5
#5-1
#5-1-1
def plot_result(y_true, y_pred, thres, title=''):
    # plot the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.show()

    # binarize the predictions
    y_pred = np.where(np.array(y_pred)>=thres, 1, 0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    ############### add to me #################
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ############### add to me #################

    print(
        f'True positive: {tp}\n'
        f'False positive: {fp}\n'
        f'False negative: {fn}\n'
        f'True negative: {tn}\n'
        f'Sensitivity: {sensitivity:.4f}\n'
        f'Specificity: {specificity:.4f}'
    )

#5-1-2
# load the best weight and infer
model.load_state_dict(best_state)
train_true, train_pred = infer(model, train_loader)
valid_true, valid_pred = infer(model, valid_loader)
test_true, test_pred = infer(model, test_loader)

#5-1-3
thres = 0.5#@param {type: "number"}

#5-1-4
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_result(y_true, y_pred, thres, title=''):
    """
    Plots the ROC curve, calculates performance metrics, and displays the confusion matrix.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        thres (float): Threshold for binarization.
        title (str, optional): Title for the plot. Defaults to ''.
    """

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.show()

    # Binarize predictions
    y_pred = np.where(np.array(y_pred) >= thres, 1, 0)

    # Performance metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(
        f'True positive: {tp}\n'
        f'False positive: {fp}\n'
        f'False negative: {fn}\n'
        f'True negative: {tn}\n'
        f'Sensitivity: {sensitivity:.4f}\n'
        f'Specificity: {specificity:.4f}'
    )

    # Confusion matrix visualization
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.text(0, 0, f'TN = {tn}', va='center', ha='center', fontsize=12)
    plt.text(1, 0, f'FP = {fp}', va='center', ha='center', fontsize=12)
    plt.text(0, 1, f'FN = {fn}', va='center', ha='center', fontsize=12)
    plt.text(1, 1, f'TP = {tp}', va='center', ha='center', fontsize=12)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({title})')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Example usage
thres = 0.5
plot_result(train_true, train_pred, thres, title='Train')
print('\n')
plot_result(valid_true, valid_pred, thres, title='Valid')
print('\n')
plot_result(test_true, test_pred, thres, title='Test')



