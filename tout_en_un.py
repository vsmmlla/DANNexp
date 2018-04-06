import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import pickle

#params.py
use_gpu = True
data_root = './data'
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)
source_path = data_root + '/MNIST'
target_path = data_root + '/mnist_m'
batch_size = 512
epochs = 90
gamma = 20
src_accuracy = []
tgt_accuracy = []
domain_accuracy = []

##utils.py
#def get_train_loader(dataset):
#    """
#    Get train dataloader of source domain or target domain
#    :return: dataloader
#    """
#    if dataset == 'MNIST':
#        transform = transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize(mean= dataset_mean, std= dataset_std)
#        ])
#
#        data = datasets.MNIST(root= source_path, train= True, transform= transform,
#                              download= True)
#
#        dataloader = DataLoader(dataset= data, batch_size= batch_size, shuffle= True, num_workers = 8)
#    elif dataset == 'MNIST_M':
#        transform = transforms.Compose([
#            transforms.RandomCrop((28)),
#            transforms.ToTensor(),
#            transforms.Normalize(mean= dataset_mean, std= dataset_std)
#        ])
#
#        data = datasets.ImageFolder(root= target_path + '/train', transform= transform)
#
#        dataloader = DataLoader(dataset = data, batch_size= batch_size, shuffle= True, num_workers = 8)
#    else:
#        raise Exception('There is no dataset named {}'.format(str(dataset)))
#
#    return dataloader
#
#def get_test_loader(dataset):
#    """
#    Get test dataloader of source domain or target domain
#    :return: dataloader
#    """
#    if dataset == 'MNIST':
#        transform = transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize(mean= dataset_mean, std= dataset_std)
#        ])
#
#        data = datasets.MNIST(root= source_path, train= False, transform= transform,
#                              download= True)
#
#        dataloader = DataLoader(dataset= data, batch_size= batch_size, shuffle= True)
#    elif dataset == 'MNIST_M':
#        transform = transforms.Compose([
#            transforms.RandomCrop((28)),
#            transforms.ToTensor(),
#            transforms.Normalize(mean= dataset_mean, std= dataset_std)
#        ])
#
#        data = datasets.ImageFolder(root= target_path + '/test', transform= transform)
#
#        dataloader = DataLoader(dataset = data, batch_size= batch_size, shuffle= True)
#    else:
#        raise Exception('There is no dataset named {}'.format(str(dataset)))
#
#    return dataloader

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer

#model.py
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size= 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size= 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.conv2_drop(self.relu2(self.conv2(x))))
        x = x.view(-1, 48 * 4 * 4)

        return x

class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, features):
        logits = self.relu1(self.fc1(features))
        logits = self.relu2(self.fc2(logits))
        logits = self.fc3(logits)

        return F.log_softmax(logits, dim=0)

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 2)

    def forward(self, features, constant):
        logits = GradReverse.apply(features, constant)
        logits = self.relu1(self.fc1(logits))
        logits = self.relu2(self.fc2(logits))
        logits = F.log_softmax(self.fc3(logits), dim=0)

        return logits

#class MNISTarch(nn.Module):
#
#    def __init__(self):
#        super(MNISTarch, self).__init__()
#        self.feature = nn.Sequential(
#            nn.Conv2d(1, 32, kernel_size=5),
#            nn.ReLu(),
#            nn.MaxPool2d(2),
#            nn.Conv2d(32, 48, kernel_size=5),
#            nn.ReLu(),
#            nn.MaxPool2d(2)
#        )
#
#        self.class_classifier = nn.Sequential(
#            nn.Linear(768, 100),
#            nn.ReLU(),
#            nn.Linear(100, 100),
#            nn.ReLU(),
#            nn.Linear(100, 10),
#            nn.ReLU()
#        )
#
#        self.domain_classifier == nn.Sequential(
#            nn.Linear(768, 100),
#            nn.ReLU(),
#            nn.Linear(100, 2),
#            nn.ReLU()
#        )
#
#    def forward(self, input_data, alpha):
#        #input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)#??
#        features = self.feature(input_data)
#        features = features.view(-1, 768)
#        class_output = self.class_classifier(features)
#        domain_output = self.domain_classifier(GradReverse.apply(features, alpha))
#
#        return class_output, domain_output


#train.py
def train(feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
          source_dataloader, target_dataloader, optimizer, epoch):
    """
    Execute target domain adaptation
    :param feature_extractor:
    :param class_classifier:
    :param domain_classifier:
    :param class_criterion:
    :param domain_criterion:
    :param source_dataloader:
    :param target_dataloader:
    :param optimizer:
    :return:
    """
    # setup models
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = epochs * len(source_dataloader)

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
        # setup hyperparameters
        p = float(batch_idx + start_steps) / total_steps
        #constant = 2. / (1. + np.exp(-gamma * p)) - 1
        constant = 1. / (1. + np.exp(-gamma * p + gamma/2))
        #constant = 10*p

        # prepare the data
        input1, label1 = sdata
        input2, label2 = tdata
        size = min((input1.shape[0], input2.shape[0]))
        input1, label1 = input1[0:size, :, :, :], label1[0:size]
        input2, label2 = input2[0:size, :, :, :], label2[0:size]
        if use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
            input2, label2 = Variable(input2), Variable(label2)

        # setup optimizer
        optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        # prepare domain labels
        if use_gpu:
            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))
            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        # compute the output of source domain and target domain
        src_feature = feature_extractor(input1)
        tgt_feature = feature_extractor(input2)

        # compute the class loss of src_feature
        class_preds = class_classifier(src_feature)
        class_loss = class_criterion(class_preds, label1)

        # compute the domain loss of src_feature and target_feature
        tgt_preds = domain_classifier(tgt_feature, constant)
        src_preds = domain_classifier(src_feature, constant)
        tgt_loss = domain_criterion(tgt_preds, target_labels)
        src_loss = domain_criterion(src_preds, source_labels)
        domain_loss = tgt_loss + src_loss

        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()

        # print loss
        if (batch_idx + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                batch_idx * len(input2), len(target_dataloader.dataset),
                100. * batch_idx / len(target_dataloader), loss.data[0], class_loss.data[0],
                domain_loss.data[0]
            ))

#def train2(model, class_criterion, domain_criterion,
#          source_dataloader, target_dataloader, optimizer, epoch):
#    """
#    Execute target domain adaptation
#    :param feature_extractor:
#    :param class_classifier:
#    :param domain_classifier:
#    :param class_criterion:
#    :param domain_criterion:
#    :param source_dataloader:
#    :param target_dataloader:
#    :param optimizer:
#    :return:
#    """
#    # setup models
#    model.train()
#
#    # steps
#    start_steps = epoch * len(source_dataloader)
#    total_steps = epochs * len(source_dataloader)
#
#    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
#        # setup hyperparameters
#        p = float(batch_idx + start_steps) / total_steps
#        constant = 2. / (1. + np.exp(-10 * p)) - 1
#
#        # prepare the data
#        input1, label1 = sdata
#        input2, label2 = tdata
#        size = min((input1.shape[0], input2.shape[0]))
#        input1, label1 = input1[0:size, :, :, :], label1[0:size]
#        input2, label2 = input2[0:size, :, :, :], label2[0:size]
#        if use_gpu:
#            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
#            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
#        else:
#            input1, label1 = Variable(input1), Variable(label1)
#            input2, label2 = Variable(input2), Variable(label2)
#
#        # setup optimizer
#        optimizer = optimizer_scheduler(optimizer, p)
#        optimizer.zero_grad()
#
#        # prepare domain labels
#        if use_gpu:
#            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
#            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
#        else:
#            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))
#            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))
#
#        # compute the output of source domain and target domain
#        s_class_pred, s_domain_pred = model(input1)
#        _, t_domain_pred = model(input2)
#        #src_feature = feature_extractor(input1)
#        #tgt_feature = feature_extractor(input2)
#
#        # compute the class loss of src_feature
#        #class_preds = class_classifier(src_feature)
#        #class_loss = class_criterion(class_preds, label1)
#        s_class_err = class_criterion(s_class_pred, label1)
#
#        # compute the domain loss of src_feature and target_feature
#        #tgt_preds = domain_classifier(tgt_feature, constant)
#        #src_preds = domain_classifier(src_feature, constant)
#        #tgt_loss = domain_criterion(tgt_preds, target_labels)
#        #src_loss = domain_criterion(src_preds, source_labels)
#        t_domain_err = domain_criterion(t_domain_pred, target_labels)
#        s_domain_err = domain_criterion(s_domain_pred, source_labels)
#
#        loss = s_class_err + t_domain_loss + s_domain_err
#        loss.backward()
#        optimizer.step()
#
#        # print loss
##        if (batch_idx + 1) % 10 == 0:
##            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
##                batch_idx * len(input2), len(target_dataloader.dataset),
##                100. * batch_idx / len(target_dataloader), loss.data[0], class_loss.data[0],
##                domain_loss.data[0]
##            ))

#test.py
def test(feature_extractor, class_classifier, domain_classifier, source_dataloader, target_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return: None
    """
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()
    #model.eval()

    source_correct = 0
    target_correct = 0
    domain_correct = 0
    domain_tgt_correct = 0
    domain_src_correct = 0

    for batch_idx, sdata in enumerate(source_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        s_input, s_label = sdata
        if use_gpu:
            s_input, s_label = Variable(s_input.cuda()), Variable(s_label.cuda())
            s_domainLabel = Variable(torch.zeros((s_input.size()[0])).type(torch.LongTensor).cuda())
        else:
            s_input, s_label = Variable(s_input), Variable(s_label)
            s_domainLabel = Variable(torch.zeros((s_input.size()[0])).type(torch.LongTensor))

        output1 = class_classifier(feature_extractor(s_input))
        pred1 = output1.data.max(1, keepdim = True)[1]
        source_correct += pred1.eq(s_label.data.view_as(pred1)).cpu().sum()

        src_preds = domain_classifier(feature_extractor(s_input), constant)
        src_preds = src_preds.data.max(1, keepdim= True)[1]
        domain_src_correct += src_preds.eq(s_domainLabel.data.view_as(src_preds)).cpu().sum()

    for batch_idx, tdata in enumerate(target_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        input2, label2 = tdata
        if use_gpu:
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            input2, label2 = Variable(input2), Variable(label2)
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        output2 = class_classifier(feature_extractor(input2))
        pred2 = output2.data.max(1, keepdim=True)[1]
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()

        tgt_preds = domain_classifier(feature_extractor(input2), constant)
        tgt_preds = tgt_preds.data.max(1, keepdim=True)[1]
        domain_tgt_correct += tgt_preds.eq(tgt_labels.data.view_as(tgt_preds)).cpu().sum()

    domain_correct = domain_tgt_correct + domain_src_correct

    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'
          'Domain Accuracy: {}/{} ({:.4f}%)\n'.
        format(
        source_correct, len(source_dataloader.dataset), 100. * source_correct / len(source_dataloader.dataset),
        target_correct, len(target_dataloader.dataset), 100. * target_correct / len(target_dataloader.dataset),
        domain_correct, len(source_dataloader.dataset) + len(target_dataloader.dataset),
        100. * domain_correct / (len(source_dataloader.dataset) + len(target_dataloader.dataset))
    ))

    src_accuracy.append(source_correct/len(source_dataloader.dataset))
    tgt_accuracy.append(target_correct/len(target_dataloader.dataset))
    domain_accuracy.append(domain_correct/(len(source_dataloader.dataset) + len(target_dataloader.dataset)))

#main.py
# prepare the source data and target data
s_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean= dataset_mean, std= dataset_std)
])

t_transform = transforms.Compose([
    transforms.RandomCrop((28)),
    transforms.ToTensor(),
    transforms.Normalize(mean= dataset_mean, std= dataset_std)
])


data = datasets.MNIST(root=source_path, train=True, transform=s_transform, download=True)

src_train_dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=8)


data = datasets.ImageFolder(root=target_path + '/train', transform=t_transform)

tgt_train_dataloader = DataLoader(dataset = data, batch_size= batch_size, shuffle= True, num_workers=8)


data = datasets.MNIST(root=source_path, train=False, transform=s_transform, download=True)
src_test_dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

data = datasets.ImageFolder(root= target_path + '/test', transform=t_transform)
tgt_test_dataloader = DataLoader(dataset = data, batch_size= batch_size, shuffle= True)

# init models
feature_extractor = Extractor()
class_classifier = Class_classifier()
domain_classifier = Domain_classifier()
#model = MNISTarch()

if use_gpu:
    feature_extractor.cuda()
    class_classifier.cuda()
    domain_classifier.cuda()
    #model.cuda()

class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()

# init optimizer
optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                        {'params': class_classifier.parameters()},
                        {'params': domain_classifier.parameters()}], lr= 0.01, momentum= 0.9)
#optimizer = optim.SDG(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train(feature_extractor, class_classifier, domain_classifier,
                class_criterion, domain_criterion,
                src_train_dataloader, tgt_train_dataloader, optimizer, epoch)
    test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader)


with open('courbes','wb') as f:
    pickle.dump((src_accuracy, tgt_accuracy, domain_accuracy), f)

#with open('courbes', 'rb') as f:
#    src_accuracy, tgt_accuracy, domain_accuracy = pickle.load(f)
