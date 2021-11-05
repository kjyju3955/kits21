import torch.nn as nn
from torch.utils.data import DataLoader
from Unet import UNet
from data_load import *
from torchvision import transforms

print('Device 확인')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print('Device:', device)

batch_size = 4
num_epoch = 1000
mode = "AND"
dir_path = './kits21/data/case_00*'
check_dir = './checkpoint/'
lr = 1e-3

transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor()])

dataset_train = Datasets(dir_path=dir_path, start=0, end=1, transform=transform, mode=mode)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

dataset_val = Datasets(dir_path=dir_path, start=1, end=2, transform=transform, mode=mode)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

num_batch_train = np.ceil(len(dataset_train) / batch_size)
num_batch_val = np.ceil(len(dataset_val) / batch_size)

net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)


def save(check_dir, net, optim, epoch):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (check_dir, epoch))


def load(check_dir, net, optim):
    if not os.path.exists(check_dir):  # 저장된 네트워크가 없으면 인풋을 그데로 반환
        epoch = 0
        return net, optim, epoch

    check_lst = os.listdir(check_dir)  # ckpt_dir 아래에 있는 모든 파일 리스트를 받아옴
    check_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('%s%s' % (check_dir, check_lst))

    dict_model = torch.load('%s%s' % (check_dir, check_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(check_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


start_epoch = 0
net, optim, start_epoch = load(check_dir=check_dir, net=net, optim=optim)

for epoch in range(start_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []
    for batch, data in enumerate(loader_train):
        label = data['label'].to(device)
        inputs = data['input'].to(device)
        output = net(inputs)

        optim.zero_grad()  # gradient 초기화
        '''print('inputs.size()')
        print("input:{}".format(inputs.size()))
        print("output:{}".format(output.size()))
        print("label:{}".format(label.size()))'''
        loss = fn_loss(output, label)  # output과 label사이의 loss 계산
        loss.backward()
        optim.step()

        loss_arr += [loss.item()]

        print('Train: Epoch %04d / %04d | Batch %04d / %04d | Loss %04f' % (
            epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

    with torch.no_grad():  # validation ( backpropa x / loss 만 계산 )
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val):
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)

            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print('Vaild: Epoch %04d / %04d | Batch %04d / %04d | Loss %04f' % (
                epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

        save(check_dir=check_dir, net=net, optim=optim, epoch=epoch)

'''st_epoch = 0
num_epoch = 30


def train_model(net, fn_loss, optim, num_epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 100

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):

            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            optim.zero_grad()  # iteration 끝나면 gradients 0으로 초기화

            loss = fn_loss(output, label)
            loss.backward()  # 각 파라미터의 gradients 계산

            optim.step()  # 신경명 파라미터 update

            loss_arr += [loss.item()]

            print('Train: Epoch %04d / %04d | Batch %04d / %04d | Loss %04f' % (
                epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                data['label '] = data['label'] * 0.5 + 0.5

                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print('Train: Epoch %04d / %04d | Batch %04d / %04d | Loss %04f' % (
                    epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            epoch_loss = np.mean(loss_arr)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss {:4f}:'.format(best_loss))

    net.load_state_dict(best_model_wts)
    return net


model_fit = train_model(net, fn_loss, optim, num_epoch)
torch.save(model_fit.state_dict(), './model_log/model_weights.pth')
'''
