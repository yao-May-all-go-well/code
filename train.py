import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

def evaluate_accuracy(data_loader, model, loss_fn, device):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss_sum += loss.cpu().item()
            acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
    return acc_sum / n, loss_sum / len(data_loader)

def plot_metrics(epochs, train_ls, val_ls, test_ls, train_acc, val_acc, test_acc):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_ls, label='Train Loss')
    plt.plot(epochs, val_ls, label='Validation Loss')
    plt.plot(epochs, test_ls, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def train_with_early_stopping(net, train_loader, val_loader, test_loader, batch_size, loss_fn, optimizer, device, num_epochs, scheduler=None, patience=5):
    net = net.to(device)
    print("training on", device)
    train_ls_list, val_ls_list, test_ls_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        net.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        val_acc, val_loss = evaluate_accuracy(val_loader, net, loss_fn, device)
        test_acc, test_loss = evaluate_accuracy(test_loader, net, loss_fn, device)

        if scheduler:
            scheduler.step()

        train_ls_list.append(train_l_sum / batch_count)
        train_acc_list.append(train_acc_sum / n)
        val_ls_list.append(val_loss)
        val_acc_list.append(val_acc)
        test_ls_list.append(test_loss)
        test_acc_list.append(test_acc)

        print('epoch %d, train loss %.4f, train acc %.3f, val loss %.4f, val acc %.3f, test loss %.4f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, val_loss, val_acc, test_loss, test_acc, time.time() - start))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = net.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            net.load_state_dict(best_model)
            break

    plot_metrics(range(1, epoch + 2), train_ls_list, val_ls_list, test_ls_list, train_acc_list, val_acc_list, test_acc_list)
