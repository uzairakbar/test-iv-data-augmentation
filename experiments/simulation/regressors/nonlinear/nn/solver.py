import torch

ARGS = {"epochs" : 150,
        "device" : "cpu",
        "batch_size" : 256,
        "print_logs" : False,
        "log_interval" : 100,
        "train_val_split" : 0.2,
        "lr" : 0.0001,
        "width" : 5,
        "layers" : 3,
        "optimizer" : "adam"}

def test(model,
         epoch,
         test_loader,
         lamda = 0.0,
         **kwargs):
    if not kwargs:
        kwargs = ARGS

    device = torch.device(kwargs["device"])
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y, g) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            g = g.to(device)

            y_pred = model(x)
            loss = iv_loss(y_pred, y, g, lamda)

            test_loss += loss.item()
    
    test_loss *= kwargs["batch_size"]/len(test_loader)
    if kwargs["print_logs"]:
        print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def train_epoch(model,
                epoch,
                train_loader,
                test_loader,
                lamda = 0.0,
                **kwargs):
    if not kwargs:
        kwargs = ARGS
    optimizer = torch.optim.Adam(model.parameters(), lr = kwargs["lr"])

    train_loss_log = []

    device = torch.device(kwargs["device"])
    model.train()
    train_loss = 0
    for batch_idx, (x, y, g) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        g = g.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = iv_loss(y_pred, y, g, lamda)

        loss.backward()
        train_loss += loss.item()
        train_loss_log.append(loss.item())
        optimizer.step()

        if batch_idx % kwargs["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))
    if kwargs["print_logs"]:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss * kwargs["batch_size"] / len(train_loader.dataset)))

    test_loss = test(model, epoch, test_loader, lamda, **kwargs)

    model.eval()
    return model, train_loss_log, test_loss

def solver(model,
           epochs,
           train_loader,
           test_loader,
           lamda = 0.0,
           **kwargs):
    if not kwargs:
        kwargs = ARGS
    test_loss_log = []
    train_loss_log = []

    test_loss = test(model,
                    0,
                    test_loader,
                    **kwargs)
    test_loss_log.append(test_loss)

    for epoch in range(1, epochs + 1):
        model, train_loss, test_loss = train_epoch(model,
                                  epoch,
                                  train_loader,
                                  test_loader,
                                  lamda,
                                  **kwargs)
        test_loss_log += [test_loss]
        train_loss_log += train_loss
    return model, train_loss_log, test_loss_log
