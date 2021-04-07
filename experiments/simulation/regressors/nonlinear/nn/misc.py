import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

def plot_loss_graphs(logs, args, train=True, test=True, fig=None, ax=None, scale="log"):
    train_logs, test_logs = logs
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.linspace(0, args["epochs"], len(train_logs))
    ax.plot(x, train_logs)
    test_log_interp = interp.interp1d(np.arange(len(test_logs)), test_logs)
    test_log_stretch = test_log_interp(np.linspace(0, len(test_logs)-1, x.size))
    if scale == "log":
        ax.set_yscale('log')
    ax.plot(x, test_log_stretch)
    ax.grid()
