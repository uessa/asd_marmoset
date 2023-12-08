# -*- coding: utf-8 -*-
import datetime
import matplotlib.pyplot as plt
import path

plt.switch_backend("agg")


class Logger:
    def __init__(self, fpath, new=True):
        if not new and fpath.is_file():
            self.f = open(fpath, "a")
            self.f.write("\n")
        else:
            self.f = open(fpath, "w")
        self.f.write(datetime.datetime.today().isoformat() + "\n")

    def __call__(self, msg):
        print(msg)  # stdout
        self.f.write(msg + "\n")
        self.f.flush()

    def __del__(self):
        if self.f is not None:
            self.f.close()

    def close(self):
        self.f.close()
        self.f = None


def plot_loss(path, results_train, results_valid):
    plt.plot(results_train, label="train")
    plt.plot(results_valid, label="valid")
    plt.legend()
    plt.title("loss")
    plt.savefig(path.model / "loss.png")
    plt.clf()
    plt.close()

def plot_acc(path, results_train, results_valid):
    plt.plot(results_train["accuracy"], label="train")
    plt.plot(results_valid["accuracy"], label="valid")
    plt.legend()
    plt.title("accuracy")
    plt.savefig(path.model / "accuracy.png")
    plt.clf()
    plt.close()

