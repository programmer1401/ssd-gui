import tkinter as tk
from tkinter import *
from gui.trainData import TrainData
from gui.utils import CreateCanvas
import matplotlib.pyplot as plt

class learningRateGUI:
    def __init__(self, window):
        self.window = window
        # self.window = tk.Tk()
        self.window.title("超参数调优——学习率")
        self.window.geometry("1000x618")
        self.createWidget()
        # self.window.mainloop()

    def createWidget(self):
        # learning_rate 的作用
        Label(self.window, text="学习率决定了权值更新的速度",
              font=("STSONG", 20, "bold")).place(x=320, y=20)
        Label(self.window,
              text="学习率小，模型学习速度慢，可能直到模型训练结束，损失函数仍未取到最优，但呈现下降趋势。",
              font=("STSONG", 13)).place(x=100, y=60)
        Label(self.window,
              text="学习率大，模型学习速度快，容易接近最优解。但训练后期，损失函数的值可能一直围绕最小值徘徊，难以取到最优。",
              font=("STSONG", 13)).place(x=100, y=90)

        # 输入
        Label(self.window, text="主干网络：", font=("STSONG", 14)).place(x=90, y=150)
        Label(self.window, text="迭代次数：", font=("STSONG", 14)).place(x=90, y=200)
        Label(self.window, text="批处理大小：", font=("STSONG", 14)).place(x=90, y=250)
        Label(self.window, text="是否使用cuda：", font=("STSONG", 14)).place(x=90, y=300)
        Label(self.window, text="请选择 learning_rate 的范围：", font=("STSONG", 14)).place(x=90, y=350)
        Label(self.window, text="起始值：", font=("STSONG", 14)).place(x=120, y=400)
        Label(self.window, text="终止值：", font=("STSONG", 14)).place(x=120, y=440)
        Label(self.window, text="增长间隔：", font=("STSONG", 14)).place(x=120, y=480)

        basenet = IntVar()
        basenet.set(1)
        Radiobutton(self.window, text="VGG", value=1, variable=basenet, font=("STSONG", 14)).place(x=250, y=150)
        Radiobutton(self.window, text="ResNet", value=2, variable=basenet, font=("STSONG", 14)).place(x=330, y=150)

        # 迭代次数
        iter_num = Entry(self.window, textvariable=tk.StringVar(value="10"))
        iter_num.place(x=250, y=200)

        # 批处理大小
        batch_size = Entry(self.window, textvariable=tk.StringVar(value="2"))
        batch_size.place(x=250, y=250)

        # 是否使用 cuda
        cuda = IntVar()
        cuda.set(2)
        Radiobutton(self.window, text="是", value=1, variable=cuda).place(x=250, y=300)
        Radiobutton(self.window, text="否", value=2, variable=cuda).place(x=330, y=300)

        # 选择 learning_rate 范围
        start = Entry(self.window, textvariable=StringVar(value="0.0001"))
        start.place(x=250, y=400)
        end = Entry(self.window, textvariable=StringVar(value="0.001"))
        end.place(x=250, y=440)
        step = Entry(self.window, textvariable=StringVar(value="0.0002"))
        step.place(x=250, y=480)

        # 按钮
        Button(self.window, text="开始", height="1", width="15",
               command=lambda:self.tuning(basenet.get(), int(iter_num.get()),
                                           int(batch_size.get()), cuda.get(),
                                           float(start.get()), float(end.get()), float(step.get())))\
            .place(x=290, y=530)

        self.matplot = CreateCanvas(self.window)
        self.canvas = self.matplot.get_canvas()
        self.figure = self.matplot.get_figure()

    def tuning(self, basenet, iter_num, batch_size, cuda, start, end, step):
        try:
            self.learning_rate = []
            self.loss_ass = []

            self.start = start
            self.end = end
            self.step = step
            self.learning_rate += [start]

            basenet = "vgg16_reducedfc.pth" if basenet == 1 else 'resnet50-19c8e357.pth'
            cuda = True if cuda == 1 else False

            while self.learning_rate[-1] <= end:
                traindata = TrainData(iter_num, basenet=basenet, batch_size=batch_size, cuda=cuda,
                                      lr=self.learning_rate[-1])
                loss = traindata.train()

                self.loss_ass += [loss[-1]]
                self.learning_rate += [round(self.learning_rate[-1] + step, 6)]

            self.learning_rate = self.learning_rate[:-1]
            self.create_matplotlib()

        except Exception as e:
            tk.messagebox.showerror(title="运行错误！", message=str(e))

    def create_matplotlib(self):
        plt.clf()
        plt.plot(self.learning_rate, self.loss_ass)
        plt.title("损失函数与learning_rate的关系", loc='center', pad=20, fontsize='xx-large', color='red')
        plt.xlabel("learning_rate")
        plt.ylabel("损失函数")
        plt.xticks(rotation=45)
        self.canvas.draw()


if __name__ == '__main__':
    learningRateGUI()

