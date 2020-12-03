import math
import tkinter as tk
from tkinter import *
from gui.trainData import TrainData
from gui.utils import CreateCanvas

import random
import numpy as np

basenet = "vgg16_reducedfc.pth"
batch_size = 2


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
        Label(self.window, text="迭代次数：", font=("STSONG", 14)).place(x=90, y=150)
        Label(self.window, text="批量大小：", font=("STSONG", 14)).place(x=90, y=200)
        Label(self.window, text="是否使用cuda：", font=("STSONG", 14)).place(x=90, y=250)
        Label(self.window, text="请选择 learning_rate 的范围：", font=("STSONG", 14)).place(x=90, y=300)
        Label(self.window, text="起始值：", font=("STSONG", 14)).place(x=120, y=350)
        Label(self.window, text="终止值：", font=("STSONG", 14)).place(x=120, y=390)
        Label(self.window, text="增长间隔：", font=("STSONG", 14)).place(x=120, y=430)

        # 迭代次数
        iter_num = Entry(self.window, textvariable=tk.StringVar(value="10"))
        iter_num.place(x=250, y=150)
        Label(self.window, text=str(batch_size) + "（最优值）").place(x=250, y=200)

        # 是否使用 cuda
        cuda = IntVar()
        cuda.set(1)
        Radiobutton(self.window, text="是", value=1, variable=cuda).place(x=250, y=250)
        Radiobutton(self.window, text="否", value=2, variable=cuda).place(x=300, y=250)

        # 选择 learning_rate 范围
        start = Entry(self.window, textvariable=StringVar(value="0.0001"))
        start.place(x=250, y=350)
        end = Entry(self.window, textvariable=StringVar(value="0.001"))
        end.place(x=250, y=390)
        step = Entry(self.window, textvariable=StringVar(value="0.0002"))
        step.place(x=250, y=430)

        # 按钮
        Button(self.window, text="开始", height="1", width="15",
               command=lambda: self.tuning(int(iter_num.get()), float(start.get()), float(end.get()),
                                           float(step.get()), cuda.get())).place(x=290, y=500)

        self.matplot = CreateCanvas(self.window)
        self.canvas = self.matplot.get_canvas()
        self.figure = self.matplot.get_figure()

    def tuning(self, iter_num, start, end, step, cuda):
        self.learning_rate = []
        self.loss_ass = []

        self.start = start
        self.end = end
        self.step = step

        cuda = True if cuda == 1 else False

        # 如果开始和截止的两个数相差＞100倍，则采用每一段随机采样的方法
        # 0.001-0.01-0.1两段中，在每一段都随机生成同样数量(如，10)的数，对生成的每个数，依次当作一个learning_rate
        start_int = math.log10(self.start)
        end_int = math.log10(self.end)

        if end_int - start_int >= 2:
            is_random = tk.messagebox.askokcancel("提示！", "起始学习率为%f，终止学习率为%f\n两者相差太大，是否分段随机生成学习率？"%(self.start,self.end))

            if is_random:
                for i in range(int(start_int), int(end_int + 1)):
                    produce_point = np.random.rand(10)
                    self.learning_rate += list(produce_point * math.pow(10, i + 1))

                self.learning_rate.sort()

                for k,v in enumerate(self.learning_rate):
                    traindata = TrainData(iter_num, basenet=basenet, batch_size=batch_size, cuda=cuda, lr=v)
                    loss = traindata.train()

                    self.loss_ass += [loss[-1]]

                self.create_matplotlib()
                return

        self.learning_rate += [start]

        while self.learning_rate[-1] < end:
            traindata = TrainData(iter_num, basenet=basenet, batch_size=batch_size, cuda=cuda,
                                  lr=self.learning_rate[-1])
            loss = traindata.train()

            self.loss_ass += [loss[-1]]
            self.learning_rate += [round(self.learning_rate[-1] + step, 6)]

        self.learning_rate = self.learning_rate[:-1]
        self.create_matplotlib()

    def create_matplotlib(self):
        axc = self.figure.add_subplot(111)
        axc.clear()

        axc.plot(self.learning_rate, self.loss_ass)
        axc.set_title("损失函数与learning_rate的关系", loc='center', pad=20, fontsize='xx-large', color='red')
        axc.set_xlabel("learning_rate")
        axc.set_ylabel("损失函数")

        self.canvas.draw()


if __name__ == '__main__':
    learningRateGUI()
