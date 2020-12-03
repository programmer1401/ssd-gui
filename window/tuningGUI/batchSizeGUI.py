from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt

import time
from gui.trainData import TrainData
from gui.utils import CreateCanvas

learning_rate = 0.0002

ERROR_TITLE = ["输入错误！", "运行错误！"]
INFO_TITLE = ["状态信息", "提示"]


class batchSizeGUI():
    def __init__(self, window):
        self.window = window
        # self.window = tk.Tk()
        self.window.title("超参数调优——批处理大小")
        self.window.geometry("1000x618")
        self.createWidget()
        # self.window.mainloop()

        # 标志位，程序是否执行，用于设置 “切换”按钮能否正常使用
        # 0：没有执行过，因此点击按钮需弹出错误提示框
        # 1：第一次执行，不需要清空canvas画布
        # 非1：执行过不止一次，需要清空canvas画布
        self.finish = 0

    def createWidget(self):
        # batch_size 的作用
        Label(self.window, text="批处理大小影响模型的优化程度和训练速度",
              font=("STSONG", 20, "bold")).place(x=250, y=20)
        Label(self.window,
              text="同时其直接影响到GPU内存的使用情况。假如GPU内存不大，该数值最好设置小一点。",
              font=("STSONG", 13)).place(x=100, y=60)

        # 输入
        Label(self.window, text="主干网络：", font=("STSONG", 14)).place(x=90, y=150)
        Label(self.window, text="迭代次数：", font=("STSONG", 14)).place(x=90, y=200)
        Label(self.window, text="学习率大小：", font=("STSONG", 14)).place(x=90, y=250)
        Label(self.window, text="是否使用cuda：", font=("STSONG", 14)).place(x=90, y=300)
        Label(self.window, text="请选择 batch_size 的范围：", font=("STSONG", 14)).place(x=90, y=350)
        Label(self.window, text="起始值：", font=("STSONG", 14)).place(x=120, y=400)
        Label(self.window, text="终止值：", font=("STSONG", 14)).place(x=120, y=440)
        Label(self.window, text="增长间隔：", font=("STSONG", 14)).place(x=120, y=480)

        basenet = IntVar()
        basenet.set(1)
        Radiobutton(self.window, text="VGG", value=1, variable=basenet, font=("STSONG", 14)).place(x=250, y=150)
        Radiobutton(self.window, text="ResNet", value=2, variable=basenet, font=("STSONG", 14)).place(x=330, y=150)

        # 迭代次数
        iter_num = Entry(self.window, textvariable=StringVar(value="10"))
        iter_num.place(x=250, y=200)

        # 学习率
        learning_rate = Entry(self.window, textvariable=StringVar(value="0.0002"))
        learning_rate.place(x=250, y=250)

        # 是否使用 cuda
        cuda = IntVar()
        cuda.set(2)
        Radiobutton(self.window, text="是", value=1, variable=cuda).place(x=250, y=300)
        Radiobutton(self.window, text="否", value=2, variable=cuda).place(x=330, y=300)

        # 选择 batch_size 范围
        start = Entry(self.window, textvariable=StringVar(value="1"))
        start.place(x=250, y=400)
        end = Entry(self.window, textvariable=StringVar(value="10"))
        end.place(x=250, y=440)
        step = Entry(self.window, textvariable=StringVar(value="1"))
        step.place(x=250, y=480)

        # 按钮
        Button(self.window, text="开始", height="1", width="15",
               command=lambda: self.tuning(basenet.get(), int(iter_num.get()),
                                           float(learning_rate.get()), cuda.get(),
                                           int(start.get()), int(end.get())+1, int(step.get())))\
            .place(x=90, y=530)

        self.change_img = Button(self.window, text="切换图像：运行速度", height="1", width="15",
                                 command=lambda: self.create_matplotlib())
        self.change_img.place(x=290, y=530)

        self.matplot = CreateCanvas(self.window)
        self.canvas = self.matplot.get_canvas()
        self.figure = self.matplot.get_figure()

    def tuning(self, basenet, iter_num, learning_rate, cuda, start, end, step):
        try:
            self.time_ass = []
            self.loss_ass = []

            self.start = start
            self.end = end
            self.step = step

            basenet = "vgg16_reducedfc.pth" if basenet == 1 else 'resnet50-19c8e357.pth'
            cuda = True if cuda == 1 else False

            for batch_size in range(start, end, step):
                traindata = TrainData(iter_num, basenet=basenet, batch_size=int(batch_size), cuda=cuda,
                                      lr=learning_rate)
                t0 = time.time()
                loss = traindata.train()
                t1 = time.time()

                self.time_ass += [t1 - t0]
                self.loss_ass += [loss[-1]]

            self.finish += 1
            # 默认显示损失函数图像
            self.create_matplotlib()

        except Exception as e:
            messagebox.showerror(title="运行错误！", message=str(e))

    def create_matplotlib(self):
        if self.finish == 0:
            messagebox.showerror(title=ERROR_TITLE[1], message="程序尚未执行！无法显示图像！")
            return
        elif self.finish != 1:
            self.axc.clear()

        x = range(self.start, self.end, self.step)
        self.axc = self.figure.add_subplot(111)
        self.axc.clear()

        btn_name = self.change_img["text"]
        if btn_name == "切换图像：运行速度":
            plt.setp(self.axc.plot(x, self.loss_ass))
            self.axc.set_title("损失函数与batch_size的关系", loc='center', pad=20, fontsize='xx-large', color='red')
            self.axc.set_xlabel("batch_size")
            self.axc.set_ylabel("损失函数")

            self.change_img["text"] = "切换图像：损失函数"

        elif btn_name == "切换图像：损失函数":
            plt.setp(self.axc.plot(x, self.time_ass))
            self.axc.set_title("运行速度与batch_size的关系", loc='center', pad=20, fontsize='xx-large', color='red')
            self.axc.set_xlabel("batch_size")
            self.axc.set_ylabel("运行速度/sec")

            self.change_img["text"] = "切换图像：运行速度"

        self.canvas.draw()


if __name__ == '__main__':
    batchSizeGUI()
