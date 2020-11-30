import tkinter as tk
from tkinter import ttk
import tkinter.messagebox

import numpy as np
from PIL import ImageTk, Image

from data import VOC_CLASSES
from gui.evalData import EvalData

ERROR_TITLE = ["输入错误！", "运行错误！"]
INFO_TITLE = ["状态信息", "提示"]


class EvalWin:
    def __init__(self, window):
        self.window = window
        # self.window = tk.Tk()
        # self.window.title("ssd 可视化")
        # self.window.geometry("1000x618")

        self.finish = False

        # --------------------标题
        tk.Label(self.window, text="基于PyTorch的目标检测—模型验证", font=("STSONG", 23, "bold")).place(x=270, y=20)
        tk.Label(self.window, text="参数设置", font=("STSONG", 17, "bold")).place(x=200, y=80)
        tk.Label(self.window, text="模型验证结果", font=("STSONG", 17, "bold")).place(x=170, y=320)
        tk.Label(self.window, text="PR曲线", font=("STSONG", 17, "bold")).place(x=690, y=80)

        # --------------------左边，包括三项：设置参数、开始验证按钮、显示AP列表
        # 是否使用 cuda
        tk.Label(self.window, text="是否使用cuda：", font=("STSONG", 14)).place(x=90, y=160)
        cuda = tk.IntVar()
        cuda.set(2)
        tk.Radiobutton(self.window, text="是", value=1, variable=cuda, font=("STSONG", 14)).place(x=250, y=160)
        tk.Radiobutton(self.window, text="否", value=2, variable=cuda, font=("STSONG", 14)).place(x=300, y=160)

        # 按钮
        self.btn_start_eval = tk.Button(self.window, text="开始验证", height="1", width="20",
                                        command=lambda: self.start_eval(cuda.get()))
        self.btn_start_eval.place(x=230, y=210)

        # 左下边：mAP标题
        # 选择框，下拉列表，选择想要展示的 物体种类的 P-R 曲线
        tk.Label(self.window, text="请选择查看物体的种类：", font=("STSONG", 14)).place(x=90, y=390)
        self.choose = ttk.Combobox(self.window, width="20")
        self.choose.place(x=230, y=430)
        box_values = ["请选择…"] + list(VOC_CLASSES)
        self.choose['value'] = box_values
        self.choose.current(0)

        # 为每个选项绑定点击事件
        self.choose.bind("<<ComboboxSelected>>", self.choose_image)

        # mAP
        self.mAP = tk.Label(self.window, text="", font=("STSONG", 14))
        self.mAP.place(x=90, y=460)

        # --------------------右边：显示 PR曲线
        # 帮助按钮
        self.help_pr_btn = tk.Button(self.window, text="?", height="1", width="2",
                                        command=lambda: self.pr_info())
        self.help_pr_btn.place(x=780, y=80)
        # 显示图片
        self.imglabel = tk.Label(self.window, relief="groove", bg="white")
        self.imglabel.place(x=500, y=130, height=430, width=470)

        # self.window.mainloop()

    def start_eval(self, cuda):
        cuda = True if cuda == 1 else False
        net = EvalData(cuda=cuda)
        net.start_eval()
        self.finish = True

        box_values = ["请选择…"]
        for i in range(len(VOC_CLASSES)):
            box_values += [VOC_CLASSES[i] + "(AP:" + str(net.aps[i]) + ")"]
        self.choose['value'] = box_values
        self.mAP['text'] = 'Mean AP = {:.4f}'.format(np.mean(net.aps))

    def choose_image(self, event):
        if self.choose.get()=="请选择…":
            return

        if self.finish == False:
            tk.messagebox.showerror(title=ERROR_TITLE[1], message="无法找到" + self.choose.get() + "对应的PR曲线，请确认是否已经进行模型验证！")
            return

        global image
        img_path = "D:\WorkSpace\PyCharmSpace\SSD\ssd.pytorch-master\gui\eval\PRcurve\\"
        img_name = self.choose.get() + ".jpg"

        image = ImageTk.PhotoImage(Image.open(img_path + img_name).resize((420, 420)))
        self.imglabel.config(image=image)

    def pr_info(self):
        tk.messagebox.showinfo(title="PR曲线",
                               message="PR曲线以Recall为横轴，Precision为竖轴，PR曲线越靠近坐标(1,1)，表明模型的分类效果越好。\n\n"
                                       "其中：\nPrecision称为查准率，指所有预测中检测正确的概率；\nRecall称为查全率，指所有正样本中正确识别的概率")

if __name__ == "__main__":
    EvalWin()
