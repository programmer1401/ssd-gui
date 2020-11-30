import tkinter as tk
from tkinter import ttk

from gui.window.evalGUI import EvalWin
from gui.window.testGUI import TestWin
from gui.window.trainGUI import TrainWin

class Window:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("基于PyTorch的SSD的研究与实现")
        self.window.geometry("1000x618")
        # 前两个参数是窗口的宽度和高度。 最后两个参数是x和y屏幕坐标。
        # self.window.geometry("2000x1236")

        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('微软雅黑', '11'),padding=[5, 5, 5, 5], width=10)

        self.tab = ttk.Notebook(self.window)

        self.frame1 = tk.Frame(self.tab)
        self.tab1 = self.tab.add(self.frame1, text="数据训练")
        TrainWin(self.frame1)

        self.frame2 = tk.Frame(self.tab)
        self.tab2 = self.tab.add(self.frame2, text="模型验证")
        EvalWin(self.frame2)

        self.frame3 = tk.Frame(self.tab)
        self.tab3 = self.tab.add(self.frame3, text="实时检测")
        TestWin(self.frame3)

        self.tab.pack(expand=True, fill=tk.BOTH)

        # 设置选中 数据训练
        self.tab.select(self.frame1)

        self.window.mainloop()


if __name__ == "__main__":
    Window()
