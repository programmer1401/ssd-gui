import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class create_counter():
    def __init__(self, init):
        self.num = init
        self.it = self.increase()

    def increase(self):  # 定义一个还有自然数算法的生成器,企图使用next来完成不断调用的递增
        while True:
            self.num = self.num + 1
            yield self.num

    def counter(self):  # 再定义一内函数
        return next(self.it)  # 调用生成器的值,每次调用均自增


# a = create_counter(0)
# print(a.counter(), a.counter())

class CreateCanvas:
    def __init__(self, Frame):
        # 设置中文显示字体
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
        mpl.rcParams['axes.unicode_minus'] = False  # 负号显示
        # 创建绘图对象f figsize=(a, b)：a 为图形的宽， b 为图形的高，单位为英寸
        # px, py = a * dpi, b * dpi
        # px=470 py=430 dpi=100
        self.figure = plt.figure(linewidth=0.3, edgecolor='grey', frameon=True)
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.figure, Frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=500, y=130, height=100, width=100)

        # 把matplotlib绘制图形的导航工具栏显示到tkinter窗口上
        toolbar = NavigationToolbar2Tk(self.canvas, Frame)
        toolbar.update()
        self.canvas._tkcanvas.place(x=500, y=140, height=430, width=470)

    def get_figure(self):
        return self.figure

    def get_canvas(self):
        return self.canvas
