import cv2
import tkinter as tk
import tkinter.messagebox
import tkinter.filedialog

from PIL import ImageTk, Image

from data import BaseTransform
from gui.testData import test_voc, test_net


class TestWin:
    def __init__(self,window):
        self.window = window
        # self.window = tk.Tk()
        # self.window.title("ssd 可视化")
        # self.window.geometry("700x500")


        # 输入 是否使用 cuda
        tk.Label(self.window, text="是否使用cuda：", font=("STSONG", 14)).place(x=90, y=500)
        cuda = tk.IntVar()
        cuda.set(1)
        tk.Radiobutton(self.window, text="是", value=1, variable=cuda, font=("STSONG", 14)).place(x=220, y=500)
        tk.Radiobutton(self.window, text="否", value=2, variable=cuda, font=("STSONG", 14)).place(x=270, y=500)

        # 图片
        self.imglabel = tk.Label(self.window, relief="groove", bg="white")
        self.imglabel.place(x=15, y=15, height=470, width=970)

        # 按钮
        self.btn_take_photo = tk.Button(self.window, text="调用摄像头",height="1", width="15", font=("STSONG", 14),
                                        command=lambda: self.change_btn_status(cuda.get()))
        self.btn_take_photo.place(x=600, y=500)
        self.btn_select_photo = tk.Button(self.window, text="从相册选择……", height="1", width="15", font=("STSONG", 14),
                                          command=lambda: self.select_photo(cuda.get())).place(x=800, y=500)

        # self.window.mainloop()


    def opncv2PIL(self, imageBGR):
        global detec_image
        # opencv 打开图片为 ndarray 格式 ，转换为 ImageTk.PhotoImage 格式，才能显示在控件里
        # 由于 OpenCV使用的是BGR模式，而PIL使用的是RGB模式 所以需要转换
        imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
        trans_img = Image.fromarray(imageRGB)
        detec_image = ImageTk.PhotoImage(image=trans_img)
        # detec_image = ImageTk.PhotoImage(image=trans_img.resize((420, 420)))
        self.imglabel.config(image=detec_image)

    def change_btn_status(self, cuda):
        if self.btn_take_photo["text"]=="停止调用摄像头":
            self.btn_take_photo["text"] = "调用摄像头"
            self.cap.release()

        elif self.btn_take_photo["text"]=="调用摄像头":
            self.btn_take_photo["text"] = "停止调用摄像头"
            self.take_photo(cuda)


    def take_photo(self, cuda):
        cuda = True if cuda == 1 else False
        # 初始化模型
        self.net = test_voc("E:\Code\Examples\Gao\ssd.pytorch-master\weights\ssd300_mAP_77.43_v2(1).pth", cuda)

        tk.messagebox.showinfo(title="设备信息", message="开始调用摄像头进行识别……")

        # 参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # 检测
            test_net(self.net, True, frame, BaseTransform(self.net.size, (104, 117, 123)))

            self.opncv2PIL(frame)
            self.imglabel.update()

        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         self.cap.release()
        #         break
        #
        # cv2.destroyWindow("camera-capture")

    def select_photo(self, cuda):
        # 弹出对话框，选择图片
        path = tk.StringVar()
        select_img_path = tk.Entry(self.window, state="readonly", text=path)
        if select_img_path != "":
            path.set(tk.filedialog.askopenfile().name)

        # 判断选择文件的类型是否能处理
        file_name_spilt = select_img_path.get().split(".")
        if file_name_spilt[-1] !="jpg" and file_name_spilt[-1] !="png":
            tk.messagebox.showerror(title="文件类型错误！", message="不能处理该类型的图片！请选择以.jpg或.png为后缀的图片！")
            return

        # 显示图片
        global origion_image
        origion_image = ImageTk.PhotoImage(Image.open(select_img_path.get()))
        self.imglabel.config(image=origion_image)
        self.imglabel.update()

        # 将图片传入模型，开始测试
        cuda = True if cuda == 1 else False
        img_input_model = cv2.imread(select_img_path.get())

        # 初始化模型
        self.net = test_voc("E:\Code\Examples\Gao\ssd.pytorch-master\weights\ssd300_mAP_77.43_v2(1).pth", cuda)
        # 检测
        test_net(self.net, cuda, img_input_model, BaseTransform(self.net.size, (104, 117, 123)))

        # 图片转换
        self.opncv2PIL(img_input_model)



if __name__ == "__main__":
    TestWin()
