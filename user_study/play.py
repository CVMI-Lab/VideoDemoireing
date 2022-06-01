import tkinter
import tkinter.messagebox
from tkinter import filedialog
from video_select import VideoSelector
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import cv2
import os
import time

# upper_video = "v10009.avi"
# lower_video = "v10004.avi"
video_folder = "demoire_videos" # contains results from different methods, e.g., demoire_videos/method1/, demoire_videos/method2/ ... 



class UserStudyTkinter:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title("camera")
        self.window_size_x = self.root.winfo_screenwidth() - 50
        self.window_size_y = self.root.winfo_screenheight() - 50
        self.root.geometry('%dx%d' % (self.window_size_x, self.window_size_y))
        self.root.configure(background='gray')
        os.makedirs("results", exist_ok=True)
        exe_time = time.asctime(time.localtime(time.time()))
        self.file = open(os.path.join("results", "{}.txt".format(exe_time)), "w")

        self.VS = VideoSelector(video_folder)

        self.canvas_width = self.window_size_y
        self.canvas_height = int(self.window_size_y/2)
        self.fps = 20
        self.sz = (int(self.canvas_width*0.94), int(self.canvas_height*0.9))
        self.padding = int(self.canvas_width / 30)

        img_gray = np.full((100, 100, 3), 128).astype(np.uint8)

        self.upper_color, self.lower_color = "gray", "gray"
        self.upper_selected, self.lower_selected = False, False
        self.video_upper_path, self.video_lower_path = "", ""
        self.xVariable = tkinter.StringVar()
        self.sbox = tkinter.Spinbox(self.root, from_=15, to=31, increment=3, textvariable=self.xVariable,
                                    justify="right", command=self.get_spine_box_value,  width=1400, borderwidth=1)
        self.sbox.pack()

        self.canvas_upper = tkinter.Canvas(self.root, bg=self.upper_color, width=self.canvas_width, height=self.canvas_height)
        self.canvas_upper.pack()
        self.img_upper = ImageTk.PhotoImage(self.convert_img(img_gray))
        self.canvas_upper.create_image(self.padding, self.padding, anchor=NW, image=self.img_upper)
        self.canvas_upper.place(x=0, y=0, anchor=NW)

        self.canvas_lower = tkinter.Canvas(self.root, bg=self.lower_color, width=self.canvas_width, height=self.canvas_height)
        self.canvas_lower.pack()
        self.img_lower = ImageTk.PhotoImage(self.convert_img(img_gray))
        self.canvas_lower.create_image(self.padding, self.padding, anchor=NW, image=self.img_lower)
        self.canvas_lower.place(x=0, y=int(self.window_size_y/2), anchor=NW)

        # 按钮
        self.btn_width = int(self.window_size_x / 80)
        self.btn_height = int(self.window_size_y / 160)
        self.btn_left_ord = int((self.window_size_x - self.canvas_width)/4)
        self.btn_right_ord = self.btn_left_ord*3 + self.canvas_width
        self.btn_select_upper = tkinter.Button(self.root, command=self.select_upper, text="Select upper video",
                                               width=self.btn_width, height=self.btn_height)
        self.btn_select_lower = tkinter.Button(self.root, command=self.select_lower, text="Select lower video",
                                               width=self.btn_width, height=self.btn_height)
        self.btn_select_none = tkinter.Button(self.root, command=self.select_none, text="Hard to distinguish",
                                              width=self.btn_width, height=self.btn_height)
        self.btn_video_replay = tkinter.Button(self.root, command=self.video_replay_set, text="Replay",
                                               width=self.btn_width, height=self.btn_height)
        self.btn_next = tkinter.Button(self.root, command=self.video_play, text="Begin",
                                       width=self.btn_width, height=self.btn_height)

        self.begin_flag = False
        self.playing_flag = False
        self.stop = False

    def select_upper(self):
        if self.upper_selected:
            self.upper_selected = False
            self.upper_color = "gray"
            self.btn_select_upper['text'] = "Select upper video"
        else:
            self.upper_selected = True
            self.upper_color = "purple"
            self.btn_select_upper['text'] = "Unselect"
            self.btn_select_lower['text'] = "Select lower video"
            if self.lower_selected:
                self.lower_selected = False
                self.lower_color = "gray"
        self.set_bg_color("upper", self.upper_color)
        self.set_bg_color("lower", self.lower_color)
        if not self.playing_flag:
            self.root.update_idletasks()
            self.root.update()

    def select_lower(self):
        if self.lower_selected:
            self.lower_selected = False
            self.lower_color = "gray"
            self.btn_select_lower['text'] = "Select lower video"
        else:
            self.lower_selected = True
            self.lower_color = "purple"
            self.btn_select_lower['text'] = "Unselect"
            self.btn_select_upper['text'] = "Select upper video"
            if self.upper_selected:
                self.upper_selected = False
                self.upper_color = "gray"
        self.set_bg_color("upper", self.upper_color)
        self.set_bg_color("lower", self.lower_color)
        if not self.playing_flag:
            self.root.update_idletasks()
            self.root.update()

    def select_none(self):
        self.lower_color, self.upper_color = "gray", "gray"
        self.lower_selected, self.upper_selected = False, False
        self.set_bg_color("upper", self.upper_color)
        self.set_bg_color("lower", self.lower_color)
        self.btn_select_upper['text'] = "Select upper video"
        self.btn_select_lower['text'] = "Select lower video"

    def set_btn_selected_disable(self):
        self.btn_select_upper['state'] = DISABLED
        self.btn_select_lower['state'] = DISABLED
        self.btn_select_none['state'] = DISABLED

    def set_btn_selected_normal(self):
        self.btn_select_upper['state'] = NORMAL
        self.btn_select_lower['state'] = NORMAL
        self.btn_select_none['state'] = NORMAL

    def get_spine_box_value(self):
        self.fps = int(self.xVariable.get())
        print(self.fps)

    def gui_set(self):
        self.btn_select_upper.place(x=int(self.window_size_x*0.7), y=int(self.window_size_y*0.2), anchor=CENTER)
        self.btn_select_lower.place(x=int(self.window_size_x*0.7), y=int(self.window_size_y*0.8), anchor=CENTER)
        self.btn_select_none.place(x=int(self.window_size_x*0.7), y=int(self.window_size_y*0.5), anchor=CENTER)
        self.btn_video_replay.place(x=int(self.window_size_x*0.9), y=int(self.window_size_y*0.3), anchor=CENTER)
        self.btn_next.place(x=int(self.window_size_x*0.9), y=int(self.window_size_y*0.7), anchor=CENTER)
        self.btn_select_upper['state'] = DISABLED
        self.btn_select_lower['state'] = DISABLED
        self.btn_select_none['state'] = DISABLED
        self.btn_video_replay['state'] = DISABLED

    def set_bg_color(self, canvas, color):
        if canvas == "upper":
            self.canvas_upper.configure(bg=color)
        elif canvas == "lower":
            self.canvas_lower.configure(bg=color)
        else:
            print("Wrong canvas!")

    def convert_img(self, frame):
        return Image.fromarray(cv2.cvtColor(cv2.resize(frame, self.sz), cv2.COLOR_RGB2BGR))

    def canvas_update(self, f1, f2):
        self.img_upper = ImageTk.PhotoImage(self.convert_img(f1))
        self.img_lower = ImageTk.PhotoImage(self.convert_img(f2))
        self.canvas_upper.create_image(self.padding, self.padding, anchor=NW, image=self.img_upper)
        self.canvas_lower.create_image(self.padding, self.padding, anchor=NW, image=self.img_lower)
        self.root.update_idletasks()
        self.root.update()

    def video_play(self):
        if not self.begin_flag:
            self.video_lower_path, self.video_upper_path = self.VS.select()
            self.begin_flag = True
            self.run_video()
        else:
            self.save_result()
            self.video_lower_path, self.video_upper_path = self.VS.select()
            self.select_none()
            self.run_video()

    def run_video(self):
        self.btn_next['text'] = "Next"
        self.btn_select_lower['state'] = NORMAL
        self.btn_select_none['state'] = NORMAL
        self.btn_video_replay['state'] = NORMAL
        self.btn_select_upper['state'] = NORMAL
        self.show_video(self.video_upper_path, self.video_lower_path)

    def video_replay_set(self):
        if not self.playing_flag:
            self.show_video(self.video_upper_path, self.video_lower_path)
        else:
            self.stop = True

    def show_video(self, video_path1, video_path2):
        self.btn_next['state'] = DISABLED
        self.btn_video_replay['text'] = "Stop"
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)
        self.playing_flag = True
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if self.stop:
                self.stop = False
                break
            if ret1 and ret2:
                self.canvas_update(frame1, frame2)
                time.sleep(1/self.fps)
            else:
                break
        self.playing_flag = False
        self.btn_next['state'] = NORMAL
        self.btn_video_replay['text'] = "Replay"

    def save_result(self):
        assert not (self.upper_selected and self.lower_selected)
        #import pdb; pdb.set_trace()
        if self.upper_selected:
            print("The upper video is better than the lower one: {} > {}".format(self.video_upper_path,
                                                                                 self.video_lower_path))
            self.file.write("{} > {}\n".format(self.video_upper_path, self.video_lower_path))
        elif self.lower_selected:
            print("The lower video is better than the upper one: {} > {}".format(self.video_lower_path,
                                                                                 self.video_upper_path))
            self.file.write("{} > {}\n".format(self.video_lower_path, self.video_upper_path))
        else:
            self.file.write("{} = {}\n".format(self.video_lower_path, self.video_upper_path))
            print("Hard to distinguish!")

    # def video_select(self):
    #     self.set_bg_color("upper", "gray")
    #     path = tkinter.filedialog.askopenfilename(filetype=[("MP4", ".mp4")])
    #     self.lastVideo = path
    #     self.btn_video_replay['state'] = NORMAL
    #     self.flag_select_video = True


if __name__ == '__main__':
    golf = UserStudyTkinter()
    golf.gui_set()
    tkinter.mainloop()
