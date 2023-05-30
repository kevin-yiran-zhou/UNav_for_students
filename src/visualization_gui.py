import tkinter as tk
from tkinter import *
import numpy as np
from tkinter import ttk,filedialog
from tkinter.messagebox import showinfo
from ttkbootstrap import Style

import cv2
from PIL import Image, ImageTk, ImageDraw, ImageOps

from track import Hloc
from navigation import Trajectory,command_type0,command_type1
from visualization.destination_selection import Destination_window

import argparse
from os.path import dirname,join,exists,realpath
from os import listdir
import yaml
from conversation import Server,Connected_Client,utils
import loader

class Main_window(ttk.Frame):
    image_types=['.jpg', '.png', '.jpeg', '.JPG', '.PNG']
    def __init__(self, master, map_data=None,hloc=None,trajectory=None,**config):
        ttk.Frame.__init__(self, master=master)
        self.config=config
        self.map_data=map_data
        self.keyframe_name=map_data['keyframe_name']
        self.keyframe_location=map_data['keyframe_location']
        self.destination_list_name,self.destination_list_location=[],[]
        anchor_name=map_data['anchor_name']
        anchor_location=map_data['anchor_location']
        ind=0
        while anchor_name[ind][0]!='w':
            self.destination_list_name.append(anchor_name[ind])
            self.destination_list_location.append(anchor_location[ind])
        self.destination_list_name=map_data['anchor_name']
        self.hloc=hloc
        self.trajectory=trajectory
        self.destination=[]
        self.retrieval=True
        self.__layout_design()

    def __layout_design(self):
        windowWidth = self.master.winfo_reqwidth()
        windowHeight = self.master.winfo_reqheight()
        self.positionRight = int(self.master.winfo_screenwidth() / 2 - windowWidth / 2)
        self.positionDown = int(self.master.winfo_screenheight() / 2 - windowHeight / 2)
        self.master.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.master.title('Visualization')
        self.pack(side="left", fill="both", expand=False)
        self.master.geometry('2000x1200')
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(3, pad=7)
        self.master.rowconfigure(3, weight=1)
        self.master.rowconfigure(6, pad=7)
        """
        Localization image list
        """

        lbl = Label(self, text="Query frames:")
        lbl.grid(row=0, column=0, sticky=W, pady=4, ipadx=2)

        self.testing_image_folder=join(self.config['IO_root'],self.config['testing_image_folder'])

        self.query_names=listdir(self.testing_image_folder)

        var2 = tk.StringVar()
        self.lb = tk.Listbox(self, listvariable=var2)

        self.scrollbar = Scrollbar(self, orient=VERTICAL)
        self.lb.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.lb.yview)

        self.lb.bind('<Double-Button-1>', lambda event, action='double':
        self.animate(action))
        self.lb.bind('<Up>', lambda event, action='up':
        self.animate(action))
        self.lb.bind('<Down>', lambda event, action='down':
        self.animate(action))
        self.testing_image_list = [i.split('.')[0] for i in sorted(self.query_names)]
        for i in self.testing_image_list:
            self.lb.insert('end', i)

        self.scrollbar.grid(row=7, column=1, columnspan=2, rowspan=4, padx=2, sticky='sn')

        """
        Function area
        """

        self.lb.grid(row=7, column=0, columnspan=1, rowspan=4, padx=2,
                     sticky=E + W + S + N)

        self.c=ttk.LabelFrame(self, text='Retrieval animation')
        self.c.grid(row=6, column=0, columnspan=2, padx=2,sticky=E + W)
        self.v1 = tk.IntVar()
        self.retrieval=True
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun on',
                                  command=self.__retrieval_setting,
                                  variable=self.v1,
                                  value=0).pack(side=LEFT)
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun off',
                                  command=self.__retrieval_setting,
                                  variable=self.v1,
                                  value=1).pack(side=RIGHT)
        self.shift = tk.Frame(self)
        self.shift.grid(row=14, column=0, padx=10, pady=1, sticky='we')
        scale = Label(self.shift, text='Retrieval number:')
        scale.pack(side=LEFT)
        self.e1 = tk.Entry(self.shift, width=3, justify='left')
        self.e1.pack(side=LEFT)
        self.e1.insert(END, '10')
        self.shift1 = tk.Frame(self)
        self.shift1.grid(row=12, column=0, padx=10, pady=1, sticky='we')
        scale = Label(self.shift1, text='Plotting scale:')
        scale.pack(side=LEFT)
        self.e2 = tk.Entry(self.shift1, width=5, justify='right')
        self.e2.insert(END, '0.1')
        self.e2.pack(side=LEFT)
        scale = Label(self.shift1, text='\'/pixel')
        scale.pack(side=LEFT)
        # ---------------------------------------------------
        separatorv1 = ttk.Separator(self, orient='vertical')
        separatorv1.grid(row=0, column=3, padx=10, columnspan=1, rowspan=70, sticky="sn")
        ebtn = tk.Button(self, text='Save Animation', width=16, command=self.gif_generator)
        ebtn.grid(row=15, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Help', width=16, command=self.__help_info)
        ebtn.grid(row=16, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Clear Destination', width=16, command=self.__clear_destination)
        ebtn.grid(row=17, column=0, padx=10, columnspan=1, rowspan=1)

        """
        Result area
        """

        location_config=self.config['location']
        floorplan_path=join(self.config['IO_root'],'data',location_config['place'],location_config['building'],location_config['floor'])

        fl_selected = filedialog.askopenfilename(initialdir=floorplan_path, title='Select Floor Plan')
        floorplan = Image.open(fl_selected)
        floorplan_draw = ImageDraw.Draw(floorplan)
        self.floorplan=floorplan.copy()
        
        for x,y in self.keyframe_location:
            floorplan_draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))

        self.floorplan_with_keyframe = floorplan.copy()
        width, height = floorplan.size
        self.plot_scale =width/3400
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        floorplan = floorplan.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(floorplan)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
                            self.select_destination(action))

    def __clear_destination(self):
        self.destination=[]
        self.set_destination()

    def __retrieval_setting(self):
        if self.v1.get()==0:
            self.retrieval=True
        else:
            self.retrieval=False

    def gif_generator(self):
        pass

    def __help_info(self):
        self.info = tk.Toplevel(self.master)
        self.info.geometry('800x650')
        self.info.title('Instruction')
        self.info.geometry("+{}+{}".format(self.positionRight - 300, self.positionDown - 200))
        # This will create a LabelFrame
        label_frame1 = LabelFrame(self.info, height=100, text='Steps')
        label_frame1.pack(expand='yes', fill='both')

        label1 = Label(label_frame1, text='1. Double click floor plan to pick a destination point.')
        label1.place(x=0, y=5)

        label2 = Label(label_frame1, text='2. Choose a query in query list to see localization.')
        label2.place(x=0, y=35)

        label3 = Label(label_frame1,
                       text='3. Click and see retrieved images and double click pairs to check local matches.')
        label3.place(x=0, y=65)

        label4 = Label(label_frame1, text='4. Repeat above to see other query results.')
        label4.place(x=0, y=95)

        label_frame1 = LabelFrame(self.info, height=400, text='Buttons')
        label_frame1.pack(expand='yes', fill='both', side='bottom')

        label_1 = LabelFrame(label_frame1, height=60, text='Retrieval animation')
        label_1.place(x=5, y=23)
        label1 = Label(label_1, text='Turn on or turn off the display of retrieval images')
        label1.pack()

        label_2 = LabelFrame(label_frame1, height=40, text='Plotting scale')
        label_2.place(x=5, y=70)
        label2 = Label(label_2, text='How many foot of per pixel.')
        label2.pack()

        label_3 = LabelFrame(label_frame1, height=40, text='Retrieval number')
        label_3.place(x=5, y=117)
        label3 = Label(label_3,
                       text='How many retrieved images to use.')
        label3.pack()

        label_4 = LabelFrame(label_frame1, height=60, text='Save Animation')
        label_4.place(x=5, y=164)
        label4 = Label(label_4,
                       text='Save the localization results.')
        label4.pack()
    
    def __star_vertices(self,center,r):
        out_vertex = [(r*self.plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[0],
                       r*self.plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[1]) for k in range(5)]
        r = r/2
        in_vertex = [(r*self.plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2 ) + center[0],
                      r*self.plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2 ) + center[1]) for k in range(5)]
        vertices = []
        for i in range(5):
            vertices.append(out_vertex[i])
            vertices.append(in_vertex[i])
        vertices = tuple(vertices)
        return vertices
        
    def set_destination(self):
        floorplan=self.floorplan.copy()
        draw_floorplan = ImageDraw.Draw(floorplan)
        for i,keyframe_name in enumerate(self.keyframe_name):
            if keyframe_name not in self.destination:
                x,y=self.keyframe_location[i]
                draw_floorplan.ellipse((x - 2*self.plot_scale, y - 2*self.plot_scale, x + 2*self.plot_scale, y + 2*self.plot_scale), fill=(0, 255, 0))

        for d in self.destination:
            x, y =self.keyframe_location[self.keyframe_name.index(d)]
            vertices = self.__star_vertices([x, y],30)
            draw_floorplan.polygon(vertices, fill='red')

        self.floorplan_with_keyframe = floorplan.copy()

        width, height = floorplan.size
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        floorplan = floorplan.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(floorplan)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
        self.select_destination(action))
    
    def visualize_result(self,image):
        draw=ImageDraw.Draw(image)
        l=60*self.plot_scale
        x_,y_=50*self.plot_scale,l
        ang=0
        x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
        draw.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(50, 0, 106))
        draw.line([(x_, y_), (x1, y1)], fill=(50, 0, 106), width=int(10*self.plot_scale))
        im=np.array(image)
        im=cv2.putText(im, 'Estimation pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
        image=Image.fromarray(im)
        draw = ImageDraw.Draw(image)
        if self.GT:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
            draw.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(255, 0, 255))
            draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 255), width=int(10*self.plot_scale))
            im = np.array(image)
            im = cv2.putText(im, 'Ground truth pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2**self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        if self.retrieval:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 20*self.plot_scale * np.sin(ang), y_ - 20 *self.plot_scale* np.cos(ang)
            draw.ellipse((x_ - 10*self.plot_scale, y_ - 10*self.plot_scale, x_ + 10*self.plot_scale, y_ + 10*self.plot_scale), fill=(255, 0, 0))
            draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 0), width=int(7*self.plot_scale))
            im = np.array(image)
            im = cv2.putText(im, 'Similar images', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        if len(self.destination)>0:
            l+=70*self.plot_scale
            vertices = self.__star_vertices([50*self.plot_scale, l], 30)
            draw.polygon(vertices, fill='red')
            im = np.array(image)
            im = cv2.putText(im, 'Destination', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        draw.rectangle([(10,5),(400+100*self.plot_scale,l+40*self.plot_scale)],outline='black',width=int(2*self.plot_scale))
        return draw,image

    def animate(self,action):
        """
        Show testing image in GUI
        """
        self.value = self.lb.get(self.lb.curselection())
        if action == 'up':
            i = self.testing_image_list.index(self.value)
            if i > 0:
                self.value = self.testing_image_list[i - 1]
        if action == 'down':
            i = self.testing_image_list.index(self.value)
            if i < (len(self.testing_image_list) - 1):
                self.value = self.testing_image_list[i + 1]
        for type in self.image_types:
            q_path=join(self.testing_image_folder, self.value + type)
            if exists(q_path):
                testing_image = Image.open(q_path)
                break
        
        width, height = testing_image.size
        scale = 210 / width
        newsize = (210, int(height * scale))

        timg=testing_image.copy()
        timg = timg.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(timg)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=1, column=0, columnspan=1, padx=10, rowspan=5, sticky="snew")

        """
        Localize testing image
        """
        scale = 640 / width
        newsize = (640, int(height * scale))
        testing_image = testing_image.resize(newsize)
        testing_image=np.array(testing_image)
        testing_image=cv2.cvtColor(testing_image,cv2.COLOR_BGR2RGB)

        pose = self.hloc.get_location(testing_image) # Localize image

        """
        Navigation
        """
        if pose and len(self.destination)>0:
            paths=[]
            for destination_id in self.destination:
                path_list=self.trajectory.calculate_path(pose[:2], destination_id)
                if len(path_list)>0:
                    paths.append(self.keyframe_location[self.keyframe_name.index(destination_id)])
                    paths+=path
            print(paths)
        """
        Write camera pose on the floor plan
        """
        floorplan_with_keyframe=self.floorplan_with_keyframe.copy()
        draw, im = self.prepare_image(im)

    def select_destination(self,w):
        self.newWindow = tk.Toplevel(self.master)
        self.app1 = Destination_window(self.newWindow, parent=self)



def main(root,hloc_config,visual_config):
    map_data=loader.load_data(visual_config)
    hloc = Hloc(root, map_data, hloc_config)
    trajectory=Trajectory(map_data)

    style = Style(theme='darkly')
    root = style.master
    Main_window(root, map_data=map_data,hloc=hloc,trajectory=trajectory,**visual_config)
    root.mainloop()

if __name__=='__main__':
    root = dirname(realpath(__file__)).replace('/src','')
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visual_config', type=str, default='configs/visualization.yaml')
    parser.add_argument('-l', '--hloc_config', type=str, default='configs/hloc.yaml')
    args = parser.parse_args()
    with open(args.hloc_config, 'r') as f:
        hloc_config = yaml.safe_load(f)
    with open(args.visual_config, 'r') as f:
        visual_config = yaml.safe_load(f)
    main(root,hloc_config,visual_config)