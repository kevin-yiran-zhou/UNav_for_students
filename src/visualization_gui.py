import tkinter as tk
from tkinter import *
import numpy as np
from tkinter import ttk,filedialog
from tkinter.messagebox import showinfo
from ttkbootstrap import Style

from PIL import Image, ImageTk, ImageDraw, ImageOps

from track import Hloc
from navigation import Trajectory,command_type0,command_type1

import argparse
from os.path import dirname,join,exists,realpath
import yaml
from conversation import Server,Connected_Client,utils
import loader

class Main_window(ttk.Frame):
    def __init__(self, master, map_data=None,hloc=None,trajectory=None,**config):
        ttk.Frame.__init__(self, master=master)
        self.config=config
        self.map_data=map_data
        self.hloc=hloc
        self.trajectory=trajectory

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
        # --------------------------------------------------
        lbl = Label(self, text="Query frames:")
        lbl.grid(row=0, column=0, sticky=W, pady=4, ipadx=2)
        self.query_names=os.listdir(opt.query_dir)
        self.set_query(self.query_names)
        self.c=ttk.LabelFrame(self, text='Retrieval animation')
        self.c.grid(row=6, column=0, columnspan=2, padx=2,sticky=E + W)
        self.v1 = tk.IntVar()
        self.retrieval=True
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun on',
                                  command=self.retrieval_setting,
                                  variable=self.v1,
                                  value=0).pack(side=LEFT)
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun off',
                                  command=self.retrieval_setting,
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
        ebtn = tk.Button(self, text='Help', width=16, command=self.help)
        ebtn.grid(row=16, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Clear Destination', width=16, command=self.clear_destination)
        ebtn.grid(row=17, column=0, padx=10, columnspan=1, rowspan=1)

        location_config=config['location']
        floorplan_path=join(config['IO_root'],location_config['place'],location_config['building'],location_config['floor'])

        testing_image_folder=join(config['IO_root'],config['testing_image_folder'])
        
        fl_selected = filedialog.askopenfilename(initialdir=floorplan_path, title='Select Floor Plan')
        im = Image.open(fl_selected)
        draw = ImageDraw.Draw(im)
        self.imtra=im.copy()
        
        for index in self.kf.keys():
            k = self.kf[index]
            x_, y_ = k['trans']
            draw.ellipse((x_ - 2, y_ - 2, x_ + 2, y_ + 2), fill=(0, 255, 0))
        self.imf = im.copy()
        width, height = im.size
        self.plot_scale =width/3400
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
                            self.show_trajectory(action))



def main(root,hloc_config,visual_config):
    map_data=loader.load_data(visual_config)
    hloc = Hloc(root, map_data, hloc_config)
    trajectory=Trajectory(map_data)

    style = Style(theme='darkly')
    root = style.master
    Main_window(root, map_data=map_data,hloc=hloc,trajectory=trajectory)
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