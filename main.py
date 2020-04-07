from tkinter import *
import random
import pandas as pd
import numpy as np
import os
import matplotlib as plt
# from multiprocessing import Pool
# from joblib.numpy_pickle_utils import xrange
import sys

class Region():    
    def __init__(self, master = None):
        self.time = 1
        self.df_pop = pd.DataFrame()
        self.df_inf = pd.DataFrame([[0,0,0,0]], columns = ['time', 'population', 'infected', 'recovered'])
        self.sum_infected = 0
        self.sum_recovered = 0
        self.master = master
        self.p = []
        self.h = 300
        self.w = self.h / 3 * 4
        self.dim = []
        self.dim.append((self.h, self.w))
        self.dim.append((self.h / 2, self.w))
        self.dim.append((self.h, self.w))
        self.margin = 0.05
        self.origin = (self.w * self.margin, self.h * self.margin)
        self.end = (self.w * (1 - self.margin), self.h * (1 - self.margin))
        self.origin_info = (self.w * self.margin, self.h)
        self.end_info = (self.end[0] ,(self.origin_info[1] + self.dim[1][0]) * (1 - self.margin))
        self.origin_graph = (self.w, self.h * self.margin)
        self.end_graph = (self.origin_graph[0] + self.w * (1 - self.margin), self.end[1])
        self.end_info = (self.end[0] ,(self.origin_info[1] + self.dim[1][0]) * (1 - self.margin))
        self.canvas = Canvas(master, height = self.dim[0][0] + self.dim[1][0], width = self.dim[0][1] + self.dim[2][1])
        self.bg = self.canvas.create_rectangle(0,0, self.dim[0][1] + self.dim[2][1], self.dim[0][0] + self.dim[1][0], fill="black")
        self.area = self.canvas.create_rectangle(self.origin[0], self.origin[1], self.end[0], self.end[1], fill="#303030", outline = 'white')
        self.infos = self.canvas.create_rectangle(self.origin_info[0], self.origin_info[1], self.end_info[0], self.end_info[1], fill="#404040", outline = 'white')
        self.graph = self.canvas.create_rectangle(self.origin_graph[0], self.origin_graph[1], self.end_graph[0], self.end_graph[1], fill="#404040", outline = 'white')
        text_start = self.origin_info[1] * (1 + self.margin)
        text_height = self.dim[1][0] * (1 - self.margin * 2) / 9
        self.tx_inf = self.canvas.create_text(self.origin_info[0] * (1 + self.margin * 4),
                                              text_start + 0 * text_height,
                                              fill="white", font="Montserrat 12", anchor = NW,
                                              text='Number of Infected people: ' + '{:.0f}'.format(self.sum_infected))
        self.tx_rec = self.canvas.create_text(self.origin_info[0] * (1 + self.margin * 4),
                                              text_start + 1 * text_height,
                                              fill="white", font="Montserrat 12", anchor = NW,
                                              text='Number of Recoved people: ' + '{:.0f}'.format(self.sum_recovered))
        self.tx_perc = self.canvas.create_text(self.origin_info[0] * (1 + self.margin * 4),
                                              text_start + 2 * text_height,
                                              fill="white", font="Montserrat 12", anchor = NW,
                                              text='Percentage of population: ' + '{:.2f}'.format(self.sum_infected/max(len(self.p),1) * 100))
        self.tx_R = self.canvas.create_text(self.origin_info[0] * (1 + self.margin * 4),
                                              text_start + 3 * text_height,
                                              fill="white", font="Montserrat 12", anchor = NW,
                                              text='R = ' + '{:.2f}'.format(self.sum_infected ** (1/self.time)))
        self.tx_elap = self.canvas.create_text(self.origin_info[0] * (1 + self.margin * 4),
                                              text_start + 4 * text_height,
                                              fill="white", font="Montserrat 12", anchor = NW,
                                              text='Time elapsed: ' + str(self.time))
        self.tx_dens = self.canvas.create_text(self.origin_info[0] * (1 + self.margin * 4),
                                              text_start + 5 * text_height,
                                              fill="white", font="Montserrat 12", anchor = NW,
                                              text='Populational density: ' + '{:0.2f}'.format(float(len(self.p)) / (self.h * self.w) * 1000))
        self.chart = Graph(self.master, self)
        self.infected_status()
        self.canvas.pack()
        
    def add_population(self, n = 1):
        for i in range(0, n):
            self.p.append(self.new_person())
             
    def new_person(self, i = 1):
        return Person(self.master, self)

    def infected_status(self):
        self.sum_infected = 0
        if len(self.p) > 0:
            self.df_pop = pd.DataFrame.from_records([i.to_dict() for i in self.p]).set_index('id', drop = True)
            self.sum_infected = self.df_pop['flag_inf'].sum()
            self.sum_recovered = self.df_pop['flag_rec'].sum()
#             print(str(len(self.df_pop)) + ',' + str(self.df_pop['flag_inf'].sum()))
        self.canvas.itemconfig(self.tx_inf, text = 'Number of Infected people: ' + '{:.0f}'.format(self.sum_infected))
        self.canvas.itemconfig(self.tx_rec, text = 'Number of Recovered people: ' + '{:.0f}'.format(self.sum_recovered))
        self.canvas.itemconfig(self.tx_perc, text = 'Percentage of population: ' + '{:.2f}'.format(self.sum_infected/max(len(self.p),1) * 100))
        if self.sum_infected < len(self.p): self.canvas.itemconfig(self.tx_R, text = 'R = ' + '{:.2f}'.format(self.sum_infected ** (1/self.time)))
        self.canvas.itemconfig(self.tx_elap, text='Time elapsed: ' + '{:.2f}'.format(self.time))
        self.canvas.itemconfig(self.tx_dens, text='Populational density: ' + '{:0.2f}'.format(float(len(self.p)) / (self.h * self.w) * 1000))
        if self.sum_recovered < len(self.p):
            self.df_inf = self.df_inf.append({'time':self.time,'population':len(self.p), 'infected':self.sum_infected, 'recovered':self.sum_recovered}, ignore_index = True)
            self.chart.update_chart()
            self.time = self.time + 0.1
#         if self.time % 1: self.add_population(2)
        self.master.after(100, self.infected_status)
            
class Person(object):
    def __init__(self, master, region):
        self.master = master
        zmeanHDT = 13
        zsdHDT = 12.7
        zmedianHDT = 9.1
        muHDT = np.log(zmedianHDT)
        sigmaHDT = np.sqrt(2*(np.log(zmeanHDT/zmedianHDT)))
        self.ttr = random.lognormvariate(muHDT, np.sqrt(2*(np.log(zmeanHDT/zmedianHDT))))
#         print(self.ttr)
        self.time_inf = np.nan
        self.dx = random.choice([-1,1])
        self.dy = random.choice([-1,1])
        self.r = region
        self.id = len(r.p)
        self.x = random.random() * (r.end[0] - r.origin[0]) + r.origin[0]
        self.y = random.random() * (r.end[1] - r.origin[1]) + r.origin[1]
        self.diam = 12
        self.p_inf = 0.01
        self.p_cont = 0.05
        self.flag_inf = 0
        self.flag_rec = 0
        self.inf_radius = 50
        self.canvas = region.canvas
        self.o = self.canvas.create_oval(self.x, self.y, self.x + self.diam/2, self.y + self.diam/2, fill = 'white', outline = 'white')
        self.repos()
        self.infection()
        
    def repos(self):
        if (self.x + self.dx + self.diam/2>= self.r.end[0]):
            self.dx = -1
        if (self.x + self.dx <= self.r.origin[0]):
            self.dx = 1
        if (self.y + self.dy + self.diam/2>= self.r.end[1]):
            self.dy = -1
        if (self.y + self.dy <= self.r.origin[1]):
            self.dy = 1
        self.x = self.x + self.dx
        self.y = self.y + self.dy
        self.canvas.move(self.o, self.dx, self.dy)
        self.master.after(5, self.repos)
        
    def infection(self):
#         clear()
#         self.flag_inf = (random.random() < (self.p_inf / 200)) * 1
        if (self.flag_inf == 1) and (self.flag_rec == 0):
            self.time_inf
            temp = self.r.df_pop.copy()
            temp = temp[temp['flag_inf'] == 0]
            temp['dist'] = ((temp['x'] - self.x) ** 2 + (temp['y'] - self.y) ** 2) ** (1/2)
            temp = temp[temp['dist'] > 0]
            temp = temp[temp['dist'] < self.inf_radius]
            for i in temp.iterrows():
                self.r.p[i[0]].infected()
        self.recovered()
        self.master.after(1000, self.infection)
        
    def recovered(self):
        if (self.flag_rec == 0) and (self.flag_inf == 1) and ((self.r.time - self.time_inf) >= self.ttr):
            self.canvas.itemconfig(self.o, fill="green", outline = 'green')
            self.flag_rec = 1

    def infected(self, p = None):
        if p == None: p = self.p_cont
        if (self.flag_rec == 0) and (self.flag_inf == 0):
            self.flag_inf = (random.random() <= (p)) * 1
            if self.flag_inf == 1:
                self.time_inf = self.r.time
                self.canvas.itemconfig(self.o, fill="red", outline = 'red')
            

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'flag_inf': self.flag_inf,
            'flag_rec': self.flag_rec,
            'id': self.id
        }
        
class Graph(object):
    def __init__(self, master, region):
        self.master = master
        self.r = region
        self.df_inf = self.r.df_inf
        self.bar_pop = []
        self.bar_inf = []
        self.bar_rec = []
        self.update_chart()
        
    def update_chart(self):
        self.df_inf = self.r.df_inf
        n_rows = len(self.df_inf)
        total_width = self.r.dim[2][1] * (1 - 2 * self.r.margin)
        unit_width = total_width / n_rows 
        total_height = self.r.dim[2][0] * (1 - 4 * self.r.margin)
        top_margin = self.r.dim[2][0] * 2 * self.r.margin
        left_margin = self.r.origin_graph[0] + self.r.dim[2][1] * self.r.margin
        self.df_inf['pop_h'] = ((self.df_inf['population'] - self.df_inf['infected']) / self.df_inf['population'] * total_height).fillna(0)
        self.df_inf['inf_h'] = ((self.df_inf['infected'] - self.df_inf['recovered']) / self.df_inf['population'] * total_height).fillna(0)
        self.df_inf['rec_h'] = ((self.df_inf['recovered']) / self.df_inf['population'] * total_height).fillna(0)
        pop = self.r.canvas.create_rectangle(0, 0, 0, 0, fill = '', outline='')
        self.bar_pop.append(pop)
        inf = self.r.canvas.create_rectangle(0, 0, 0, 0, fill = '', outline='')
        self.bar_inf.append(inf)
        rec = self.r.canvas.create_rectangle(0, 0, 0, 0, fill = '', outline='')
        self.bar_rec.append(rec)
        for i in range(0, n_rows):
            self.r.canvas.coords(self.bar_pop[i], left_margin + unit_width * (i - 1), top_margin, left_margin + unit_width * i, top_margin + self.df_inf['pop_h'].loc[i])
            self.r.canvas.itemconfig(self.bar_pop[i], fill = 'white')
            self.r.canvas.coords(self.bar_inf[i], left_margin + unit_width * (i - 1), top_margin + self.df_inf['pop_h'].loc[i], left_margin + unit_width * i, top_margin + self.df_inf['pop_h'].loc[i] + self.df_inf['inf_h'].loc[i])
            self.r.canvas.itemconfig(self.bar_inf[i], fill = 'red')
            self.r.canvas.coords(self.bar_rec[i], left_margin + unit_width * (i - 1), top_margin + self.df_inf['pop_h'].loc[i] + self.df_inf['inf_h'].loc[i], left_margin + unit_width * i, top_margin + self.df_inf['pop_h'].loc[i] + self.df_inf['inf_h'].loc[i] + self.df_inf['rec_h'].loc[i])
            self.r.canvas.itemconfig(self.bar_rec[i], fill = 'green')
#         print(self.df_inf)
#         self.r.canvas.itemconfig(self.r.tx_df, text=self.df_inf)
#         self.master.after(1000, self.update_chart())

master = Tk()
r = Region(master)
r.add_population(150)
r.p[0].infected(1)
master.mainloop()

