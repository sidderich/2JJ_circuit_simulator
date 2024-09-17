import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import os
import sys
import json

import threading
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from coupled_JJs import JJs


def load_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

class RedirectText(object):
    def __init__(self, widget):
        self.widget = widget
        self.widget.config(state=tk.NORMAL)

    def write(self, string):
        self.widget.insert(tk.END, string)
        self.widget.see(tk.END)

    def flush(self):
        pass


class storage:
    def __init__(self):
        self.result = pd.DataFrame(columns=['I', 'V1', 'V2'])
        self.dynamics = pd.DataFrame(columns=['t' ,'phi1', 'phi2'])
        self.path = None
    
    def set_path(self, path):
        self.path = path

class GUI:
    def __init__(self, store, JJ):
        # Initialisiere den Speicher und die JJs
        self.store = store
        self.JJ = JJ


        
        # Erstelle das Fenster
        self.window = tk.Tk()
        self.window.title("2 JJ Simulation")
        self.window.geometry(f"{self.window.winfo_screenwidth()}x{self.window.winfo_screenheight()}")

        # Erstelle Frame für die Eingabefelder
        self.param_frame = tk.Frame(self.window)
        self.param_frame.grid(row=0, column=2)
        # Erstelle Frame für die Buttons
        self.button_frame = tk.Frame(self.window)
        self.button_frame.grid(row=0, column=3)
        # Erstelle Frame für die Plots
        self.plot_frame = tk.Frame(self.window)
        self.plot_frame.grid(row=0, column=0, columnspan=2, rowspan=2)
        # Erstelle Frame für die Konsole
        self.console_frame = tk.Frame(self.window)
        self.console_frame.grid(row=1, column=2, columnspan=2, padx=10, pady=10)

        # Erstelle die Konsole
        self.console = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, width=50, height=10)
        self.console.grid(row=0, column=0)
        # Leite die Ausgabe der Konsole um
        # sys.stdout = RedirectText(self.console)
        # sys.stderr = RedirectText(self.console)

        # Erstelle einen canvas
        self.fig = Figure(figsize=(12,8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # Erstelle die Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        # Erstelle Eingabefelder für die Parameter
        self.Ic1_label = tk.Label(self.param_frame, text="Ic1")
        self.Ic1_label.pack()
        self.Ic1 = tk.Entry(self.param_frame)
        self.Ic1.pack()
        self.Ic1.insert(0, JJ.Ic1)

        self.Ic2_label = tk.Label(self.param_frame, text="Ic2")
        self.Ic2_label.pack()
        self.Ic2 = tk.Entry(self.param_frame)
        self.Ic2.pack()
        self.Ic2.insert(0, JJ.Ic2)

        self.R1_label = tk.Label(self.param_frame, text="R1")
        self.R1_label.pack()
        self.R1 = tk.Entry(self.param_frame)
        self.R1.pack()
        self.R1.insert(0, JJ.R1)


        self.R2_label = tk.Label(self.param_frame, text="R2")
        self.R2_label.pack()
        self.R2 = tk.Entry(self.param_frame)
        self.R2.pack()
        self.R2.insert(0, JJ.R2)

        self.R_label = tk.Label(self.param_frame, text="R")
        self.R_label.pack()
        self.R = tk.Entry(self.param_frame)
        self.R.pack()
        self.R.insert(0, JJ.R)

        self.L_label = tk.Label(self.param_frame, text="L")
        self.L_label.pack()
        self.L = tk.Entry(self.param_frame)
        self.L.pack()
        self.L.insert(0, JJ.L)

        # Bis zu diesem Strom soll simuliert werden
        self.I_label = tk.Label(self.param_frame, text="I_max")
        self.I_label.pack()
        self.I = tk.Entry(self.param_frame)
        self.I.pack()
        self.I.insert(0, "200")

        # Erstelle button um JJs zu erstellen
        self.create_JJs_button = tk.Button(self.button_frame, text="Create JJs", command=self.create_JJs)
        self.create_JJs_button.pack()
        # Erstelle button um zu simulieren
        self.simulate_button = tk.Button(self.button_frame, text="Simulate", command= self.thread_simulation)
        self.simulate_button.pack()

        # Erstelle button um den Pfad zu setzen, wo die Ergebnisse gespeichert werden sollen
        self.set_path_button = tk.Button(self.button_frame, text="Set path", command= self.set_path_btn)
        self.set_path_button.pack()

        self.plot_button = tk.Button(self.button_frame, text="Plot IVC", command= self.plot_IVC)
        self.plot_button.pack()

        # Erstelle togglebox für die iterative Berechnung der Spannung
        self.iterative_av = tk.BooleanVar()
        self.iterative_av.set(True)
        self.iterative_av_check = tk.Checkbutton(self.button_frame, text="Iterative averaging", variable=self.iterative_av)
        self.iterative_av_check.pack()

        # Erstelle entry für den Index der dynamic file
        self.index_label = tk.Label(self.param_frame, text="Index")
        self.index_label.pack()
        self.index = tk.Entry(self.param_frame)
        self.index.pack()
        self.index.insert(0, "0")

        # Erstelle button um die Dynamik zu plotten
        self.plot_dynamics_button = tk.Button(self.button_frame, text="Plot phi", command= self.plot_dynamics)
        self.plot_dynamics_button.pack()

        # Aktualisiere Fenster und passe Größe an
        self.window.update()
        self.window.geometry("")


        self.window.mainloop()


    def set_path_btn(self):
        # browse folder tkinter
        directory = filedialog.askdirectory()
        self.store.set_path(directory)

    def create_JJs(self):
        Ic1 = float(self.Ic1.get())
        Ic2 = float(self.Ic2.get())
        R1 = float(self.R1.get())
        R2 = float(self.R2.get())
        R = float(self.R.get())
        L = float(self.L.get())

        self.JJ = JJs(Ic1, Ic2, R1, R2, R, L)
        print("JJs created")

    def thread_simulation(self):
        t = threading.Thread(target=self.simulate, args=('ind',))
        t.start()

    def simulate(self, model = 'ind'):
        if self.store.path is None:
            print("Please set a path first")
            return
        I_max = float(self.I.get())
        t_span = (0, 500)
        t_av_start = 100
        if model == 'ind':
            x0 = [0,0,0]
        else:
            x0 = [0,0]
        self.store.result = self.store.result.iloc[0:0]
        self.store.dynamics = self.store.dynamics.iloc[0:0]
        
        # Erstelle die Ordner für die Simulation, falls sie noch nicht existieren
        # Erstelle zunächst Ordnernamen
        #results_folder = f'{self.store.path}/results_{int(self.JJ.Ic1*1e6)}uA_{int(self.JJ.Ic2*1e6)}uA_{int(self.JJ.R1)}ohm_{int(self.JJ.R2)}ohm_{int(self.JJ.R)}ohm'
        simu_path = f'{self.store.path}/simu{int(self.JJ.Ic1*1e6)}uA_{int(self.JJ.Ic2*1e6)}uA_{int(self.JJ.R1)}ohm_{int(self.JJ.R2)}ohm_{int(self.JJ.R)}ohm'
        dynamics_folder = f'{simu_path}/dynamics'
                
        if not os.path.exists(simu_path):
            os.makedirs(simu_path)

        if not os.path.exists(dynamics_folder):
            os.makedirs(dynamics_folder)

        # Definiere die Bereiche für die Schleife
        current_sweep = list(range(int(I_max))) + list(range(int(I_max), 0, -1))
        counter = 0
        for i in tqdm(current_sweep, file=sys.stdout):
            I = i * 1e-6
            if model == 'ind':
                V1, V2, phi1, phi2, ir, t = JJs.solve(
                    self.JJ, 
                    t_span, 
                    t_av_start, 
                    x0, 
                    I, 
                    iterative_av = self.iterative_av.get())

            else:
                V1, V2, phi1, phi2, t = JJs.solve(
                    self.JJ, 
                    t_span, 
                    t_av_start, 
                    x0, 
                    I, 
                    iterative_av = self.iterative_av.get())

            # Speichere die IVC im Dataframe
            self.store.result = pd.concat([self.store.result, pd.DataFrame({'I': [I], 'V1': [V1], 'V2': [V2]})], ignore_index=True)
            # Speichere die Dynamik als .dat Datei
            data = np.column_stack((t, phi1, phi2))
            np.savetxt(f"{dynamics_folder}/{counter}_{i}A.dat", data, delimiter="\t", header="t\tphi1\tphi2")
            # Lege die neuen Startbedingungen fest
            if model == 'ind':
                x0 = [JJs.phi_adjust(phi1[-1]), JJs.phi_adjust(phi2[-1]), ir[-1]]
            else:
                x0 = [JJs.phi_adjust(phi1[-1]), JJs.phi_adjust(phi2[-1]),]
            counter += 1
        # Speichere die IVC als .csv Datei
        self.store.result.to_csv(f"{dynamics_folder.split('/dynamics')[0]}/IVC.csv", index=False)
        

        
    def plot_IVC(self):
        self.ax.clear()
        
        self.ax.plot(self.store.result['V1'], self.store.result['I']*1e6, label='V1')
        self.ax.plot(self.store.result['V2'], self.store.result['I']*1e6, label='V2')
        self.canvas.draw()

    def plot_dynamics(self):
        # Lade die Dynamik Datei
        index = int(self.index.get())

        # Erstelle Liste mit allen Dateien
        files = os.listdir(f'{self.store.path}/simu{int(self.JJ.Ic1*1e6)}uA_{int(self.JJ.Ic2*1e6)}uA_{int(self.JJ.R1)}ohm_{int(self.JJ.R2)}ohm_{int(self.JJ.R)}ohm/dynamics')
        # Sortiere die Liste Files, damit die Dateien in der richtigen Reihenfolge geplottet werden
        files = sorted(files, key=lambda x: int(x.split('_')[0]))
        # Lade die Datei
        data = np.loadtxt(f'{self.store.path}/simu{int(self.JJ.Ic1*1e6)}uA_{int(self.JJ.Ic2*1e6)}uA_{int(self.JJ.R1)}ohm_{int(self.JJ.R2)}ohm_{int(self.JJ.R)}ohm/dynamics/{files[index]}', delimiter="\t")
        self.ax.clear()
        self.ax.plot(data[:,0], np.sin(data[:,1])+1, label='phi1')
        self.ax.plot(data[:,0], np.sin(data[:,2])-1, label='phi2')
        self.ax.set_title(files[index].split('_')[1])
        self.canvas.draw()



if __name__ == "__main__":
    store = storage()
    jj = JJs(15e-6, 11e-6, 11, 12, 40, 24e-12)
    gui = GUI(store, jj)
