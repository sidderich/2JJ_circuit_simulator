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

    def get_dynamic(self, index):
        '''
        Returnes data with dynamics.
        Takes the path of the dynamics folder and returns the dataframe of the file with given index.
        :param index: Index of the file to be loaded
        :return: np.array with the data of the file
        '''
        files = os.listdir(f'{self.path}/dynamics')
        files = sorted(files, key=lambda x: int(x.split('_')[0]))
        data = np.loadtxt(f'{self.path}/dynamics/{files[index]}', delimiter="\t")
        return data



class GUI:
    def __init__(self, store, JJ):
        # Initialisiere den Speicher und die JJs
        self.store = store
        self.JJ = JJ

        self.config = load_config('config.json')
        
        # Erstelle das Fenster
        self.window = tk.Tk()
        self.window.title("2 JJ Simulation")
        self.window.geometry(f"{self.window.winfo_screenwidth()}x{self.window.winfo_screenheight()}")

        # Initialisiere das Subwindow
        self.param_window = None

        # Erstelle eine Menüleiste
        self.menubar = self.create_menu()

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
        # self.console = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, width=70, height=20)
        # self.console.grid(row=0, column=0)
        # # Leite die Ausgabe der Konsole um
        # sys.stdout = RedirectText(self.console)
        # sys.stderr = RedirectText(self.console)
        
        # Ersttelle variable für iterative Berechnung der Spannung
        self.iterative_av = tk.BooleanVar()
        self.iterative_av.set(True)

        # Erstelle einen canvas
        self.fig = Figure(figsize=(14,10))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # Erstelle die Toolbar für den canvas
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
           
        # Erstelle entry für den Index der dynamic file
        self.index_label = tk.Label(self.button_frame, text="Index")
        self.index_label.pack()
        self.index = tk.Entry(self.button_frame)
        self.index.pack()
        self.index.insert(0, "0")

        # Erstelle button um die Dynamik zu plotten
        self.plot_dynamics_button = tk.Button(self.button_frame, text="Plot phi", command= self.plot_dynamics)
        self.plot_dynamics_button.pack()

        # Aktualisiere Fenster und passe Größe an
        self.window.update()
        self.window.geometry("")
        self.window.mainloop()

    def create_menu(self):
        menubar = tk.Menu(self.window)

        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load IVC", command=self.load_IVC)
        filemenu.add_command(label="Load config", command=self.browse_config)
        filemenu.add_command(label="Results directory", command=self.set_path_btn)
        menubar.add_cascade(label="File", menu=filemenu)

        # Simulation menu
        simmenu = tk.Menu(menubar, tearoff=0)
        simmenu.add_command(label="Parameters", command=self.param_window_btn)
        simmenu.add_command(label="Simulate", command=self.thread_simulation)
        simmenu.add_command(label="Simulate multiple", command=self.thread_simulate_multiple)
        menubar.add_cascade(label="Simulation", menu=simmenu)

        # Canvas menu
        canvasmenu = tk.Menu(menubar, tearoff=0)
        canvasmenu.add_command(label="Plot IVC", command=self.plot_IVC)
        canvasmenu.add_command(label="Plot phi", command=self.plot_dynamics)
        menubar.add_cascade(label="Plotter", menu=canvasmenu)

        # Menüleiste zum Fenster hinzufügen
        self.window.config(menu=menubar)

    def param_window_btn(self):
        # Erstelle das Subwindow nur, wenn es noch nicht existiert
        if self.param_window is None or not self.param_window.winfo_exists():
            self.param_window = tk.Toplevel(self.window)
            self.param_window.title("Parameters")
            self.param_window.geometry("300x400")
            
        # Erstelle Eingabefelder für die Parameter
        self.Ic1_label = tk.Label(self.param_window, text="Ic1")
        self.Ic1_label.pack()
        self.Ic1 = tk.Entry(self.param_window)
        self.Ic1.pack()
        self.Ic1.insert(0, self.JJ.Ic1)

        self.Ic2_label = tk.Label(self.param_window, text="Ic2")
        self.Ic2_label.pack()
        self.Ic2 = tk.Entry(self.param_window)
        self.Ic2.pack()
        self.Ic2.insert(0, self.JJ.Ic2)

        self.R1_label = tk.Label(self.param_window, text="R1")
        self.R1_label.pack()
        self.R1 = tk.Entry(self.param_window)
        self.R1.pack()
        self.R1.insert(0, self.JJ.R1)


        self.R2_label = tk.Label(self.param_window, text="R2")
        self.R2_label.pack()
        self.R2 = tk.Entry(self.param_window)
        self.R2.pack()
        self.R2.insert(0, self.JJ.R2)

        self.R_label = tk.Label(self.param_window, text="R")
        self.R_label.pack()
        self.R = tk.Entry(self.param_window)
        self.R.pack()
        self.R.insert(0, self.JJ.R)

        self.L_label = tk.Label(self.param_window, text="L")
        self.L_label.pack()
        self.L = tk.Entry(self.param_window)
        self.L.pack()
        self.L.insert(0, self.JJ.L)

        # Bis zu diesem Strom soll simuliert werden
        self.I_label = tk.Label(self.param_window, text="I_max")
        self.I_label.pack()
        self.I = tk.Entry(self.param_window)
        self.I.pack()
        try:
            self.I.insert(0, self.config["simulation_parameters"]["max_current"])
        except TypeError:
            self.I.insert(0, '0.0003')

        # Zeitschritt für die Simulation
        self.dt_label = tk.Label(self.param_window, text="Time step")
        self.dt_label.pack()
        self.dt = tk.Entry(self.param_window)
        self.dt.pack()
        try:
            self.dt.insert(0, self.config["simulation_parameters"]["time_step"])
        except TypeError:
            self.dt.insert(0, '0.05')





        # Erstelle togglebox für die iterative Berechnung der Spannung
        self.iterative_av_check = tk.Checkbutton(self.param_window, text="Iterative averaging", variable=self.iterative_av)
        self.iterative_av_check.pack()

        # Erstelle button um JJs zu erstellen
        self.create_JJs_button = tk.Button(self.param_window, text="Create JJs", command=self.create_JJs)
        self.create_JJs_button.pack()

    def create_JJs(self, dt = 0.01):
        '''
        Erstelle die JJs mit den eingegebenen Parametern
        Existiert kein Subwindow, so werden die Parameter aus self.config übernommen
        '''
        if self.param_window is not None and self.param_window.winfo_exists():
            Ic1 = float(self.Ic1.get())
            Ic2 = float(self.Ic2.get())
            R1 = float(self.R1.get())
            R2 = float(self.R2.get())
            R = float(self.R.get())
            L = float(self.L.get())
            self.config["simulation_parameters"]["max_current"] = float(self.I.get())
            dt = float(self.dt.get())
            self.config["simulation_parameters"]["time_step"] = dt

        else:
            Ic1 = self.config["junction_parameters"]["I_C1"]
            Ic2 = self.config["junction_parameters"]["I_C2"]
            R1 = self.config["junction_parameters"]["R_1"]
            R2 = self.config["junction_parameters"]["R_2"]
            R = self.config["junction_parameters"]["R"]
            L = self.config["junction_parameters"]["L"]
            dt = self.config["simulation_parameters"]["time_step"]

        self.JJ = JJs(Ic1, Ic2, R1, R2, R, L, dt = dt)
        print("JJs created")

    def browse_config(self):
        '''
        Lade einzelne Konfigurationsdatei und Erstelle die JJ
        '''
        try:
            config_path = filedialog.askopenfilename()
            self.config = load_config(config_path)
            # Erstelle die JJs
            self.create_JJs(dt = float(self.config["simulation_parameters"]["time_step"]))
        except:
            print("Error loading config file")

       
        

    def create_folder_name(self):
        Ic1 = int(self.JJ.Ic1*1e6) #Umrechnung in A von uA
        Ic2 = int(self.JJ.Ic2*1e6) #Umrechnung in A von uA
        R1 = int(self.JJ.R1)
        R2 = int(self.JJ.R2)
        R = int(self.JJ.R)
        L = int(self.JJ.L*1e12) #Umrechnung in H von pH
        return f'{self.store.path}/simu{Ic1}uA_{Ic2}uA_{L}pH_{R1}ohm_{R2}ohm_{R}ohm'
        
    def set_path_btn(self):
        # browse folder tkinter
        directory = filedialog.askdirectory()
        self.store.set_path(directory)
        print(f"Results directory set to {directory}")

    def thread_simulate_multiple(self):
        config_files = filedialog.askopenfilenames()
        t = threading.Thread(target=self.simulate_multiple, args=(config_files,False))
        t.start()

    def simulate_multiple(self, config_files, save_dynamics = False):
        counter = 1
        for config_file in config_files:
            print(f"Simulation {counter}/{len(config_files)}")
            self.config = load_config(config_file)

            self.JJ = JJs(
                Ic1 = self.config["junction_parameters"]["I_C1"],
                Ic2 = self.config["junction_parameters"]["I_C2"],
                R1 = self.config["junction_parameters"]["R_1"],
                R2 = self.config["junction_parameters"]["R_2"],
                R = self.config["junction_parameters"]["R"],
                L = self.config["junction_parameters"]["L"],
                dt = self.config["simulation_parameters"]["time_step"]
            )
            self.simulate(save_dynamics = save_dynamics)
            counter += 1
            # Speichere die Konfiguration
            with open(f"{self.create_folder_name()}/config.json", 'w') as f:
                json.dump(self.config, f, indent=4)


    def thread_simulation(self):
        t = threading.Thread(target=self.simulate, args=('ind',))
        t.start()

    def simulate(self, model = 'ind', save_dynamics = True):
        '''
        Simliert die IVC der JJs
        :param model: 'ind' für induktivitätsmodell, 'else' für resistives Modell
        :param save_dynamics: True, wenn die Dynamik gespeichert werden soll (Default: True)
        '''
        if self.store.path is None:
            print("Please set a path first")
            return
        if self.config is not None:
            I_max = self.config["simulation_parameters"]["max_current"]*1e6
        else:
            I_max = float(self.I.get())
        t_span = (0, 500)
        t_av_start = 100
        if model == 'ind':
            x0 = [0,0,0]
        else:
            x0 = [0,0]

        # Lösche die Ergebnisse und die Dynamik
        self.store.result = self.store.result.iloc[0:0]
        self.store.dynamics = self.store.dynamics.iloc[0:0]
        
        # Erstelle die Ordner für die Simulation, falls sie noch nicht existieren
        # Erstelle zunächst Ordnernamen
        #results_folder = f'{self.store.path}/results_{int(self.JJ.Ic1*1e6)}uA_{int(self.JJ.Ic2*1e6)}uA_{int(self.JJ.R1)}ohm_{int(self.JJ.R2)}ohm_{int(self.JJ.R)}ohm'
        simu_path = self.create_folder_name()
        dynamics_folder = f'{simu_path}/dynamics'
                
        if not os.path.exists(simu_path):
            os.makedirs(simu_path)

        if not os.path.exists(dynamics_folder) and save_dynamics:
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
                    dt = self.JJ.dt,
                    iterative_av = self.iterative_av.get())

            else:
                V1, V2, phi1, phi2, t = JJs.solve(
                    self.JJ, 
                    t_span, 
                    t_av_start, 
                    x0, 
                    I,
                    dt = self.JJ.dt, 
                    iterative_av = self.iterative_av.get())

            # Speichere die IVC im Dataframe
            self.store.result = pd.concat([self.store.result, pd.DataFrame({'I': [I], 'V1': [V1], 'V2': [V2]})], ignore_index=True)
            # Speichere die Dynamik als .dat Datei
            data = np.column_stack((t, phi1, phi2))
            if save_dynamics:
                np.savetxt(f"{dynamics_folder}/{counter}_{i}A.dat", data, delimiter="\t", header="t\tphi1\tphi2")
            # Lege die neuen Startbedingungen fest
            if model == 'ind':
                x0 = [JJs.phi_adjust(phi1[-1]), JJs.phi_adjust(phi2[-1]), ir[-1]]
            else:
                x0 = [JJs.phi_adjust(phi1[-1]), JJs.phi_adjust(phi2[-1])]
            counter += 1
        # Speichere die IVC als .csv Datei
        self.store.result.to_csv(f"{dynamics_folder.split('/dynamics')[0]}/IVC.csv", index=False)


    def load_IVC(self):
        '''
        Lade die IVC.csv Datei
        '''
        try:
            self.store.path = filedialog.askopenfilename()
        except:
            print("Error loading IVC file")
        self.store.result = pd.read_csv(f'{self.store.path}', delimiter=',')
        self.plot_IVC()


    def plot_IVC(self):
        '''
        Stelle die IVC der JJs dar
        '''
        self.ax.clear()
        
        self.ax.plot(self.store.result['V1'], self.store.result['I']*1e6, label='V1')
        self.ax.plot(self.store.result['V2'], self.store.result['I']*1e6, label='V2')
        self.canvas.draw()

    def load_dynamics(self):
        '''
        Lade die Dynamik Datei
        '''
        try:
            self.store.path = filedialog.askopenfilename()
        except:
            print("Error loading dynamics file")
        self.store.dynamics = store.get_dynamic(int(self.index.get()))

    def plot_dynamics(self):
        '''
        Stelle die Dynamik der JJs dar
        '''
        # Lade die Dynamik Datei
        index = int(self.index.get())

        # Erstelle Liste mit allen Dateien
        folder_name = self.create_folder_name()
        files = os.listdir(f'{folder_name}/dynamics')
        # Sortiere die Liste Files, damit die Dateien in der richtigen Reihenfolge geplottet werden
        files = sorted(files, key=lambda x: int(x.split('_')[0]))
        # Lade die Datei
        data = np.loadtxt(f'{folder_name}/dynamics/{files[index]}', delimiter="\t")
        self.ax.clear()
        self.ax.plot(data[:,0], np.sin(data[:,1])+1, label='phi1')
        self.ax.plot(data[:,0], np.sin(data[:,2])-1, label='phi2')
        self.ax.set_title(files[index].split('_')[1])
        self.canvas.draw()



if __name__ == "__main__":
    store = storage()
    jj = JJs(15e-6, 11e-6, 11, 12, 40, 24e-12)
    gui = GUI(store, jj)
