import tkinter as tk
from coupled_JJs import JJs
import threading
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class storage:
    def __init__(self):
        self.result = pd.DataFrame(columns=['I', 'V1', 'V2'])
    

class GUI:
    def __init__(self, store, JJ):
        # Initialisiere den Speicher und die JJs
        self.store = store
        self.JJ = JJ

        # Erstelle das Fenster
        self.window = tk.Tk()
        self.window.title("2 JJ Simulation")
        self.window.geometry("400x400")

        # Erstelle Eingabefelder für die Parameter
        self.Ic1_label = tk.Label(self.window, text="Ic1")
        self.Ic1_label.pack()
        self.Ic1 = tk.Entry(self.window)
        self.Ic1.pack()
        self.Ic1.insert(0, "50e-6")

        self.Ic2_label = tk.Label(self.window, text="Ic2")
        self.Ic2_label.pack()
        self.Ic2 = tk.Entry(self.window)
        self.Ic2.pack()
        self.Ic2.insert(0, "40e-6")

        self.R1_label = tk.Label(self.window, text="R1")
        self.R1_label.pack()
        self.R1 = tk.Entry(self.window)
        self.R1.pack()
        self.R1.insert(0, "19")


        self.R2_label = tk.Label(self.window, text="R2")
        self.R2_label.pack()
        self.R2 = tk.Entry(self.window)
        self.R2.pack()
        self.R2.insert(0, "12")

        self.R_label = tk.Label(self.window, text="R")
        self.R_label.pack()
        self.R = tk.Entry(self.window)
        self.R.pack()
        self.R.insert(0, "40")

        # Bis zu diesem Strom soll simuliert werden
        self.I_label = tk.Label(self.window, text="I_max")
        self.I_label.pack()
        self.I = tk.Entry(self.window)
        self.I.pack()
        self.I.insert(0, "400")

        # Erstelle button um JJs zu erstellen
        self.create_JJs_button = tk.Button(self.window, text="Create JJs", command=self.create_JJs)
        self.create_JJs_button.pack()

        self.simulate_button = tk.Button(self.window, text="Simulate", command= self.thread_simulation)
        self.simulate_button.pack()

        self.plot_button = tk.Button(self.window, text="Plot IVC", command= self.plot_IVC)
        self.plot_button.pack()

        self.window.mainloop()

    def create_JJs(self):
        Ic1 = float(self.Ic1.get())
        Ic2 = float(self.Ic2.get())
        R1 = float(self.R1.get())
        R2 = float(self.R2.get())
        R = float(self.R.get())

        self.JJ = JJs(Ic1, Ic2, R1, R2, R)
        print("JJs created")

    def thread_simulation(self):
        t = threading.Thread(target=self.simulate, args=())
        t.start()

    def simulate(self):
        I_max = float(self.I.get())
        t_span = (0, 500)
        dt = 0.05
        t_av_start = 100
        x0 = [0, 0]
        self.store.result = self.store.result.iloc[0:0]

        # Definiere die Bereiche für die Schleife
        current_sweep = list(range(int(I_max))) + list(range(int(I_max), -int(I_max), -1)) + list(range(-int(I_max), 0))

        for i in tqdm(current_sweep):
            I = i * 1e-6
            V1, V2, phi1, phi2 = JJs.solve(self.JJ, t_span, dt, t_av_start, x0, I)

            self.store.result = pd.concat([self.store.result, pd.DataFrame({'I': [I], 'V1': [V1], 'V2': [V2]})])
            x0 = [JJs.phi_adjust(phi1), JJs.phi_adjust(phi2)]
        

        
    def plot_IVC(self):
        plt.figure()
        plt.plot(self.store.result['V1'], self.store.result['I']*1e6, label='V1')
        plt.plot(self.store.result['V2'], self.store.result['I']*1e6, label='V2')
        plt.show()



if __name__ == "__main__":
    store = storage()
    jj = JJs(50e-6, 40e-6, 19, 12, 40)
    gui = GUI(store, jj)
