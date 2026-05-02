import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from src.physics.model import KapitzaModel
from src.utils.constants import DEFAULT_PARAMS

def animate_film():
    params = DEFAULT_PARAMS
    model = KapitzaModel(
        rho=params["rho"],
        mu=params["mu"],
        sigma=params["sigma"],
        h0=params["h0"],
        L=params["L"],
        Nx=params["Nx"],
        delta=params["delta"]
    )
    
    dt = params["dt"]
    steps_per_frame = 100  # Speed up animation by doing multiple steps per frame
    
    fig, ax = plt.subplots()
    line, = ax.plot(model.x, model.get_h(), color='blue')
    ax.set_xlim(0, params["L"])
    ax.set_ylim(params["h0"] * 0.5, params["h0"] * 1.5)
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Thickness h [m]")
    ax.set_title("Kapitza Waves Simulation")
    ax.grid(True)

    def update(frame):
        for _ in range(steps_per_frame):
            model.step(dt)
        line.set_ydata(model.get_h())
        return line,

    ani = FuncAnimation(fig, update, frames=800, interval=20, blit=True)
    writer = PillowWriter(fps=25)
    ani.save("animation.gif", writer=writer)

if __name__ == "__main__":
    animate_film()
