import time

import matplotlib
from pynput.keyboard import Controller, Key


def recorder():
    key = Controller()
    key.press(Key.f8)
    time.sleep(0.05)
    key.release(Key.f8)
    time.sleep(2)


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


if __name__ == '__main__':
    recorder()
