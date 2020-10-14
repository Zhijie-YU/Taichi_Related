import taichi as ti
import numpy as np

ti.init()

N = 320

img = ti.Vector(3, dt=ti.f32, shape=(N, N))
canvas = ti.Vector(3, dt=ti.f32, shape=(N * 2, N))
F = ti.Matrix(2, 2, dt=ti.f32, shape=())
cursor_rest = ti.Vector(2, dt=ti.f32, shape=())
cursor_deformed = ti.Vector(2, dt=ti.f32, shape=())

img.from_numpy(ti.imread('D:\\My_Learning\\taichi\\demos-lec3\\bob.png')[:, :, :3].astype(np.float32) / 255)
print(img.shape)

@ti.kernel
def paint():
    for i, j in canvas:
        if i < N:
            canvas[i, j] = img[i, j]
            #pass
        else:
            x_deformed = ti.Vector([(i - N) / N - 0.5, j / N - 0.5])
            Finv = F[None].inverse()
            x_rest = Finv @ x_deformed
            p = min(N - 1, max(0, int((x_rest[0] + 0.5) * N)))
            q = min(N - 1, max(0, int((x_rest[1] + 0.5) * N)))
            canvas[i, j] = img[p, q]

F[None] = [[2, 0], [0.5, 1]]

gui = ti.GUI('Deformation', (N * 2, N))

while True:
    paint()
    gui.set_image(canvas.to_numpy())
    gui.show()
