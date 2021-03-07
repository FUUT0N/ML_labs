from PIL import Image, ImageDraw
from random import randrange as rand
from pathlib import Path


class Figure:

    def __init__(self):
        pass

    def update_xy(xy: list, iter):
        if xy[0] != 512:
            xy[0] += iter
        else:
            xy[0] = iter
            xy[1] += iter \
                if xy[1] != 512 \
                else 0

    def draw_rectangle(draw, xy):
        draw.rectangle((xy[0] - rand(32, 128), xy[1] - rand(32, 128), xy[0], xy[1]),
                       fill=(rand(75, 255), rand(75, 255), rand(75, 255)))

    def draw_ellipse(draw, xy):
        step = rand(32, 128)
        draw.ellipse((xy[0] - step, xy[1] - step, xy[0], xy[1]),
                     fill=(rand(75, 255), rand(75, 255), rand(75, 255)))

    def draw_polygon(draw, xy):
        draw.regular_polygon(((xy[0] - 50), (xy[1] - 50), rand(20, 64)), rand(3, 7), rotation=rand(0, 180),
                             fill=(rand(75, 255), rand(75, 255), rand(75, 255)))

    def generate(self, N):
        iter = 128
        centre = [iter, iter]
        figure_pool = [Figure.draw_rectangle, Figure.draw_ellipse, Figure.draw_polygon]
        for i in range(N):
            image = Image.new('RGB', (600, 600), (0, 0, 0))
            draw = ImageDraw.Draw(image)
            for fig in range(16):
                figure_pool[rand(0, 3)](draw, centre)
                Figure.update_xy(centre, iter)
            Path("images").mkdir(parents=True, exist_ok=True)
            image.save(f"images/test{i}.jpg")
            centre = [iter, iter]
