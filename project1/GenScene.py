# Manim code generated with OpenAI GPT
# Command to generate animation: manim -pql GenScene.py GenScene 

from manim import *
from math import *

class GenScene(ThreeDScene):
    def construct(self):

        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)

        sun = Sphere(radius=0.5).set_color(ORANGE).move_to(ORIGIN)
        earth = Sphere(radius=0.2).set_color(GREEN).move_to(RIGHT * 3)
        mars = Sphere(radius=0.15).set_color(PURPLE).move_to(RIGHT * 4)

        earth_orbit = Circle(radius=3, color=WHITE, stroke_width=1).rotate(PI/2, axis=RIGHT)
        mars_orbit = Circle(radius=4, color=WHITE, stroke_width=1).rotate(PI/2, axis=RIGHT)

        self.play(Create(sun), Create(earth_orbit), Create(mars_orbit))
        self.play(Create(earth), Create(mars))

        earth_path = Circle(radius=3).rotate(PI/2, axis=RIGHT)
        mars_path = Circle(radius=4).rotate(PI/2, axis=RIGHT)

        self.play(MoveAlongPath(earth, earth_path, rate_func=linear, run_time=8),
                  MoveAlongPath(mars, mars_path, rate_func=linear, run_time=12))

        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(10)
        self.stop_ambient_camera_rotation()