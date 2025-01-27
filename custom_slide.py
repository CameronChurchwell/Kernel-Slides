from manim import *
from manim_slides import Slide
from manim_slides.slide.animation import Wipe
import textwrap

class CustomSlide(Slide):
    def update_canvas(self):
        self.counter += 1
        old_slide_number = self.canvas["slide_number"]
        new_slide_number = Text(f"{self.counter}").move_to(old_slide_number)
        return Transform(old_slide_number, new_slide_number, run_time=0.25)

    def transition(self, future):
        current = self.mobjects_without_canvas
        anim = Wipe(current, future, run_time=0.5)
        anim = AnimationGroup(anim, self.update_canvas())
        self.play(anim)

    def play_sequence(self, iterator):
        for anim in iter(iterator):
            self.play(anim)

    def slide_title(self, title):
        return Text(title, font_size=70, color=ORANGE).to_corner(UL)

    def bullet_slide(self, title, *bullets):
        title = self.slide_title(title)
        self.transition(title)
        group = Group()
        group.add(title)
        self.next_slide()
        previous = title
        for bullet in bullets:
            bullet = textwrap.fill(bullet, 50)
            text = Text(bullet, font_size=40)
            text.next_to(previous, DOWN)
            text.align_to(previous, LEFT)
            text.shift((1 if previous is title else 0, -0.3, 0))
            bullet_point = Dot()
            bullet_point.next_to(text, LEFT)
            bullet_point.align_to(text, UP)
            bullet_point.shift((0, -0.1, 0))
            self.play(FadeIn(text), Write(bullet_point, rate_func=rate_functions.ease_out_sine))
            previous = text
            self.next_slide()
            group.add(bullet_point)
            group.add(text)
        self.next_slide()

    def title_slide(self, title, author):
        self.play(Wait()); self.next_slide(auto_next=True)
        text = Text(title, font_size=86, color=ORANGE)
        subtext = Text(author, font_size=40)
        subtext.next_to(text, DOWN)
        self.play(Succession(
            Write(text),
            Write(subtext),
        ),)
        self.next_slide(loop=True)
        self.play(AnimationGroup(
            ApplyWave(text, amplitude=0.02, run_time=1.33),
            ApplyWave(subtext, amplitude=0.02, run_time=1.33),
            Wait(1),
            lag_ratio=0.2
        ),)
        self.next_slide()