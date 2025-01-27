from manim import *

def title_slide(self, title, author):
    self.play(Wait()); self.next_slide(auto_next=True)

    # Title slide
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
        Wait(0.5),
        lag_ratio=0.2
    ),)
    return Group(text, subtext)