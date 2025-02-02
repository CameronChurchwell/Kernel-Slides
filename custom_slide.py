from manim import *
from manim_slides import Slide
from manim_slides.slide.animation import Wipe


tex_preamble = r"""
\usepackage[english]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{dsfont}
\usepackage{setspace}
\usepackage{tipa}
\usepackage{relsize}
\usepackage{textcomp}
\usepackage{mathrsfs}
\usepackage{calligra}
\usepackage{wasysym}
\usepackage{ragged2e}
\usepackage{physics}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{enumitem}
\DisableLigatures{encoding = *, family = * }
\setlist[itemize]{leftmargin=1em, topsep=0em, itemsep=0em} % First-level itemize indentation
\setlength{\textwidth}{30em}
\usepackage[none]{hyphenat}
\singlespacing
"""
tex_template = TexTemplate(preamble=tex_preamble)
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
        self.next_slide()
        indentation = []
        bullets = list(bullets)
        for i in range(0, len(bullets)):
            indentation.append(0)
            while bullets[i].startswith('\t'):
                bullets[i] = bullets[i][1:]
                indentation[i] += 1
        bullets = [
            r'\item ' + r'\begin{itemize}' * ind + bullet + r'\end{itemize}' * ind 
            for bullet, ind in zip(bullets, indentation)
        ]
        bullets = Tex(
            *bullets,
            arg_separator=r'',
            tex_environment='itemize',
            tex_template=tex_template,
            font_size=40
        )
        bullets.next_to(title, DOWN)
        bullets.align_to(title, LEFT)
        bullets.shift(DOWN * 0.25)

        for bullet, ind in zip(bullets, indentation):
            bullet.fade(0.2*ind)
            self.play(FadeIn(bullet, run_time=0.75))
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
            ApplyWave(text, amplitude=0.02, run_time=2.),
            ApplyWave(subtext, amplitude=0.02, run_time=2.),
            Wait(1),
            lag_ratio=0.2
        ),)
        self.next_slide()