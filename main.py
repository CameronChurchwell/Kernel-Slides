from manim import *
import numpy as np
from helpers import *
from custom_slide import CustomSlide

from tensors import Tensor2D

np.random.seed(8888)

class GridOfSquares(CustomSlide):

    def indexing_slide(self):
        # flat indexing
        title = self.slide_title("Flat Indexing")
        tensor = Tensor2D(4, 4, 1, content='indices')
        self.transition(Group(title, tensor))
        self.next_slide()

        flat_indexed_tensor = Tensor2D(4, 4, 1, content='flat_indices')
        self.play(ReplacementTransform(tensor, flat_indexed_tensor))
        self.next_slide()

        flat_tensor = Tensor2D(1, 16, 0.5, content='flat_indices')
        self.play(
            ReplacementTransform(a, b) for a, b in zip(flat_indexed_tensor.squares.flatten(), flat_tensor.squares.flatten())
        )
        self.next_slide()

    def coalesce_slide(self):
        vals = np.random.randint(0, 10, (6, 32))
        tensor = Tensor2D(6, 32, 0.3, content=vals)

        title = self.slide_title('GPU Memory Coalescing')

        self.transition(Group(tensor, title))
        self.next_slide()

        indices = Tensor2D(1, 32, 0.3, content=np.arange(0, 32)[None] + 32)
        indices.next_to(tensor, UP)
        self.play(FadeIn(indices))
        self.next_slide()

        self.play(indices.animate.highlight(GREEN))
        self.next_slide()

        self.play_sequence(tensor.coalesced_gather(indices))
        self.next_slide()

        disordered_indices = Tensor2D(1, 32, 0.3, content=np.random.permutation(np.arange(0, 32))[None] + 32)
        disordered_indices.move_to(indices)
        self.play(ReplacementTransform(indices, disordered_indices))
        self.next_slide()

        self.play(disordered_indices.animate.highlight(GREEN))
        self.next_slide()

        self.play_sequence(tensor.coalesced_gather(disordered_indices))
        self.next_slide()

        misaligned_indices = Tensor2D(1, 32, 0.3, content=np.arange(0, 32)[None] + 32 + 8)
        misaligned_indices.move_to(indices)
        self.play(ReplacementTransform(disordered_indices, misaligned_indices))
        self.next_slide()

        self.play(misaligned_indices.animate.highlight(YELLOW))
        self.next_slide()

        self.play_sequence(tensor.coalesced_gather(misaligned_indices))
        self.next_slide()

        random_indices = Tensor2D(1, 32, 0.3, content=np.random.randint(0, 32*6, (1, 32)))
        random_indices.move_to(misaligned_indices)
        self.play(ReplacementTransform(misaligned_indices, random_indices))
        self.next_slide()

        self.play(random_indices.animate.highlight(RED))
        self.next_slide()

        self.play_sequence(tensor.coalesced_gather(random_indices))
        self.next_slide()

    def construct(self):
        # slide number
        self.counter = 0
        slide_number = Text("0").to_corner(DL)
        self.add(slide_number)
        self.add_to_canvas(slide_number=slide_number)

        self.title_slide('Kernel Programming', 'Cameron Churchwell')

        # # self.bullet_slide(
        # #     'Title',
        # #     'first', 
        # #     'second',
        # #     'third',
        # #     'fourth',
        # #     'fifth'
        # # )

        # self.bullet_slide(
        #     'What is a Kernel?',
        #     'first', 
        #     'second',
        #     'third',
        #     'fourth',
        #     'fifth'
        # )

        self.bullet_slide(
            'Why Write a Kernel?',
            'Your code is impractically slow', 
            'Your code uses too much VRAM',
            'The built-in torch/jax operations cannot be combined to achieve what you want',
        )

        self.bullet_slide(
            'Why is Your Code Inefficient?',
            'You\'re using a python "for" or "while" loop', 
            'You\'re launching a lot of kernels',
            'You have unnecessary memory copies',
            'You have wasted computations',
            'Your memory access patterns are inefficient'
        )

        # # self.bullet_slide(
        # #     'How GPUs Work',
        # #     '', 
        # #     '',
        # #     '',
        # # )

        self.indexing_slide()

        self.coalesce_slide()