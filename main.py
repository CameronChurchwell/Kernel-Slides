from manim import *
import numpy as np
from helpers import *
from custom_slide import CustomSlide
from custom_animations import TransformMatchingTexInOrder
from copy import deepcopy

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

    def broadcasting_slide(self):
        title = self.slide_title('Broadcasting')
        self.transition(title)
        self.next_slide()

        plus = MathTex('+')
        a = Tensor2D(1, 3, 1, content=np.array([1, 2, 3])[None])
        b = Tensor2D(2, 1, 1, content=np.array([7, 8])[:, None])
        a.next_to(plus, LEFT)
        b.next_to(plus, RIGHT)
        self.play(FadeIn(Group(plus, a, b)))
        self.next_slide()

        a_shape = Tex('(', '1', ',', ' 3', ')')
        b_shape = Tex('(', '2', ',', ')')
        a_shape.next_to(a, DOWN)
        b_shape.next_to(b, DOWN)
        a_shape.align_to(b_shape, UP)
        self.play(FadeIn(Group(a_shape, b_shape)))
        self.next_slide()

        b_broadcast_shape = Tex('(', '2', ',', ' 1', ')')
        b_broadcast_shape.move_to(b_shape)
        self.play(TransformMatchingTex(b_shape, b_broadcast_shape))
        self.next_slide()

        a_expanded_shape = Tex('(', '2', ',', ' 3', ')')
        b_expanded_shape = Tex('(', '2', ',', ' 3', ')')
        a_expanded_shape.move_to(a_shape)
        b_expanded_shape.move_to(b_broadcast_shape)
        b_expanded_shape.shift(RIGHT * b.square_size)

        self.play(AnimationGroup(
            a.animate.expand(0, 2),
            b.animate.expand(1, 3, recenter=False),
            TransformMatchingTexInOrder(a_shape, a_expanded_shape),
            TransformMatchingTexInOrder(b_broadcast_shape, b_expanded_shape),
        ))
        self.next_slide()

        # self.play(a.elementwise_op(b))
        # self.play(a.animate.elementwise_op(b))
        # # self.wait(1)
        # self.next_slide()
        # self.play(a.animate.highlight(YELLOW))
        # self.next_slide()
        self.play(AnimationGroup(
            a.animate.__iadd__(b),
            FadeOut(plus),
            FadeOut(b_expanded_shape)
        ))
        self.play(a.animate.center(), FadeOut(a_expanded_shape))
        self.next_slide()

    def stride_slide(self):
        title = self.slide_title('Strides (and Offsets)')
        self.transition(title)
        self.next_slide()

        # normal example

        tensor = Tensor2D(4, 6, 0.75, content=np.arange(0, 4*6).reshape(4, 6))
        strided_tensor = deepcopy(tensor)
        offset_tensor = deepcopy(tensor)
        # text = MathTex(r'i \cdot 6 + j \cdot 1')
        # text = MathTex(
        #     'i' + r'\iffalse 0 \fi',
        #     r'\cdot',
        #     '6' + r'\iffalse 1 \fi',
        #     '+',
        #     'j' + r'\iffalse 2 \fi',
        #     r'\cdot',
        #    '1' + r'\iffalse 3 \fi',
        # )
        text = MathTex(
            'i',
            r'\cdot',
            '6',
            '+',
            'j',
            r'\cdot',
           '1',
        )
        tensor_text = Tex(r'my\_tensor[', r'...', r']')
        # tensor_text._break_up_by_substrings()
        tensor_text.next_to(tensor, UP)

        vd = VDict({
            'tensor': tensor,
            'text': text
        })
        text.next_to(tensor, DOWN)
        self.play(FadeIn(Group(tensor, text, tensor_text)))
        self.next_slide()

        self.play_sequence(tensor.enumerate(vd, 'stride_indexing'))
        self.next_slide()

        # strided example

        # new_tensor.set_opacity(0.25)
        def in_slice_offset(idx):
            i = idx // 6
            j = idx % 6
            return i % 2 == 0 and j % 3 == 0
        for idx, sq in enumerate(strided_tensor.squares.flatten()):
            if not in_slice_offset(idx):
                sq.fade(0.75)
        slice_text = Tex(r'my\_tensor[', r'::2, ::3', r']')
        slice_text.move_to(tensor_text[0])
        # new_text = MathTex(
        #     'i' + r'\iffalse 0 \fi',
        #     r'\cdot',
        #     '12' + r'\iffalse 1 \fi',
        #     '+',
        #     'j' + r'\iffalse 2 \fi',
        #     r'\cdot',
        #    '3' + r'\iffalse 3 \fi',
        # )
        new_text = MathTex(
            'i',
            r'\cdot',
            '12',
            '+',
            'j',
            r'\cdot',
           '3',
        )
        new_text.move_to(text)
        text = new_text
        self.play(AnimationGroup(
            ReplacementTransform(tensor, strided_tensor),
            TransformMatchingTex(tensor_text, slice_text, transform_mismatches=True, replace_mobject_with_target_in_scene=True),
            TransformMatchingTexInOrder(
                vd['text'],
                text,
                transform_mismatches=True,
                replace_mobject_with_target_in_scene=True,
            )
        ))
        self.next_slide()

        slice = strided_tensor[::2, ::3]
        vd['tensor'] = slice
        vd['text'] = text
        self.play_sequence(slice.enumerate(vd, 'stride_indexing'))
        self.next_slide()

        # strided and offset (slided) example
        def in_slice(idx):
            i = idx // 6
            j = idx % 6
            return (i-1) % 2 == 0 and (j-2) % 3 == 0
        for idx, sq in enumerate(offset_tensor.squares.flatten()):
            if not in_slice(idx):
                sq.fade(0.75)
        tensor_text = slice_text
        slice_text = Tex(r'my\_tensor[', r'1::2, 2::3', r']')
        slice_text.move_to(tensor_text)
        # new_text = MathTex(
        #     'i' + r'\iffalse 0 \fi',
        #     r'\cdot',
        #     '12' + r'\iffalse 1 \fi',
        #     '+',
        #     'j' + r'\iffalse 2 \fi',
        #     r'\cdot',
        #    '3' + r'\iffalse 3 \fi',
        # )
        new_text = MathTex(
            'i',
            r'\cdot',
            '12',
            '+',
            'j',
            r'\cdot',
           '3',
           '+',
           '8'
        )
        new_text.move_to(vd['text'])
        text = new_text
        self.play(AnimationGroup(
            ReplacementTransform(strided_tensor, offset_tensor),
            TransformMatchingTexInOrder(tensor_text, slice_text, transform_mismatches=False, replace_mobject_with_target_in_scene=True),
            TransformMatchingTexInOrder(
                vd['text'],
                text,
                transform_mismatches=False,
                replace_mobject_with_target_in_scene=True,
            )
        ))
        self.next_slide()

        slice = offset_tensor[1::2, 2::3]
        vd['tensor'] = slice
        vd['text'] = text
        self.play_sequence(slice.enumerate(vd, 'stride_indexing'))
        self.next_slide()

    def triton_load_slide(self):
        # self.next_slide = lambda: self.wait(1)
        tensor_size = 0.6
        title = self.slide_title('tl.load indexing')

        big_tensor = Tensor2D(8, 8, tensor_size, np.random.randint(0, 10, (8, 8)))
        big_tensor.to_corner(DR)

        code_area_line = Line(
            (-config['frame_x_radius'], 0, 0),
            (big_tensor.get_edge_center(LEFT)[0], 0, 0)
        )
        code_width = code_area_line.get_length() - 0.75
        code_center = code_area_line.get_midpoint()

        start = Group(title, big_tensor)

        self.transition(start)

        self.next_slide()

        self.play(big_tensor[4:, :4].animate.highlight(GOLD))

        self.next_slide()

        code = Tex(
            'tl.load(',
            'X_ptr+',
            'tl.arange(0,B)[None]',
            '+',
            'tl.arange(B,B*2)[:,None]',
            '*',
            'W',
            ', ...)',
            tex_environment='verbatim'
        )

        code.scale_to_fit_width(code_width)
        code.move_to(code_center)
        self.play(Write(code, run_time=0.85))

        self.next_slide()

        self.play(
            FadeOut(code[0]),
            FadeOut(code[-1]),
            code[1:-1].animate.scale_to_fit_width(code_width).move_to(code_center)
        )

        self.next_slide()

        # big_tensor.save_state()
        self.play(
            # code[1].animate.set_color(RED),
            Indicate(code[1], color=RED, run_time=5, scale_factor=1.5, rate_func=there_and_back_with_pause),
            Indicate(code[2:-1], color=GREY, run_time=5, scale_factor=0.9, rate_func=there_and_back_with_pause),
            # big_tensor[:1, :1].animate.highlight(RED)
            Indicate(big_tensor[:1, :1], color=RED, run_time=5, rate_func=there_and_back_with_pause)
        )


        self.next_slide()

        self.play(
            # Restore(big_tensor),
            # big_tensor[:1, :1].animate.reset_color(),
            FadeOut(code[1]),
            code[2:-1].animate.scale_to_fit_width(code_width).move_to(code_center)
        )

        self.next_slide()

        self.play(
            Indicate(code[6], color=BLUE, scale_factor=2, run_time=5, rate_func=there_and_back_with_pause),
            Indicate(code[2:-2], color=GREY, run_time=5, scale_factor=0.9, rate_func=there_and_back_with_pause),
            AnimationGroup(*[
                Indicate(big_tensor[:, i:i+1], color=BLUE, rate_func=there_and_back_with_pause)
                for i in range(0, big_tensor.M)
            ],
                lag_ratio=0.1,
                run_time=5
            )
        )

        self.next_slide()

        t0 = Tensor2D(1, 4, tensor_size, np.arange(0, 4)[None])
        t1 = Tensor2D(4, 1, tensor_size, np.arange(4, 8)[:, None])
        m = Tensor2D(1, 1, tensor_size, np.array([8])[None])
        # t0 = Tensor2D(1, 4, 1, np.arange(0, 4)[None])
        # t1 = Tensor2D(4, 1, 1, np.arange(4, 8)[:, None])
        # m = Tensor2D(1, 1, 1, np.array([8])[None])
        m.squares[0,0]['square'].fade(1)
        t0.to_edge(LEFT)
        t1.next_to(t0, RIGHT).shift(RIGHT*2)
        m.next_to(t1, RIGHT).shift(RIGHT*2)

        plus_dest = Line(t0.get_edge_center(RIGHT), t1.get_edge_center(LEFT)).get_midpoint()
        asterisk_dest = Line(t1.get_edge_center(RIGHT), m.get_edge_center(LEFT)).get_midpoint()
        self.play(AnimationGroup(
            ReplacementTransform(code[2], t0),
            ReplacementTransform(code[4], t1),
            ReplacementTransform(code[6], m),
            code[3].animate.move_to(plus_dest).scale(1.2),
            code[5].animate.move_to(asterisk_dest).scale(1.2)
        ))
        self.next_slide()


        self.play(m.animate.expand(0, 4))
        self.play(t1.animate.__imul__(m), FadeOut(code[5]))

        self.next_slide()

        self.play(AnimationGroup(
            t0.animate.expand(0, 4),
            t1.animate.expand(1, 4, recenter=False)
        ))

        self.next_slide()

        self.play(t0.animate.__iadd__(t1), FadeOut(code[3]))

        self.next_slide()

        self.play(t0.animate.highlight(GREEN))

        self.play_sequence(big_tensor.gather(t0))

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

        # self.bullet_slide(
        #     'Why Write a Kernel?',
        #     'Your code is impractically slow', 
        #     'Your code uses too much VRAM',
        #     'The built-in torch/jax operations cannot be combined to achieve what you want',
        # )

        # self.bullet_slide(
        #     'Why is Your Code Inefficient?',
        #     'You\'re using a python "for" or "while" loop', 
        #     'You\'re launching a lot of kernels',
        #     'You have unnecessary memory copies',
        #     'You have wasted computations',
        #     'Your memory access patterns are inefficient'
        # )

        # # self.bullet_slide(
        # #     'How GPUs Work',
        # #     '', 
        # #     '',
        # #     '',
        # # )

        # self.indexing_slide()

        # self.stride_slide()

        # self.broadcasting_slide()

        # self.coalesce_slide()

        self.triton_load_slide()

