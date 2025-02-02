from manim import *
import numpy as np
from copy import deepcopy

from custom_animations import *

class Tensor2D(VGroup):
    def __init__(self, N, M, square_size, content=None, **kwargs):
        self.squares = np.array([[]])
        super().__init__()
        self.N = N
        self.M = M
        self.square_size = square_size
        self.squares = np.empty((N, M), dtype=object)
        self.content = np.empty((N, M), dtype=object)

        # create squares
        for i in range(0, N):
            for j in range(0, M):
                square = Square(side_length=square_size, **kwargs)
                if 'stroke_width' not in kwargs:
                    square.set_stroke(width=2)
                square.move_to([j*square_size, -i*square_size, 0])
                square = VDict({
                    'square': square, 
                    'content': MathTex(''),
                })
                self.squares[i,j] = square
                self.add(square)
        self.set_content(content)
        self.move_to((0, 0, 0))
        self.saved_states = []

    def highlight(self, color=YELLOW):
        self.set_color(color)
        self.set_fill(color, opacity=0.25)
        self.get_all_content().set_fill(color, opacity=1.)

    def reset_color(self):
        self.set_color(WHITE)
        self.set_fill(None, opacity=0.)
        for sq in self.submobjects:
            sq['content'].set_fill(WHITE, opacity=1.)

    def __getitem__(self, slices):
        result = self.squares[slices]
        if isinstance(result, np.ndarray):
            slice = Tensor2D(result.shape[0], result.shape[1], self.square_size)
            slice.squares = result
            slice.submobjects = []
            slice.add(list(result.flatten()))
            return slice
        else:
            return result

    def clone(self):
        return deepcopy(self)

    def get_all_content(self):
        return VGroup(
            sq['content'] for sq in self.squares.flatten()
        )

    def get_content_at(self, i, j):
        return self.squares[i, j]['content']

    def set_content(self, content=None):
        N = self.N
        M = self.M
        if content is None:
            content = np.empty((N, M), dtype=object)
            content[:, :] = ''
        elif isinstance(content, np.ndarray):
            assert content.shape == (N, M), (content.shape, (N, M))
        elif content == "indices":
            content = np.array([[f'{i}{{,}}{j}' for j in range(M)] for i in range(N)])
        elif content == "flat_indices":
            content = np.array([[i*M+j for j in range(M)] for i in range(N)])
        else:
            raise ValueError('unsupported content type:', content)

        assert content.shape == self.squares.shape
        for i in range(0, N):
            for j in range(0, M):
                self.set_content_at(i, j, content[i, j])

    def set_content_at(self, i, j, *content):
        try:
            self.content[i, j] = content[0] if len(content) == 1 else content
        except:
            breakpoint()
        tex = MathTex(*list(str(c) for c in content))
        square = self.squares[i, j]['square']
        if tex.height > tex.width:
            tex.scale_to_fit_height(square.width*0.7)
        else:
            tex.scale_to_fit_width(square.width*0.7)
        tex.move_to(square.get_center())
        self.squares[i, j]['content'] = tex

    def update_tex_strings_from(self, other):
        assert self.squares.shape == other.squares.shape
        for i in range(0, self.N):
            for j in range(0, self.M):
                self.squares[i, j]['content'].tex_string = other.squares[i, j]['content'].tex_string

    def gather_from(self, value_tensor):
        assert isinstance(value_tensor, Tensor2D)
        indices = self.content.astype(int)
        values = value_tensor.content.flatten()

        for i in range(0, self.N):
            for j in range(0, self.M):
                old_tex = self.squares[i, j]['content']
                self.set_content_at(i, j, values[indices[i, j]])
                self.squares[i, j]['content'].set_style(**old_tex.get_style())
        return self

    @override_animate(gather_from)
    def gather_from_animate(self, value_tensor, anim_args=None):
        index_tensor = deepcopy(self)
        self.gather_from(value_tensor)
        indices = index_tensor.content.astype(int)

        from_animations = []
        for i in range(0, self.N):
            for j in range(0, self.M):
                original_position = self[i, j]['square'].get_center()

                # move to
                index = indices[i, j]
                target = value_tensor.squares.flatten()[index]
                dest = target.get_center()
                path = Line(original_position, dest)
                self[i, j].move_to(dest)

                # move back (from) animations
                path = Line(dest, original_position)
                anim = MoveAlongPath(self[i, j], path)
                from_animations.append(anim)

        self.save_state()
        self.become(index_tensor)

        animations = Succession(
            Restore(self),
            Wait(1),
            AnimationGroup(
                *from_animations
            ),
        )

        return animations

    def coalesced_gather(self, index_tensor, warp_size=32):
        """note that index_tensor should contain flattened indices"""
        assert isinstance(index_tensor, Tensor2D)
        try:
            indices = index_tensor.content.astype(int)
        except:
            raise ValueError('index_tensor cannot be converted to int')

        self.save_state()

        index_tensor.set_z_index(self.z_index+1)

        sorting = indices.argsort(axis=None)

        fade_opacity = 0.7
        fade_in_lag_ratio = 0.025

        run_time = 1.0

        to_animations = []
        from_animations = []
        chunk_start = None
        chunk_squares = None
        current_group = []
        for idx in sorting:

            i = idx // index_tensor.M
            j = idx % index_tensor.M
            original_position = index_tensor[i, j]['square'].get_center()
            index = indices[i, j]

            if chunk_start is None:
                chunk_start = (index // warp_size) * warp_size
                chunk_squares = self.squares.flatten()[chunk_start:chunk_start+warp_size]
            if index >= chunk_start + warp_size:
                new_chunk_squares = self.squares.flatten()[chunk_start:chunk_start+warp_size]
                from_animations.append(AnimationGroup(
                    AnimationGroup(
                        *[AnimationGroup(sq.animate.fade(fade_opacity), run_time=run_time) for sq in chunk_squares]
                    ),
                    AnimationGroup(
                        *[
                            IndicationTransform(sq, deepcopy(sq).fade(0), run_time=run_time) for sq in new_chunk_squares
                        ],
                        lag_ratio=fade_in_lag_ratio
                    )
                ))
                chunk_squares = new_chunk_squares
                from_animations.append(AnimationGroup(*current_group))
                current_group = []
                chunk_start = (index // warp_size) * warp_size
                run_time = run_time * 0.9

            # move to animation
            target = self.squares.flatten()[index]
            dest = target.get_center()
            path = Line(original_position, dest)
            anim = MoveAlongPath(index_tensor[i, j]['square'], path)
            to_animations.append(anim)

            # indices become values animation
            current_content = index_tensor[i, j]['content']
            target_content: MathTex = self.squares.flatten()[index]['content'].copy()
            target_content.set_style(**current_content.get_style())
            anim = Transform(
                current_content,
                target_content
            )
            to_animations.append(anim)

            # move back (from) animations
            path = Line(dest, original_position)
            anim = MoveAlongPath(index_tensor[i, j], path, run_time=run_time)
            current_group.append(anim)

        new_chunk_squares = self.squares.flatten()[chunk_start:chunk_start+warp_size]
        from_animations.append(AnimationGroup(
            AnimationGroup(
                *[AnimationGroup(sq.animate.fade(fade_opacity), run_time=run_time) for sq in chunk_squares]
            ),
            AnimationGroup(
                *[
                    IndicationTransform(sq, deepcopy(sq).fade(0), run_time=run_time) for sq in new_chunk_squares
                ],
                lag_ratio=fade_in_lag_ratio
            )
        ))
        chunk_squares = new_chunk_squares
        from_animations.append(AnimationGroup(*current_group))
        from_animations.append(AnimationGroup(
            *[AnimationGroup(sq.animate.fade(fade_opacity), run_time=run_time) for sq in chunk_squares],
        ))

        yield AnimationGroup(*to_animations)
        yield self.animate.fade(fade_opacity)
        for anim in from_animations:
            yield anim
        yield self.animate.restore()

    def enumerate(self, vd, function='flat_indices'):
        text = vd['text']
        original_text = deepcopy(text)
        if function == 'flat_indices':
            function = lambda idx, content: [str(idx)]
        elif function == 'indices':
            function = lambda idx, content: [str((idx // self.M, idx % self.M))]
        elif function == 'stride_indexing':
            itemsize = self.squares.itemsize
            def function(idx, content):
                i = idx // self.M
                j = idx % self.M
                i_stride, j_stride = self.squares.strides
                if self.squares.base is not None:
                    offset = (self.squares.__array_interface__['data'][0] - self.squares.base.__array_interface__['data'][0]) // self.squares.itemsize
                else:
                    offset = 0
                output = [
                    str(i),
                    r'\cdot',
                    str(i_stride//itemsize),
                    '+',
                    str(j),
                    r'\cdot',
                    str(j_stride//itemsize),
                ]
                if offset > 0:
                    output.append('+')
                    output.append(str(offset))
                return output
        for idx in range(0, len(self.content.flatten())):
            new_text = MathTex(*function(idx, self.content.flatten()))
            new_text.move_to(text)
            new_text.align_to(text, LEFT)
            new_text.align_to(text, UP)
            yield AnimationGroup(
                Indicate(self.squares.flatten()[idx]),
                TransformMatchingTexInOrder(
                    text,
                    new_text,
                    run_time=0.5,
                    transform_mismatches=False,
                    fade_transform_mismatches=False,
                    replace_mobject_with_target_in_scene=True,
                )
            )
            text = new_text
        yield TransformMatchingTexInOrder(
            text,
            original_text,
            run_time=0.5,
            transform_mismatches=False,
            fade_transform_mismatches=False,
            replace_mobject_with_target_in_scene=True,
        )
        vd['text'] = original_text

    def expand(self, axis, new_size, recenter=True):
        assert self.squares.shape[axis] == 1, breakpoint()
        squares = self.squares.flatten()
        original_position = self.get_center()
        new_shape = (new_size, self.M) if axis == 0 else (self.N, new_size)
        self.N = new_shape[0]
        self.M = new_shape[1]
        self.content = np.broadcast_to(self.content, new_shape)
        self.content = self.content.copy().reshape(self.content.shape)
        self.submobjects = []
        self.squares = np.array([squares] + [
            deepcopy(squares) for _ in range(1, new_size)
        ])
        shift_vector = (self.square_size, 0, 0) if axis == 1 else (0, -self.square_size, 0)
        for i, vector in enumerate(self.squares):
            for sq in vector:
                self.add(sq)
                for _ in range(0, i):
                    sq.shift(shift_vector)
        if axis == 1:
            self.squares = self.squares.T
        if recenter:
            self.move_to(original_position)

    @override_animate(expand)
    def expand_animate(self, axis, new_size, recenter=True, anim_args=None):
        original_position = self.get_center()
        self.expand(axis, new_size, recenter=recenter)
        self.save_state()
        if axis == 0:
            first_group = self[:1]
            rest_group = self[1:]
        else:
            first_group = self[:, :1]
            rest_group = self[:, 1:]
        first_group.move_to(original_position)
        rest_group.fade(1.0)
        return Restore(self, **anim_args)

    def elementwise_op(self, other, op_string='+'):
        assert self.content.shape == other.content.shape
        for i in range(0, self.N):
            for j in range(0, self.M):
                self_tex_string = self.squares[i, j]['content'].tex_string
                other_tex_string = other.squares[i, j]['content'].tex_string
                self.set_content_at(i, j, self_tex_string, op_string, other_tex_string)

    @override_animate(elementwise_op)
    def elementwise_op_animate(self, other, op_string='+', anim_args=None):
        if hasattr(self, 'saved_state'):
            del self.saved_state
        source = deepcopy(self)
        self.elementwise_op(other, op_string)
        animations = []
        if hasattr(self, 'saved_state'):
            breakpoint()
        self.save_state()
        for i in range(0, self.N):
            for j in range(0, self.M):
                source_tex = source.squares[i, j]['content']
                other_tex = other.squares[i, j]['content']
                target_tex = self.squares[i, j]['content']
                target_tex[0].become(source_tex)
                target_tex[2].become(other_tex)
                target_tex[1].fade(1)
                other_tex.fade(1.) # no need for the duplicate number

                source_sq = source.squares[i, j]['square']
                other_sq = other.squares[i, j]['square']
                source_sq.fade(1) # no need for the duplicate square, nothing is changing for it

                animations.append(FadeOut(other_sq))
        animations.append(Restore(self))
        return AnimationGroup(*animations)

    def __iadd__(self, other):
        content_result = self.content + other.content
        self.set_content(content_result)
        return self

    @override_animate(__iadd__)
    def __iadd__animate(self, other, anim_args=None):
        target = deepcopy(self)
        target += other
        animations = Succession(
            self.animate.elementwise_op(other, '+'),
            self.animate.become(target)
        )
        self.content = target.content
        self.update_tex_strings_from(target)
        return animations

    def __isub__(self, other):
        content_result = self.content - other.content
        self.set_content(content_result)
        return self

    @override_animate(__isub__)
    def __isub__animate(self, other, anim_args=None):
        target = deepcopy(self)
        target -= other

        animations = Succession(
            self.animate.elementwise_op(other, '-'),
            self.animate.become(target)
        )
        self.content = target.content
        self.update_tex_strings_from(target)
        return animations

    def __imul__(self, other):
        content_result = self.content * other.content
        self.set_content(content_result)
        return self

    @override_animate(__imul__)
    def __imul__animate(self, other, anim_args=None):
        target = deepcopy(self)
        target *= other

        animations = Succession(
            self.animate.elementwise_op(other, r'\cdot'),
            self.animate.become(target)
        )
        self.content = target.content
        self.update_tex_strings_from(target)
        return animations

    def __itruediv__(self, other):
        content_result = self.content / other.content
        self.set_content(content_result)
        return self

    @override_animate(__itruediv__)
    def __itruediv__animate(self, other, anim_args=None):
        target = deepcopy(self)
        target /= other

        animations = Succession(
            self.animate.elementwise_op(other, '/'),
            self.animate.become(target)
        )
        self.content = target.content
        self.update_tex_strings_from(target)
        return animations

