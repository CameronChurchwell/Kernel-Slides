from manim import *
import numpy as np
from copy import deepcopy
from helpers import *

from custom_animations import IndicationTransform

class Tensor2D(VGroup):
    def __init__(self, N, M, square_size, content=None, **kwargs):
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

    def highlight(self, color=YELLOW):
        self.set_color(color)
        self.set_fill(color, opacity=0.25)
        self.get_all_content().set_fill(color, opacity=1.)

    def reset_color(self):
        self.set_color(WHITE)
        self.set_fill(None, opacity=0.)

    def __getitem__(self, slices):
        result = self.squares[*slices]
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

    def set_content_at(self, i, j, content):
        self.content[i, j] = content
        tex = MathTex(str(content))
        square = self.squares[i, j]['square']
        if tex.height > tex.width:
            tex.scale_to_fit_height(square.width*0.7)
        else:
            tex.scale_to_fit_width(square.width*0.7)
        tex.move_to(square.get_center())
        self.squares[i, j]['content'] = tex

    def gather(self, index_tensor):
        """note that index_tensor should contain flattened indices"""
        assert isinstance(index_tensor, Tensor2D)
        try:
            indices = index_tensor.content.astype(int)
        except:
            raise ValueError('index_tensor cannot be converted to int')

        to_animations = []
        from_animations = []
        for i in range(0, index_tensor.N):
            for j in range(0, index_tensor.M):
                original_position = index_tensor[i, j]['square'].get_center()

                # move to animation
                index = indices[i, j]
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
                anim = MoveAlongPath(index_tensor[i, j], path)
                from_animations.append(anim)

        animations = Succession(
            AnimationGroup(
                *to_animations
            ),
            Wait(1),
            AnimationGroup(
                *from_animations
            ),
        )

        return [animations]
    
    # def stash_squares(self):
        # self._style_copy = deepcopy(self)

    # def restore_squares(self):
        # self.set_style(**self._style_copy.get_style())
        # for sq, osq in zip(self.squares.flatten(), self._style_copy.squares.flatten()):
        #     sq['square'].set_style(**osq['square'].get_style())
        #     sq['content'].set_style(**osq['content'].get_style())
        # self.squares = deepcopy(self._style_copy.squares)

    def coalesced_gather(self, index_tensor, warp_size=32):
        """note that index_tensor should contain flattened indices"""
        assert isinstance(index_tensor, Tensor2D)
        try:
            indices = index_tensor.content.astype(int)
        except:
            raise ValueError('index_tensor cannot be converted to int')

        # self.stash_squares()
        self.save_state()

        index_tensor.set_z_index(self.z_index+1)

        sorting = indices.argsort(axis=None)

        fade_opacity = 0.7
        fade_in_lag_ratio = 0.025

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
                        *[sq.animate.fade(fade_opacity) for sq in chunk_squares]
                    ),
                    AnimationGroup(
                        *[
                            # sq.animate.fade(0) for sq in new_chunk_squares
                            IndicationTransform(sq, deepcopy(sq).fade(0)) for sq in new_chunk_squares
                        ],
                        lag_ratio=fade_in_lag_ratio
                    )
                ))
                chunk_squares = new_chunk_squares
                from_animations.append(AnimationGroup(*current_group))
                current_group = []
                chunk_start = (index // warp_size) * warp_size

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
            anim = MoveAlongPath(index_tensor[i, j], path)
            current_group.append(anim)

        new_chunk_squares = self.squares.flatten()[chunk_start:chunk_start+warp_size]
        from_animations.append(AnimationGroup(
            AnimationGroup(
                *[sq.animate.fade(fade_opacity) for sq in chunk_squares]
            ),
            AnimationGroup(
                *[
                    IndicationTransform(sq, deepcopy(sq).fade(0)) for sq in new_chunk_squares
                ],
                lag_ratio=fade_in_lag_ratio
            )
        ))
        chunk_squares = new_chunk_squares
        from_animations.append(AnimationGroup(*current_group))
        from_animations.append(AnimationGroup(
            *[sq.animate.fade(fade_opacity) for sq in chunk_squares],
        ))

        yield AnimationGroup(*to_animations)
        yield self.animate.fade(fade_opacity)
        for anim in from_animations:
            yield anim
        yield self.animate.restore()