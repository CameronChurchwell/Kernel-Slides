from manim import *

class IndicationTransform(Transform):

    def __init__(
        self,
        mobject: Mobject | None,
        target_mobject: Mobject | None = None,
        scale_factor: float = 1.1,
        path_func = None,
        path_arc: float = 0,
        path_arc_axis: np.ndarray = OUT,
        path_arc_centers: np.ndarray = None,
        replace_mobject_with_target_in_scene: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            mobject,
            target_mobject,
            path_func,
            path_arc,
            path_arc_axis,
            path_arc_centers,
            replace_mobject_with_target_in_scene,
            **kwargs
        )
        self.indication_copy = mobject.copy().scale(scale_factor)

    def interpolate_mobject(self, alpha: float) -> None:
        """Interpolates the mobject of the :class:`Animation` based on alpha value.

        Parameters
        ----------
        alpha
            A float between 0 and 1 expressing the ratio to which the animation
            is completed. For example, alpha-values of 0, 0.5, and 1 correspond
            to the animation being completed 0%, 50%, and 100%, respectively.
        """
        tc = self.target_copy
        s = self.starting_mobject
        if alpha < 0.5:
            self.starting_mobject = s
            self.target_copy = self.indication_copy
            alpha = alpha * 2
        else:
            self.starting_mobject = self.indication_copy
            self.target_copy = tc
            alpha = (alpha - 0.5) * 2
        retval = super().interpolate_mobject(alpha)
        self.target_copy = tc
        self.starting_mobject = s
        return retval


class TransformMatchingTexInOrder(TransformMatchingTex):
    def get_shape_map(self, mobject: Mobject) -> dict:
        shape_map = {}
        for i, sm in enumerate(self.get_mobject_parts(mobject)):
            key = i
            if key not in shape_map:
                shape_map[key] = VGroup()
            shape_map[key].add(sm)
        return shape_map