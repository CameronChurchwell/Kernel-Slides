from manim import *
import types
from typing import Iterable, Callable

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

    # def interpolate_submobject(
    #     self,
    #     submobject: Mobject,
    #     starting_submobject: Mobject,
    #     target_copy: Mobject,
    #     alpha: float,
    # ) -> Transform:
    #     submobject.interpolate(starting_submobject, self.target_copy, alpha, self.path_func)
    #     # if alpha < 0.5:
    #     #     submobject.interpolate(starting_submobject, self.indication_copy, alpha*2, self.path_func)
    #     # else:
    #     #     submobject.interpolate(self.indication_copy, target_copy, (alpha-0.5)*2, self.path_func)
    #     return self
    
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
    
class CleanupAfter(AnimationGroup):
    def __init__(
        self,
        *animations: Animation | Iterable[Animation] | types.GeneratorType[Animation],
        cleanup_func: Callable = None,
        group: Group | VGroup = None,
        run_time: float | None = None,
        rate_func: Callable[[float], float] = linear,
        lag_ratio: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            *animations, 
            group=group, 
            run_time=run_time,
            rate_func=rate_func,
            lag_ratio=lag_ratio,
            **kwargs
        )
        self.cleanup_func = cleanup_func

    def clean_up_from_scene(self, scene):
        # self.source.highlight(GREEN)
        # self.target.highlight(RED)
        self.cleanup_func(scene)
        # return super().clean_up_from_scene(scene)