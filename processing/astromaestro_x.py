"""
AstroMaestro RC Astro-inspired family.

These are not copies of RC Astro's proprietary AI models. They are transparent,
classical/heuristic implementations based on publicly documented workflow
principles such as:
- direct linear-safe handling for deconvolution
- stellar/non-stellar blending
- successive-approximation denoise
- intensity/color and HF/LF separation
- detail/strength-controlled gradient removal
- iterative star shrinking
- structure-safe star removal
"""

from processing.background import astro_gradient_x, gradient_terminator
from processing.deconvolution import astro_blur_x
from processing.mastro_starless import astro_star_x
from processing.noisexterminator import astro_noise_x
from processing.star_shrink import astro_star_shrink


SUITE = {
    "astro_blur_x": astro_blur_x,
    "astro_noise_x": astro_noise_x,
    "astro_star_x": astro_star_x,
    "astro_gradient_x": astro_gradient_x,
    "astro_star_shrink": astro_star_shrink,
    "gradient_terminator": gradient_terminator,
}
