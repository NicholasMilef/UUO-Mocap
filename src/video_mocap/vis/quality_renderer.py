import os
import pyrender
from OpenGL.GL import *


QUALITY_SHADOW_TEX_SZ = 1024 * 32


# from https://github.com/mmatl/pyrender/blob/master/pyrender/offscreen.py
class OffscreenRendererQuality(pyrender.OffscreenRenderer):
    def __init__(self, viewport_width, viewport_height, point_size=1.0):
        super().__init__(viewport_width, viewport_height, point_size=1.0)

    def _create(self):
        if 'PYOPENGL_PLATFORM' not in os.environ:
            from pyrender.platforms.pyglet_platform import PygletPlatform
            self._platform = PygletPlatform(self.viewport_width,
                                            self.viewport_height)
        elif os.environ['PYOPENGL_PLATFORM'] == 'egl':
            from pyrender.platforms import egl
            device_id = int(os.environ.get('EGL_DEVICE_ID', '0'))
            egl_device = egl.get_device_by_index(device_id)
            self._platform = egl.EGLPlatform(self.viewport_width,
                                             self.viewport_height,
                                             device=egl_device)
        elif os.environ['PYOPENGL_PLATFORM'] == 'osmesa':
            from pyrender.platforms.osmesa import OSMesaPlatform
            self._platform = OSMesaPlatform(self.viewport_width,
                                            self.viewport_height)
        else:
            raise ValueError('Unsupported PyOpenGL platform: {}'.format(
                os.environ['PYOPENGL_PLATFORM']
            ))
        self._platform.init_context()
        self._platform.make_current()
        self._renderer = RendererQuality(self.viewport_width, self.viewport_height)


# from https://github.com/mmatl/pyrender/blob/master/pyrender/renderer.py
class RendererQuality(pyrender.Renderer):
    def __init__(self, viewport_width, viewport_height, point_size=1.0):
        super().__init__(viewport_width, viewport_height, point_size=1.0)

    def _configure_shadow_mapping_viewport(self, light, flags):
        self._configure_shadow_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fb)
        light.shadow_texture._bind()
        light.shadow_texture._bind_as_depth_attachment()
        glActiveTexture(GL_TEXTURE0)
        light.shadow_texture._bind()
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        
        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, QUALITY_SHADOW_TEX_SZ, QUALITY_SHADOW_TEX_SZ)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)


# https://github.com/mmatl/pyrender/blob/master/pyrender/light.py
class DirectionalLightQuality(pyrender.DirectionalLight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _generate_shadow_texture(self, size=None):
        if size is None:
            size = QUALITY_SHADOW_TEX_SZ
        self.shadow_texture = pyrender.Texture(width=size, height=size,
                                               source_channels='D', data_format=GL_FLOAT)

