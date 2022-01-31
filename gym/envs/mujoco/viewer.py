"""Rendering simulation using dm_control."""

import sys

import numpy as np
import dm_control.mujoco as dm_mujoco
import dm_control.viewer as dm_viewer
import dm_control._render as dm_render

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 640
DEFAULT_WINDOW_HEIGHT = 480

# Default window title.
DEFAULT_WINDOW_TITLE = 'MuJoCo Viewer'

# Internal renderbuffer size, in pixels.
_MAX_RENDERBUFFER_SIZE = 2048

class WindowViewer:
    """Renders DM Control Physics objects."""

    def __init__(self, sim: dm_mujoco.Physics):
        self._window = None
        self._sim = sim
        self.set_free_camera_settings()

    def render_to_window(self):
        """Renders the Physics object to a window.
        The window continuously renders the Physics in a separate thread.
        This function is a no-op if the window was already created.
        """
        if not self._window:
            self._window = DMRenderWindow()
            self._window.load_model(self._sim)
            self._update_camera_properties(self._window.camera)

        self._window.run_frame()

    def refresh_window(self):
        """Refreshes the rendered window if one is present."""
        if self._window is None:
            return
        self._window.run_frame()

    def close(self):
        """Cleans up any resources being used by the renderer."""
        if self._window:
            self._window.close()
            self._window = None
    
    def set_free_camera_settings(
            self,
            trackbodyid = None,
            distance = None,
            azimuth = None,
            elevation = None,
            lookat = None,
            center = True,
    ):
        """Sets the free camera parameters.
        Args:
            distance: The distance of the camera from the target.
            azimuth: Horizontal angle of the camera, in degrees.
            elevation: Vertical angle of the camera, in degrees.
            lookat: The (x, y, z) position in world coordinates to target.
            center: If True and `lookat` is not given, targets the camera at the
                median position of the simulation geometry.
        """
        settings = {}
        if trackbodyid is not None:
            settings['trackbodyid'] = trackbodyid
        if distance is not None:
            settings['distance'] = distance
        if azimuth is not None:
            settings['azimuth'] = azimuth
        if elevation is not None:
            settings['elevation'] = elevation
        if lookat is not None:
            settings['lookat'] = np.array(lookat, dtype=np.float32)
        elif center:
            # Calculate the center of the simulation geometry.
            settings['lookat'] = np.array(
                [np.median(self._sim.data.geom_xpos[:, i]) for i in range(3)],
                dtype=np.float32)

        self._camera_settings = settings

    def close(self):
        """Cleans up any resources being used by the renderer."""

    def _update_camera_properties(self, camera):
        """Updates the given camera object with the current camera settings."""
        for key, value in self._camera_settings.items():
            if key == 'lookat':
                getattr(camera, key)[:] = value
            else:
                setattr(camera, key, value)

    def __del__(self):
        """Automatically clean up when out of scope."""
        self.close()


class DMRenderWindow:
    """Class that encapsulates a graphical window."""

    def __init__(self,
                 width = DEFAULT_WINDOW_WIDTH,
                 height = DEFAULT_WINDOW_HEIGHT,
                 title = DEFAULT_WINDOW_TITLE):
        """Creates a graphical render window.
        Args:
            width: The width of the window.
            height: The height of the window.
            title: The title of the window.
        """
        self._viewport = dm_viewer.renderer.Viewport(width, height)
        self._window = dm_viewer.gui.RenderWindow(width, height, title)
        self._viewer = dm_viewer.viewer.Viewer(
            self._viewport, self._window.mouse, self._window.keyboard)
        self._draw_surface = None
        self._renderer = dm_viewer.renderer.NullRenderer()

    @property
    def camera(self):
        return self._viewer._camera._camera  # pylint: disable=protected-access

    def close(self):
        self._viewer.deinitialize()
        self._renderer.release()
        self._draw_surface.free()
        self._window.close()

    def load_model(self, physics):
        """Loads the given Physics object to render."""
        self._viewer.deinitialize()

        self._draw_surface = dm_render.Renderer(
            max_width=_MAX_RENDERBUFFER_SIZE, max_height=_MAX_RENDERBUFFER_SIZE)
        self._renderer = dm_viewer.renderer.OffScreenRenderer(
            physics.model, self._draw_surface)

        self._viewer.initialize(physics, self._renderer, touchpad=False)

    def run_frame(self):
        """Renders one frame of the simulation.
        NOTE: This is extremely slow at the moment.
        """
        # pylint: disable=protected-access
        glfw = dm_viewer.gui.glfw_gui.glfw
        glfw_window = self._window._context.window
        if glfw.window_should_close(glfw_window):
            sys.exit(0)

        self._viewport.set_size(*self._window.shape)
        self._viewer.render()
        pixels = self._renderer.pixels

        with self._window._context.make_current() as ctx:
            ctx.call(self._window._update_gui_on_render_thread, glfw_window,
                     pixels)
        self._window._mouse.process_events()
        self._window._keyboard.process_events()
        # pylint: enable=protected-access