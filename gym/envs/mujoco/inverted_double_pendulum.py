import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "inverted_double_pendulum.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.velocity()[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.position()[:1],  # cart x pos
                np.sin(self.sim.position()[1:]),  # link angles
                np.cos(self.sim.position()[1:]),
                np.clip(self.sim.velocity(), -10, 10),
                np.clip(self.sim.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.set_free_camera_settings(trackbodyid=0, distance=self.model.stat.extent*0.5,
                                             lookat=np.array((0.0, 0.0, 0.12250000000000005)))
