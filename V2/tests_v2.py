"""Checks for the V2 capture rule and safe numerical reset options."""
import math
import pickle
import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from gym_env import ENV_ID,NightShiftEnv
from night_env import GoalShaping
from simulation import Game,Controls,CAPTURE_SECONDS,CAPTURE_RELEASE_DISTANCE


class CaptureTests(unittest.TestCase):
    def game_with_cop(self,distance=60,speed=0):
        game=Game()
        game.player.angle=0
        game.player.vx=speed
        game.cops=game.cops[:1]
        game.cops[0].x=game.player.x-distance
        game.cops[0].y=game.player.y
        game.capture=0.4
        # Keep the officer's position fixed while exercising real player physics.
        game._update_police=lambda cop,index,dt:None
        return game

    def test_fast_nearby_player_no_longer_clears_capture(self):
        game=self.game_with_cop(speed=120)
        game.step(Controls(throttle=1),0.1)
        self.assertGreater(game.player.speed,65)
        self.assertEqual(game.capture,0.4)
        self.assertEqual(game.capture_state,"held")
        self.assertEqual(game.health,100)

    def test_counter_clears_after_real_separation(self):
        game=self.game_with_cop(distance=300,speed=120)
        game.step(Controls(throttle=1),0.1)
        self.assertAlmostEqual(game.capture,0.4-0.065)
        self.assertEqual(game.capture_state,"releasing")

    def test_gap_between_capture_and_release_holds_the_counter(self):
        game=self.game_with_cop(distance=100)
        game.step(Controls(),0.1)
        self.assertEqual(game.capture,0.4)
        game.cops[0].x=game.player.x-CAPTURE_RELEASE_DISTANCE
        game._update_capture(0.1)
        self.assertAlmostEqual(game.capture,0.335)

    def test_repeated_slow_fast_cycles_cannot_reset_the_counter(self):
        game=self.game_with_cop()
        game.capture=0
        for _ in range(4):
            game.player.vx=0
            game._update_capture(0.5)
            before=game.capture
            game.player.vx=150
            game._update_capture(0.5)
            self.assertEqual(game.capture,before)
        self.assertAlmostEqual(game.capture,2/CAPTURE_SECONDS)

    def test_close_stationary_player_is_still_captured(self):
        game=self.game_with_cop()
        game.capture=0
        for _ in range(34):
            game.step(Controls(),0.1)
        self.assertEqual(game.status,"lost")
        self.assertEqual(game.reason,"The police boxed you in")
        self.assertEqual(game.capture,1)
        self.assertEqual(game.health,100)


class EnvironmentTests(unittest.TestCase):
    def test_checker_and_new_registration(self):
        env=NightShiftEnv()
        check_env(env,skip_render_check=True)
        with gym.make(ENV_ID) as registered:
            observation,info=registered.reset(seed=42)
            self.assertEqual(observation.shape,(60,))
            self.assertEqual(registered.action_space.n,15)
            self.assertEqual(info["env_version"],2)
        env.close()

    def test_offset_rebuilds_dispatch_and_observation(self):
        env=NightShiftEnv()
        observation,info=env.reset(seed=42,options={"start_offset":(30,-30,10)})
        player=env.game.player
        self.assertEqual((player.x,player.y),(1470,1830))
        self.assertAlmostEqual(player.angle,-math.pi/2+math.radians(10))
        self.assertEqual(env.game._pursuit.last_seen,(player.x,player.y))
        self.assertEqual(env.game._pursuit.last_velocity,(0,0))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["start_offset"],(30,-30,10))
        _,info=env.reset(seed=42)
        self.assertEqual((env.game.player.x,env.game.player.y),(1440,1860))
        self.assertEqual(info["start_offset"],(0,0,0))
        env.close()

    def test_invalid_offsets_are_rejected_without_changing_the_game(self):
        env=NightShiftEnv()
        env.reset(seed=42)
        game=env.game
        for options in [{"other":1},{"start_offset":(31,0,0)},
                        {"start_offset":(0,0,11)},{"start_offset":(0,0,float("nan"))},
                        {"start_offset":(True,0,0)},{"start_offset":(0,0)},
                        {"start_offset":np.array(0)},False]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    env.reset(seed=42,options=options)
                self.assertIs(env.game,game)
                self.assertEqual((game.player.x,game.player.y),(1440,1860))
        env.close()

    def test_jittered_replay_and_pickled_continuation_are_exact(self):
        env=GoalShaping(NightShiftEnv())
        offset={"start_offset":(-22,17,-8)}
        first,_=env.reset(seed=11,options=offset)
        first_run=[env.step(action) for action in [1,12,2,9,0,11]]
        second,_=env.reset(seed=11,options=offset)
        np.testing.assert_array_equal(first,second)
        second_run=[env.step(action) for action in [1,12,2,9,0,11]]
        for a,b in zip(first_run,second_run):
            np.testing.assert_array_equal(a[0],b[0])
            self.assertEqual(a[1:],b[1:])
        restored=pickle.loads(pickle.dumps(env))
        for action in [3,13,7,14]:
            a=env.step(action)
            b=restored.step(action)
            np.testing.assert_array_equal(a[0],b[0])
            self.assertEqual(a[1:],b[1:])
        env.close()
        restored.close()


if __name__=="__main__":
    unittest.main()
