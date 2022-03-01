from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np

from keras import __version__ as KERAS_VERSION
from keras.callbacks import CallbackList as KerasCallbackList

from rl.callbacks import Callback, CallbackList, \
    Visualizer, FileLogger, TrainIntervalLogger, ModelIntervalCheckpoint, \
    TrainEpisodeLogger

from rl.core import History

class MyCallback(Callback):
    def on_episode_end(self, episode, logs={}, single_cycle=None):
        """Called at end of each episode"""
        pass

class MyCallbackList(CallbackList):
    def on_episode_end(self, episode, logs={}, single_cycle=None):
        """ Called at end of each episode for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_end` callback.
            # If not, fall back to `on_epoch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs,
                                        single_cycle=single_cycle)
            else:
                callback.on_epoch_end(episode, logs=logs)

    def on_train_end(self, logs={}, single_cycle=None):
        """ Called at end of training for each callback in callbackList"""
        # We have to check if the callback supports the single cycle argument or not
        for callback in self.callbacks:
            if isinstance(callback, TrainEpisodeLogger):
                callback.on_train_end(logs, single_cycle)
            else:
                callback.on_train_end(logs)

class MyTestLogger(MyCallback):
    """ Logger Class for Test """

    def __init__(self, interval=25):
        self.interval = interval

    def on_train_begin(self, logs):
        """ Print logs at beginning of training"""
        print('Testing for {} episodes ...'.format(self.params['nb_episodes']))

    def on_episode_end(self, episode, logs, single_cycle):
        if episode % self.interval == 0:
            if not single_cycle:
                variables = {
                    'episode': episode + 1,
                    'episode_reward': logs['episode_reward'],
                    'episode_steps': logs['nb_episode_steps'],
                    'episode_lifetime': logs['episode_lifetime'],
                    'episode_lifetimes_rolling_avg': logs[
                        "episode_lifetimes_rolling_avg"]}

                template = """-----------------
Episode: {episode}
This Episode Length: {episode_steps}
This Episode Reward: {episode_reward}

Rolling Lifetimes Avg: {episode_lifetimes_rolling_avg:.3f}
"""
            else:

                variables = {
                    'episode': episode + 1,
                    'episode_reward': logs['episode_reward'],
                    'episode_steps': logs['nb_episode_steps'],
                    'rolling_win_fraction': logs["rolling_win_fraction"]}

                template = """-----------------
Episode: {episode}
Rolling Win Fraction: {rolling_win_fraction:.3f}"""

            print(template.format(**variables))


class MyTrainEpisodeLogger(TrainEpisodeLogger, MyCallback):
    def __init__(self, interval=25):
        super(MyTrainEpisodeLogger, self).__init__()
        self.interval = interval

    def on_train_end(self, logs, single_cycle):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start

        if not single_cycle:
            variables = {"succeeded": logs['has_succeeded'],
                         "stopped_improving": logs['stopped_improving'],
                         "episode_lifetimes_rolling_avg": logs[
                             'episode_lifetimes_rolling_avg'],
                         "duration": duration,
                         "step": logs["step"]}

            template = """Training Finished in {duration:.3f} seconds

Final Step: {step}
Succeeded: {succeeded}
Stopped_Improving: {stopped_improving}
Final Episode Lifetimes Rolling Avg: {episode_lifetimes_rolling_avg:.3f}"""
            print(template.format(**variables))

        else:
            variables = {"succeeded": logs['has_succeeded'],
                         "stopped_improving": logs['stopped_improving'],
                         "rolling_success_prob": logs['rolling_win_fraction'],
                         "duration": duration,
                         "step": logs["step"]}

            template = """Training Finished in {duration:.3f} seconds

Final Step: {step}
Succeeded: {succeeded}
Stopped_Improving: {stopped_improving}
Final Num Errors Rolling Avg: {rolling_success_prob:.3f}"""
            print(template.format(**variables))

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs, single_cycle):
        """ Compute and print training statistics of the episode when done """
        if (episode + 1) % self.interval == 0:
            duration = timeit.default_timer() - self.episode_start[episode]
            episode_steps = len(self.observations[episode])

            # Format all metrics.
            metrics = np.array(self.metrics[episode])
            metrics_template = ''
            metrics_variables = []
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                for idx, name in enumerate(self.metrics_names):
                    if idx > 0:
                        metrics_template += ', '
                    try:
                        value = np.nanmean(metrics[:, idx])
                        metrics_template += '{}: {:f}'
                    except Warning:
                        value = '--'
                        metrics_template += '{}: {}'
                    metrics_variables += [name, value]
            metrics_text = metrics_template.format(*metrics_variables)

            elapsed_duration = timeit.default_timer() - self.train_start

            if not single_cycle:
                variables = {
                    'step': self.step,
                    'nb_steps': self.params['nb_steps'],
                    'episode': episode + 1,
                    'duration': duration,
                    'episode_reward': np.sum(self.rewards[episode]),
                    'episode_steps': episode_steps,
                    'elapsed_duration': elapsed_duration,
                    'metrics': metrics_text,
                    'episode_lifetimes_rolling_avg': logs[
                        "episode_lifetimes_rolling_avg"],
                    'best_rolling_avg': logs["best_rolling_avg"],
                    'best_episode': logs["best_episode"],
                    'time_since_best': logs["time_since_best"],
                    'has_succeeded': logs["has_succeeded"],
                    'stopped_improving': logs["stopped_improving"]}

                template = """-----------------

Episode: {episode}
Step: {step}/{nb_steps}
This Episode Steps: {episode_steps}
This Episode Reward: {episode_reward}
This Episode Duration: {duration:.3f}s
Rolling Lifetime length: {episode_lifetimes_rolling_avg:.3f}
Best Lifetime Rolling Avg: {best_rolling_avg}
Best Episode: {best_episode}
Time Since Best: {time_since_best}
Has Succeeded: {has_succeeded}
Stopped Improving: {stopped_improving}
Metrics: {metrics}
Total Training Time: {elapsed_duration:.3f}s
"""
                print(template.format(**variables))

            else:
                variables = {
                    'step': self.step,
                    'nb_steps': self.params['nb_steps'],
                    'episode': episode + 1,
                    'duration': duration,
                    'episode_reward': np.sum(self.rewards[episode]),
                    'episode_steps': episode_steps,
                    'elapsed_duration': elapsed_duration,
                    'metrics': metrics_text,
                    'rolling_win_fraction': logs["rolling_win_fraction"],
                    'best_rolling_fraction': logs["best_rolling_fraction"],
                    'best_episode': logs["best_episode"],
                    'time_since_best': logs["time_since_best"],
                    'has_succeeded': logs["has_succeeded"],
                    'stopped_improving': logs["stopped_improving"]}

                template = """-----------------

Episode: {episode}
Step: {step}/{nb_steps}
This Episode Steps: {episode_steps}
This Episode Reward: {episode_reward}
This Episode Duration: {duration:.3f}s
Rolling Win Fraction: {rolling_win_fraction:.3f}
Best Rolling Win Fraction: {best_rolling_fraction}
Best Episode: {best_episode}
Time Since Best: {time_since_best}
Has Succeeded: {has_succeeded}
Stopped Improving: {stopped_improving}
Metrics: {metrics}
Total Training Time: {elapsed_duration:.3f}s
"""
                print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1


class MyTrainIntervalLogger(TrainIntervalLogger, MyCallback):
    def __init__(self, interval=10000):
        super(MyTrainIntervalLogger, self).__init__(interval)

class MyFileLogger(FileLogger, MyCallback):
    def __init__(self, filepath, interval=None):
        super(MyFileLogger, self).__init__(filepath, interval)

    def on_episode_end(self, episode, logs, single_cycle=None):
        # Is this the method from the FileLoger instead of from the MyCallback?
        MyCallback.on_episode_end(episode, logs, single_cycle)


class MyVisualizer(Visualizer, MyCallback):
    def __init__(self):
        super(MyVisualizer, self).__init__()

    def on_episode_end(self, episode, logs, single_cycle=None):
        # Is this the method from the FileLoger instead of from the MyCallback?
        MyCallback.on_episode_end(episode, logs, single_cycle)

class MyModelIntervalCheckpoint(ModelIntervalCheckpoint, MyCallback):
    def __init__(self, filepath, interval, verbose=0):
        super(MyModelIntervalCheckpoint, self).__init__(filepath, interval, verbose)

    def on_episode_end(self, episode, logs, single_cycle=None):
        # Is this the method from the FileLoger instead of from the MyCallback?
        MyCallback.on_episode_end(episode, logs, single_cycle)

class MyHistory(History):
    def __init__(self):
        super(MyHistory, self).__init__()

    # def on_episode_end(self, episode, logs, single_cycle=None):
    #     # Is this the method from the FileLoger instead of from the MyCallback?
    #     MyCallback.on_episode_end(episode, logs, single_cycle)

