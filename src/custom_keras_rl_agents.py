from __future__ import division
import warnings
from copy import deepcopy

import os

from collections import deque

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
from rl.callbacks import TrainIntervalLogger, Visualizer

from keras.callbacks import History

from custom_keras_rl_policy import MyEpsGreedyQPolicy, MyGreedyQPolicy
from rl.util import *

from rl.agents.dqn import DQNAgent

# class MyAbstractDQNAgent(AbstractDQNAgent):
#     """Write me
#     """
#
#     def __init__(self, nb_actions, memory, gamma=.99, batch_size=32,
#                  nb_steps_warmup=1000,
#                  train_interval=1, memory_interval=1, target_model_update=10000,
#                  delta_range=None, delta_clip=np.inf, custom_model_objects={},
#                  *args,
#                  **kwargs):
#         super(MyAbstractDQNAgent, self).__init__(nb_actions, memory, gamma,
#                                                  batch_size, nb_steps_warmup,
#                  train_interval, memory_interval, target_model_update,
#                  delta_range, delta_clip, custom_model_objects, **kwargs)
#
    # def compute_q_values(self, state):
    #     # state = np.expand_dims(state[0], 0)
    #     q_values = self.compute_batch_q_values(state).flatten()
    #     assert q_values.shape == (self.nb_actions,)
    #     return q_values


# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
from src.custom_keras_rl_callbacks import MyTestLogger, MyCallbackList, \
    MyFileLogger, MyTrainEpisodeLogger


class MyDQNAgent(DQNAgent):
    """
    # Arguments
        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)

    """

    def __init__(self, model, policy=None, test_policy=None,
                 enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):

        super(MyDQNAgent, self).__init__(model, policy, test_policy,
                 enable_double_dqn, enable_dueling_network,
                 dueling_type, *args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError(
                'Model "{}" has more than one output. DQN expects a model that has a single output.'.format(
                    model))
        if model.output._keras_shape != (None, self.nb_actions):
            raise ValueError(
                'Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(
                    model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output._keras_shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(
                    lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(
                        a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(
                    lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(
                        a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(
                    lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:],
                    output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=outputlayer)

        # Related objects.
        self.model = model
        if policy is None:
            policy = MyEpsGreedyQPolicy()
        if test_policy is None:
            test_policy = MyGreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        # State.
        self.reset_states()

    # Overwritting from DQNAgent
    def compute_q_values(self, state):
        state = np.expand_dims(state[0], 0)
        q_values = self.compute_batch_q_values(state).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def forward(self, observation, legal_actions=None):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values,
                                               legal_actions=legal_actions)
        else:
            action = self.test_policy.select_action(q_values=q_values,
                                                    legal_actions=legal_actions)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action,
                               reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0[0])
                state1_batch.append(e.state1[0])
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(
                    state1_batch)
                assert target_q_values.shape == (
                self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(
                    state1_batch)
                assert target_q_values.shape == (
                self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(
                    zip(targets, masks, Rs, action_batch)):
                target[
                    action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(
                self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(
                ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    """
        These methods were in Agent. Now I am overwritting them here.
    """

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None,
            log_interval=10000,
            nb_max_episode_steps=None, episode_averaging_length=10,
            success_threshold=None,
            stopping_patience=None, min_nb_steps=500, single_cycle=True):

        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(
                action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        for cb in callbacks:
            if isinstance(cb, MyFileLogger):
                save_path = cb.filepath
                folder_index = save_path.index("training_history.json")
                weights_file = os.path.join(save_path[:folder_index],
                                            "dqn_weights.h5f")

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [MyTrainEpisodeLogger(interval=log_interval)]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = MyCallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        episode_num_errors = None
        did_abort = False

        # ------ Early stopping and reporting averages ------------------
        #
        # It would be ideal to do this via a callback, but returning flags from callbacks seems tricky. Eish!
        # So, we automatically include early stopping here in the fit method.
        # NB: We have hardcoded in something which is probably not ideal to hard code, but I just want it
        # to work, and can fix things and make them nicer/more flexible at a later stage!
        #
        # --------------------------------------------------------------

        if not single_cycle:

            recent_episode_lifetimes = deque([], episode_averaging_length)
            episode_lifetimes_rolling_avg = 0
            best_rolling_avg = 0
            best_episode = 0
            time_since_best = 0


        elif single_cycle:

            recent_episode_wins = deque([], episode_averaging_length)
            best_rolling_avg = 0
            best_episode = 0
            time_since_best = 0
            rolling_win_fraction = 0

        stop_training = False
        has_succeeded = False
        stopped_improving = False

        try:
            while self.step < nb_steps and not stop_training:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    # print("Episode Step:", episode_step)
                    # print("hidden state: ")
                    # print(env.hidden_state)
                    # print("Board State: ")
                    # print(observation)
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                        nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(
                                observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn(
                                'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(
                                    observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # print("Episode Step:", episode_step)
                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                if hasattr(env, "legal_actions"):
                    legal_actions = list(env.legal_actions)
                    action = self.forward(observation, legal_actions)
                    # print("legal actions: ", legal_actions)
                    # print("chosen action: ", action)
                else:
                    action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(
                            observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                # print("new hidden state: ")
                # print(env.hidden_state)
                # print("new board state: ")
                # print(observation)
                # print("reward: ", r, "episode reward: ", episode_reward)
                # print("done: ", done)

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.

                    action = self.forward(observation)
                    self.backward(0., terminal=False)

                    # Now we want to work out the recent averages, this will go into early stopping

                    if not single_cycle:

                        recent_episode_lifetimes.append(env.lifetime)
                        episode_lifetimes_rolling_avg = np.mean(
                            recent_episode_lifetimes)

                        if episode_lifetimes_rolling_avg > best_rolling_avg:
                            best_rolling_avg = episode_lifetimes_rolling_avg
                            best_episode = episode
                            time_since_best = 0
                        else:
                            time_since_best = episode - best_episode

                        if episode_lifetimes_rolling_avg > success_threshold:
                            stop_training = True
                            has_succeeded = True

                        if self.step > min_nb_steps and time_since_best > stopping_patience:
                            stop_training = True
                            stopped_improving = True

                    else:

                        if episode_reward == 1:
                            recent_episode_wins.append(1)
                        else:
                            recent_episode_wins.append(0)

                        num_wins = np.sum(recent_episode_wins)
                        rolling_win_fraction = num_wins / episode_averaging_length

                        if rolling_win_fraction > best_rolling_avg:
                            best_rolling_avg = rolling_win_fraction
                            best_episode = episode
                            time_since_best = 0

                            # Here I need to add something to save the net - I'm worried this will make things really slow while its improving, because it will be saving every time
                            # For a long time. Eish!
                            if self.step > min_nb_steps:
                                self.save_weights(weights_file, overwrite=True)

                        else:
                            time_since_best = episode - best_episode

                        if rolling_win_fraction > success_threshold:
                            stop_training = True
                            has_succeeded = True

                        if self.step > min_nb_steps and time_since_best > stopping_patience:
                            stop_training = True
                            stopped_improving = True

                    # This episode is finished, report and reset.

                    if not single_cycle:
                        episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                            'episode_lifetimes_rolling_avg': episode_lifetimes_rolling_avg,
                            'best_rolling_avg': best_rolling_avg,
                            'best_episode': best_episode,
                            'time_since_best': time_since_best,
                            'has_succeeded': has_succeeded,
                            'stopped_improving': stopped_improving
                        }

                    else:
                        episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                            'rolling_win_fraction': rolling_win_fraction,
                            'best_rolling_fraction': best_rolling_avg,
                            'best_episode': best_episode,
                            'time_since_best': time_since_best,
                            'has_succeeded': has_succeeded,
                            'stopped_improving': stopped_improving
                        }

                    callbacks.on_episode_end(episode, episode_logs,
                                             single_cycle)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True

        if not single_cycle:
            callbacks.on_train_end(logs={'did_abort': did_abort,
                                         'has_succeeded': has_succeeded,
                                         'stopped_improving': stopped_improving,
                                         'episode_lifetimes_rolling_avg': episode_lifetimes_rolling_avg,
                                         'step': self.step
                                         }, single_cycle=single_cycle)

        else:
            callbacks.on_train_end(logs={'did_abort': did_abort,
                                         'has_succeeded': has_succeeded,
                                         'stopped_improving': stopped_improving,
                                         'rolling_win_fraction': rolling_win_fraction,
                                         'step': self.step
                                         }, single_cycle=single_cycle)

        self._on_train_end()

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None,
             visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0,
             start_step_policy=None, verbose=1,
             episode_averaging_length=200, interval=100, single_cycle=True):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(
                action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [MyTestLogger(interval=interval)]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = MyCallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()

        if not single_cycle:

            recent_episode_lifetimes = []
            episode_lifetimes_rolling_avg = 0
            best_rolling_avg = 0
            best_episode = 0
            time_since_best = 0

        else:

            recent_episode_wins = []
            best_rolling_avg = 0
            best_episode = 0
            time_since_best = 0
            rolling_win_fraction = 0

        stop_training = False
        has_succeeded = False
        stopped_improving = False

        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())

            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn(
                        'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                            nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                if hasattr(env, "legal_actions"):
                    legal_actions = list(env.legal_actions)
                    action = self.forward(observation, legal_actions)
                else:
                    action = self.forward(observation)

                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            if not single_cycle:
                recent_episode_lifetimes.append(env.lifetime)
                episode_lifetimes_rolling_avg = np.mean(
                    recent_episode_lifetimes)

            else:

                if episode_reward == 1:
                    recent_episode_wins.append(1)
                else:
                    recent_episode_wins.append(0)

                num_wins = np.sum(recent_episode_wins)
                rolling_win_fraction = num_wins / len(recent_episode_wins)

            if not single_cycle:
                episode_logs = {
                    'episode_reward': episode_reward,
                    'nb_episode_steps': episode_step,
                    'episode_lifetime': env.lifetime,
                    'episode_lifetimes_rolling_avg': episode_lifetimes_rolling_avg}

            else:
                episode_logs = {
                    'episode_reward': episode_reward,
                    'nb_episode_steps': episode_step,
                    'rolling_win_fraction': rolling_win_fraction}

            callbacks.on_episode_end(episode, episode_logs, single_cycle)

        callbacks.on_train_end()
        self._on_test_end()

        return history