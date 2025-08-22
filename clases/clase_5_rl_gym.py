"""Reinforcement Learning: Deep Q-Learning based trader"""
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_BACKEND'] = 'torch'

from enum import Enum
from typing import NamedTuple
import random
from collections import deque

import numpy as np
import gymnasium as gym
import keras


type Book = list[float]  # Pyhton 3.12 required


class Action(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class State(NamedTuple):
    hist_prices: np.ndarray
    trade_price: float
    book: Book
    realized_pnl: float


class Decision(NamedTuple):
    action: Action
    conviction: bool


class TradingDesk(gym.Env):
    """Clase para modelar el Trading Desk usando gymnasium.

    Parámetros:
        prices -- array con precios del activo a tradear.
        render_mode -- output para el reporte de estado.
    """
    metadata = {'render_modes': ('human',)}

    def __init__(self, prices: np.ndarray, render_mode: str | None='human') -> None:
        super().__init__()

        self.action_space = gym.spaces.Discrete(n=len(Action))
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                dtype=np.float32)
        self.reward_range = (-1.0, 1.0)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self._prices = prices

        self.reset()

    def reset(self, *, seed: int | None=None, options: dict | None=None) -> tuple[State, dict]:
        super().reset(seed=seed)  # We need the following line to seed self.np_random

        self.terminated = False
        self.truncated = False
        self._current_step = 0

        self.realized_pnl = 0.
        self.book = []

        trade_price = self._prices[0]
        prices = self._prices[:0]
        next_state = State(prices, trade_price, self.book, self.realized_pnl)

        return next_state, {'step_num': self._current_step}

    def step(self, action: Action) -> tuple[State, float, bool, bool, dict]:
        trade_price = self._prices[self._current_step]

        match action:
            case Action.Hold:
                reward = -1  # Se lo castiga por no hacer nada

            case Action.Buy:
                self.book.append(trade_price)
                if len(self.book) <= 5:
                    reward = 0  # No se le da un castigo por comprar pero tampoco premio
                else:
                    reward = -1  # Se lo castiga por estar muy largo

            case Action.Sell:
                if len(self.book) > 0:
                    oldest_price = self.book.pop(0)  # FIFO
                    profit = trade_price - oldest_price
                    self.realized_pnl += profit
                    reward = 1 if profit > 0 else -1  # Sólo se premia si el trade es positivo
                else:
                    reward = -1  # Se lo castiga por shortearse

        self._current_step += 1

        if self._current_step + 1 == len(self._prices):
            self.terminated = True

        trade_price = self._prices[self._current_step]
        hist_prices = self._prices[:self._current_step]
        next_state = State(hist_prices, trade_price, self.book, self.realized_pnl)

        return next_state, reward, self.terminated, self.truncated, {'step_num': self._current_step}

    def render(self, mode='human') -> str:
        assert mode is None or mode in self.metadata['render_modes']

        return f'Step : {self._current_step} | Realized P&L: {self.realized_pnl:,.2f}'

    @classmethod
    def from_file(cls, prices_file_name: str, history_window: int | None=None, **kwargs) -> 'TradingDesk':
        with open(prices_file_name) as csvfile:
            lines = csvfile.read().splitlines()
            prices = np.array([float(line.split(',')[1]) for line in lines[1:]])

        if history_window:
            _prices = prices[-history_window-1:]
        else:
            _prices = prices

        return cls(prices=_prices, **kwargs)


class Trader:
    """Clase para modelar un Trader.

    Parámetros:
        strategy_window -- cantidad de ruedas a utilizar en la estrategia de trading.
        min_experience -- cantidad de días mínima para que comience a aprender.
    """
    def __init__(self, strategy_window: int, min_experience: int) -> None:
        self.state_size = strategy_window
        self.memory = deque(maxlen=1000)  # Jornadas que recuerda
        self.min_experience = min_experience

        self.gamma = 0.95  # Tasa de descuento / impaciencia
        self.epsilon = 1.0  # Tasa de 'curiosidad' / Exploration Rate
        self.epsilon_min = 0.01  # Mínimo umbral de 'curiosidad'
        self.epsilon_decay = 0.95  # Ritmo de disminusión de la 'curiosidad'

        # Trading rule
        input_tensor = keras.Input(shape=(self.state_size,))
        hl1 = keras.layers.Dense(64, activation='relu')(input_tensor)
        hl2 = keras.layers.Dense(32, activation='relu')(hl1)
        hl3 = keras.layers.Dense(8, activation='relu')(hl2)

        output_tensor = keras.layers.Dense(len(Action), activation='linear')(hl3)

        model = keras.Model(input_tensor, output_tensor)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))

        self.model = model

    def act(self, state: State) -> Decision:
        if random.random() <= self.epsilon:
            return Decision(Action(random.randrange(len(Action))), False)

        tgt_prices = state.hist_prices[-self.state_size-1:]
        returns = tgt_prices[1:] / tgt_prices[:-1] - 1.

        if len(returns) < self.state_size:
            return Decision(Action(random.randrange(len(Action))), False)

        views_strength = self.model.predict(np.array([returns]), verbose=0)
        return Decision(Action(np.argmax(views_strength[0])), True)

    def learn(self, current_state: State, action: Action, reward: float,
              next_state: State) -> None:
        c_prices = current_state.hist_prices[-self.state_size-1:]
        c_returns = c_prices[1:] / c_prices[:-1] - 1.

        n_prices = next_state.hist_prices[-self.state_size-1:]
        n_returns = n_prices[1:] / n_prices[:-1] - 1.

        c_state = np.array([c_returns])
        n_state = np.array([n_returns])

        if len(c_returns) < self.state_size:
            return

        self.memory.append((c_state, action, reward, n_state))

        ml = len(self.memory)

        if ml < self.min_experience:
            return

        mini_batch = (self.memory[i] for i in range(ml - self.min_experience + 1, ml))

        # Algoritmo de Q-Learning
        for _c_state, _action, _reward, _n_state in mini_batch:
            target = _reward + self.gamma * np.amax(self.model.predict(_n_state, verbose=0)[0])

            target_f = self.model.predict(_c_state, verbose=0)
            target_f[0][_action.value] = target

            self.model.fit(_c_state, target_f, epochs=1, verbose=0)

        # Al incrementar el conocimiento bajo el umbral de 'curiosidad'
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    EPISODES = 2
    HISTORY_WINDOW = 100
    STRATEGY_WINDOW = 10
    MIN_EXPERIENCE = 10

    td = TradingDesk.from_file('./data/spx.csv', HISTORY_WINDOW)

    trader = Trader(STRATEGY_WINDOW, MIN_EXPERIENCE)

    for e in range(EPISODES):
        print(38 * '=')
        print(f'Episode {e + 1} of {EPISODES}'.center(38))
        print(38 * '-')
        print(f'{"Día":^5}|{"Decisión":^10}|{"Reward":^8}|{"Criterio":^12}')

        state, info = td.reset()

        while True:
            decision = trader.act(state)

            new_state, reward, terminated, truncated, info = td.step(decision.action)

            print(f'{info["step_num"]:^5}|{decision.action.name:^10}|{reward:^8}|{"Convicción" if decision.conviction else "Azar":^12}')

            if terminated:
                break

            trader.learn(state, decision.action, reward, new_state)
            state = new_state

        total_profit = new_state.realized_pnl
        if new_state.book:
            total_profit += sum(new_state.trade_price - bp for bp in new_state.book)

        print(38 * '=')
        print(f'Ganancia Total: {total_profit:,.2f}')
        print(38 * '=', '\n')
