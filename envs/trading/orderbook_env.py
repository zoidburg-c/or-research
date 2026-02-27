"""Multi-agent order book trading environment (PettingZoo ParallelEnv)."""
from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from envs.trading.orderbook import LimitOrderBook, Order, Side


NUM_ACTIONS = 14
ORDER_QUANTITY = 10


class TradingEnv(ParallelEnv):
    metadata = {"name": "trading_v0"}

    def __init__(
        self,
        num_agents: int = 4,
        agent_types: list[str] | None = None,
        num_objectives: int = 3,
        orderbook_depth: int = 10,
        tick_size: float = 0.01,
        initial_cash: float = 100_000.0,
        initial_price: float = 100.0,
        episode_length: int = 1000,
        price_history_len: int = 50,
    ):
        if agent_types is None:
            agent_types = ["market_maker"] * num_agents
        if len(agent_types) != num_agents:
            raise ValueError(
                f"agent_types length ({len(agent_types)}) must match num_agents ({num_agents})"
            )

        self.num_objectives = num_objectives
        self.orderbook_depth = orderbook_depth
        self.tick_size = tick_size
        self.initial_cash = initial_cash
        self.initial_price = initial_price
        self.episode_length = episode_length
        self.price_history_len = price_history_len

        self.possible_agents = [f"trader_{i}" for i in range(num_agents)]
        self.agent_types = {self.possible_agents[i]: agent_types[i] for i in range(num_agents)}
        self.agents: list[str] = []

        self._cash: dict[str, float] = {}
        self._inventory: dict[str, int] = {}
        self._peak_value: dict[str, float] = {}

        self._book: LimitOrderBook | None = None
        self._price_history: list[float] = []
        self._step_count = 0
        self._rng: np.random.Generator | None = None
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(num_objectives,), dtype=np.float64)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        size = 5 + self.price_history_len + self.orderbook_depth * 4
        return Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Discrete:
        return Discrete(NUM_ACTIONS)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self._rng = np.random.default_rng(seed)
        self._book = LimitOrderBook(tick_size=self.tick_size)
        self._step_count = 0
        self._price_history = [self.initial_price] * self.price_history_len
        self.agents = self.possible_agents[:]

        for a in self.agents:
            self._cash[a] = self.initial_cash
            self._inventory[a] = 0
            self._peak_value[a] = self.initial_cash

        # Seed initial book with liquidity
        for offset in range(1, 6):
            self._book.add_order(
                Order(
                    agent="__init__",
                    side=Side.BUY,
                    price=self.initial_price - offset * self.tick_size,
                    quantity=100,
                )
            )
            self._book.add_order(
                Order(
                    agent="__init__",
                    side=Side.SELL,
                    price=self.initial_price + offset * self.tick_size,
                    quantity=100,
                )
            )

        obs = {a: self._get_obs(a) for a in self.agents}
        infos: dict[str, dict] = {
            a: self._get_info(a, np.zeros(self.num_objectives)) for a in self.agents
        }
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        self._step_count += 1
        agent_trades: dict[str, list] = {a: [] for a in self.agents}

        for agent, action in actions.items():
            trades = self._process_action(agent, action)
            for t in trades:
                if t.buyer == agent:
                    agent_trades[agent].append(t)
                if t.seller == agent:
                    agent_trades[agent].append(t)

        # Update price history
        mid = self._book.mid_price()
        if mid is not None:
            self._price_history.append(mid)
        else:
            self._price_history.append(self._price_history[-1])
        if len(self._price_history) > self.price_history_len:
            self._price_history = self._price_history[-self.price_history_len:]

        # Compute vector rewards
        vec_rewards = {}
        for a in self.agents:
            vec_rewards[a] = self._compute_reward(a, agent_trades[a])

        team_reward = np.sum(list(vec_rewards.values()), axis=0)

        done = self._step_count >= self.episode_length

        if done:
            obs = {a: self._get_obs(a) for a in self.agents}
            rewards = {a: 0.0 for a in self.agents}
            terminations = {a: True for a in self.agents}
            truncations = {a: False for a in self.agents}
            infos = {
                a: self._get_info(a, vec_rewards[a], team_reward) for a in self.agents
            }
            self.agents = []
        else:
            obs = {a: self._get_obs(a) for a in self.agents}
            rewards = {a: 0.0 for a in self.agents}
            terminations = {a: False for a in self.agents}
            truncations = {a: False for a in self.agents}
            infos = {
                a: self._get_info(a, vec_rewards[a], team_reward) for a in self.agents
            }

        return obs, rewards, terminations, truncations, infos

    def _process_action(self, agent: str, action: int) -> list:
        mid = self._book.mid_price() or self.initial_price
        trades: list = []

        if action == 0:
            pass
        elif 1 <= action <= 5:
            offset = action
            price = round(mid - offset * self.tick_size, 8)
            trades = self._book.add_order(
                Order(agent=agent, side=Side.BUY, price=price, quantity=ORDER_QUANTITY)
            )
        elif 6 <= action <= 10:
            offset = action - 5
            price = round(mid + offset * self.tick_size, 8)
            trades = self._book.add_order(
                Order(agent=agent, side=Side.SELL, price=price, quantity=ORDER_QUANTITY)
            )
        elif action == 11:
            trades = self._book.add_order(
                Order(agent=agent, side=Side.BUY, price=None, quantity=ORDER_QUANTITY)
            )
        elif action == 12:
            trades = self._book.add_order(
                Order(agent=agent, side=Side.SELL, price=None, quantity=ORDER_QUANTITY)
            )
        elif action == 13:
            for order in self._book.get_agent_orders(agent):
                self._book.cancel_order(order.order_id)

        # Update cash and inventory for all trades involving this agent
        for t in trades:
            if t.buyer == agent:
                self._cash[agent] -= t.price * t.quantity
                self._inventory[agent] += t.quantity
            if t.seller == agent:
                self._cash[agent] += t.price * t.quantity
                self._inventory[agent] -= t.quantity

        return trades

    def _compute_reward(self, agent: str, trades: list) -> np.ndarray:
        mid = self._book.mid_price() or self._price_history[-1]

        # Objective 1: P&L
        portfolio_value = self._cash[agent] + self._inventory[agent] * mid
        pnl = portfolio_value - self.initial_cash

        # Objective 2: Risk (negative drawdown)
        if portfolio_value > self._peak_value[agent]:
            self._peak_value[agent] = portfolio_value
        drawdown = (self._peak_value[agent] - portfolio_value) / max(
            self._peak_value[agent], 1.0
        )
        risk = -drawdown

        # Objective 3: Liquidity cost (negative slippage)
        slippage = 0.0
        for t in trades:
            slippage += abs(t.price - mid) * t.quantity
        liquidity_cost = -slippage

        return np.array([pnl, risk, liquidity_cost], dtype=np.float64)

    def _get_obs(self, agent: str) -> np.ndarray:
        mid = self._book.mid_price() or self._price_history[-1]
        spread = self._book.spread() or 0.0
        unrealized = self._inventory[agent] * mid

        ph = np.array(
            self._price_history[-self.price_history_len:], dtype=np.float32
        )
        if len(ph) < self.price_history_len:
            ph = np.pad(
                ph, (self.price_history_len - len(ph), 0), constant_values=mid
            )

        bids, asks = self._book.depth(levels=self.orderbook_depth)
        bid_flat = np.zeros(self.orderbook_depth * 2, dtype=np.float32)
        ask_flat = np.zeros(self.orderbook_depth * 2, dtype=np.float32)
        for i, (p, q) in enumerate(bids):
            bid_flat[i * 2] = p
            bid_flat[i * 2 + 1] = q
        for i, (p, q) in enumerate(asks):
            ask_flat[i * 2] = p
            ask_flat[i * 2 + 1] = q

        obs = np.concatenate(
            [
                np.array(
                    [mid, spread, self._cash[agent], self._inventory[agent], unrealized],
                    dtype=np.float32,
                ),
                ph,
                bid_flat,
                ask_flat,
            ]
        )
        return obs

    def _get_info(
        self,
        agent: str,
        vec_reward: np.ndarray,
        team_reward: np.ndarray | None = None,
    ) -> dict:
        info: dict[str, Any] = {"vec_reward": vec_reward}
        if team_reward is not None:
            info["team_vec_reward"] = team_reward
        else:
            info["team_vec_reward"] = vec_reward.copy()
        return info
