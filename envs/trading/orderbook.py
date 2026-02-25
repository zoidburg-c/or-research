from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple
import itertools


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


_order_counter = itertools.count()


@dataclass
class Order:
    agent: str
    side: Side
    price: float | None  # None = market order
    quantity: int
    order_id: int = field(default_factory=lambda: next(_order_counter))
    timestamp: int = 0


class Trade(NamedTuple):
    buyer: str
    seller: str
    price: float
    quantity: int


class LimitOrderBook:
    """Simplified limit order book with price-time priority."""

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self._bids: dict[float, list[Order]] = defaultdict(list)
        self._asks: dict[float, list[Order]] = defaultdict(list)
        self._orders: dict[int, Order] = {}
        self._timestamp = 0

    def add_order(self, order: Order) -> list[Trade]:
        self._timestamp += 1
        order.timestamp = self._timestamp
        trades: list[Trade] = []

        if order.side == Side.BUY:
            trades = self._match_buy(order)
        else:
            trades = self._match_sell(order)

        if order.quantity > 0 and order.price is not None:
            book = self._bids if order.side == Side.BUY else self._asks
            book[order.price].append(order)
            self._orders[order.order_id] = order

        return trades

    def _match_buy(self, order: Order) -> list[Trade]:
        trades = []
        while order.quantity > 0 and self._asks:
            best_ask_price = min(self._asks.keys())
            if order.price is not None and order.price < best_ask_price:
                break
            trades.extend(self._fill_against(order, self._asks, best_ask_price, Side.BUY))
        return trades

    def _match_sell(self, order: Order) -> list[Trade]:
        trades = []
        while order.quantity > 0 and self._bids:
            best_bid_price = max(self._bids.keys())
            if order.price is not None and order.price > best_bid_price:
                break
            trades.extend(self._fill_against(order, self._bids, best_bid_price, Side.SELL))
        return trades

    def _fill_against(
        self,
        aggressor: Order,
        book_side: dict[float, list[Order]],
        price_level: float,
        aggressor_side: Side,
    ) -> list[Trade]:
        trades = []
        queue = book_side[price_level]
        while aggressor.quantity > 0 and queue:
            resting = queue[0]
            fill_qty = min(aggressor.quantity, resting.quantity)

            buyer = aggressor.agent if aggressor_side == Side.BUY else resting.agent
            seller = resting.agent if aggressor_side == Side.BUY else aggressor.agent

            trades.append(Trade(buyer=buyer, seller=seller, price=price_level, quantity=fill_qty))

            aggressor.quantity -= fill_qty
            resting.quantity -= fill_qty

            if resting.quantity == 0:
                queue.pop(0)
                self._orders.pop(resting.order_id, None)

        if not queue:
            del book_side[price_level]

        return trades

    def cancel_order(self, order_id: int) -> bool:
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        book = self._bids if order.side == Side.BUY else self._asks
        if order.price in book:
            queue = book[order.price]
            queue[:] = [o for o in queue if o.order_id != order_id]
            if not queue:
                del book[order.price]
        return True

    def best_bid(self) -> float | None:
        return max(self._bids.keys()) if self._bids else None

    def best_ask(self) -> float | None:
        return min(self._asks.keys()) if self._asks else None

    def mid_price(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, levels: int = 5) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        bids = sorted(self._bids.keys(), reverse=True)[:levels]
        asks = sorted(self._asks.keys())[:levels]
        bid_depth = [(p, sum(o.quantity for o in self._bids[p])) for p in bids]
        ask_depth = [(p, sum(o.quantity for o in self._asks[p])) for p in asks]
        return bid_depth, ask_depth

    def get_agent_orders(self, agent: str) -> list[Order]:
        return [o for o in self._orders.values() if o.agent == agent]
