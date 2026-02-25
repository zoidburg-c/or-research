import pytest
from envs.trading.orderbook import LimitOrderBook, Order, Side


class TestOrderBook:
    def setup_method(self):
        self.book = LimitOrderBook(tick_size=0.01)

    def test_add_limit_buy(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=100.0, quantity=10))
        assert self.book.best_bid() == 100.0
        assert self.book.best_ask() is None

    def test_add_limit_sell(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=101.0, quantity=10))
        assert self.book.best_ask() == 101.0
        assert self.book.best_bid() is None

    def test_crossing_order_executes(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=100.0, quantity=10))
        trades = self.book.add_order(Order(agent="b", side=Side.BUY, price=100.0, quantity=5))
        assert len(trades) == 1
        assert trades[0].price == 100.0
        assert trades[0].quantity == 5
        assert trades[0].buyer == "b"
        assert trades[0].seller == "a"
        assert self.book.best_ask() == 100.0

    def test_price_time_priority(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=100.0, quantity=5))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=100.0, quantity=5))
        trades = self.book.add_order(Order(agent="c", side=Side.BUY, price=100.0, quantity=5))
        assert len(trades) == 1
        assert trades[0].seller == "a"

    def test_market_buy(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=100.0, quantity=10))
        trades = self.book.add_order(Order(agent="b", side=Side.BUY, price=None, quantity=5))
        assert len(trades) == 1
        assert trades[0].price == 100.0

    def test_market_sell(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=100.0, quantity=10))
        trades = self.book.add_order(Order(agent="b", side=Side.SELL, price=None, quantity=5))
        assert len(trades) == 1
        assert trades[0].price == 100.0

    def test_cancel_order(self):
        order = Order(agent="a", side=Side.BUY, price=100.0, quantity=10)
        self.book.add_order(order)
        assert self.book.best_bid() == 100.0
        self.book.cancel_order(order.order_id)
        assert self.book.best_bid() is None

    def test_mid_price(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=99.0, quantity=10))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=101.0, quantity=10))
        assert self.book.mid_price() == pytest.approx(100.0)

    def test_spread(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=99.0, quantity=10))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=101.0, quantity=10))
        assert self.book.spread() == pytest.approx(2.0)

    def test_depth(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=99.0, quantity=10))
        self.book.add_order(Order(agent="a", side=Side.BUY, price=98.0, quantity=5))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=101.0, quantity=7))
        bids, asks = self.book.depth(levels=5)
        assert len(bids) == 2
        assert bids[0] == (99.0, 10)
        assert len(asks) == 1
