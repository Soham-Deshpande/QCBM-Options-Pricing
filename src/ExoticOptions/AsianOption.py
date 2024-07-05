
class AsianOption():
    def __init__(self, option_type, strike_price, average_price=None):
        self.option_type = option_type
        self.average_price = average_price
        self.strike_price = strike_price

    def calculate_payoff(self):
        if self.average_price is None:
            raise ValueError("Average price not set.")
        if self.option_type == "call":
            return max(self.average_price - self.strike_price, 0)
        elif self.option_type == "put":
            return max(self.strike_price - self.average_price, 0)
        else:
            raise ValueError("Invalid option type.")