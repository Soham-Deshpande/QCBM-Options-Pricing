
from ExoticOptions.AsianOption import *

def main():
    ao = AsianOption("call", 100, 130)
    print(ao.calculate_payoff())

if __name__ == "__main__":
    main()
    