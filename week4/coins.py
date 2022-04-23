from typing import Tuple, List, Optional
from copy import copy

# Real coins from 1 cent to 2 euro
# COINS = [1, 2, 5, 10, 20, 50, 100, 200]

# With real set of coints we can exchange any amount, lets have also
# a set of coins which doesn't guarantee exchange of any amount
#
COINS = [3, 7]


def change(amount: int) -> Tuple[Optional[int], List[int]]:
    """Returns
    * the minimal amount of coins needed to exchange <amount of cents>
    * optional log how this exchange was done
    """
    dynamic_table: List[Optional[int]] = [None] * (amount + 1)
    # links L1 lists will be more memory efficient here
    log_table: List[List[int]] = [list()] * (amount + 1)
    dynamic_table[0] = 0
    log_table[0] = list()
    for current_amount in range(2, amount + 1):
        for coin in COINS:
            prev = current_amount - coin
            prev_minimal_exchange = dynamic_table[prev]
            if (prev < 0) or (prev_minimal_exchange is None):
                continue
            candidate = prev_minimal_exchange + 1
            current_minimal_exchange = dynamic_table[current_amount]
            if (current_minimal_exchange is None) or (
                candidate < current_minimal_exchange
            ):
                dynamic_table[current_amount] = candidate
                log_table[current_amount] = copy(log_table[prev])
                log_table[current_amount].append(coin)
    return dynamic_table[amount], log_table[amount]


if __name__ == "__main__":
    # keep the demo simple, no need to `pip install click`

    # minimal amount of coins to exchange 27 euro 54 cents
    # print(change(int(27.97 * 100)))
    print(change(int(1.17 * 100)))
    print(change(int(0.08 * 100)))
