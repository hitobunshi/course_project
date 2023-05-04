from collections import deque
from interval import interval
from typing import Callable


def hansen(F: Callable[[interval], interval], X: interval, tol=1e-6) -> tuple[interval, float]:
    Y: interval = X
    y: float = F(Y)[0].inf
    c: interval = Y.midpoint
    f_midpoint: float = F(c)[0].sup
    L = deque([(Y, y)])
    
    res = (Y, y)

    while True:
        V_1 = interval[Y[0].inf, (Y[0].sup + Y[0].inf) / 2]
        V_2 = interval[(Y[0].sup + Y[0].inf) / 2, Y[0].sup]

        v_1 = F(V_1)[0].inf
        v_2 = F(V_2)[0].inf

        L.append((V_1, v_1))
        L.append((V_2, v_2))

        res = min(L, key=lambda x: x[1])

        L = deque(filter(lambda x: x[1] <= f_midpoint, L))

        if sum([x[0][0].sup - x[0][0].inf for x in L]) < tol:
            break

        Y, y = L.popleft()
        c = Y.midpoint
        f_midpoint = min(f_midpoint, F(c)[0].sup)

    return res
