import numpy as np

# I am not sure where the best place to put this is, will ask Aleksi

def lattice_search(N, d):
    m = np.ceil(np.log2(N)).astype(int)

    # ----------------------------------------------------------------------
    # Set up rhovector
    # ----------------------------------------------------------------------
    bits = np.zeros((N, m), dtype=int)
    for i in range(N):
        # 2*bitget(i,1:m) in MATLAB
        bits[i, :] = 2 * np.array([((i >> j) & 1) for j in range(m)], dtype=int)

    cumsumbits = np.cumsum(bits, axis=0)        # N x m
    rhovector = np.dot((1.0 / np.arange(1, N + 1)), cumsumbits)  # 1 x m

    rhovectorNx1 = np.zeros((2**m - 1, 1))
    rIdx1 = 0
    for r in range(m, 0, -1):
        rIdx2 = rIdx1 + 2**(r - 1) - 1
        rhovectorNx1[rIdx1:rIdx2 + 1, 0] = rhovector[r - 1]
        rIdx1 = rIdx2 + 1

    # ----------------------------------------------------------------------
    # Get ordering of the search space
    # ----------------------------------------------------------------------
    gR = np.ones(2**(m - 2), dtype=int)
    intMod = 2**m
    for idx in range(1, 2**(m - 2)):
        temp = (gR[idx - 1] * 5) % intMod
        gR[idx] = min(intMod - temp, temp)

    gRows = np.ones(2**(m - 1), dtype=int)
    gRows[-1] = 0
    rowVects = np.ones(2**m - 1, dtype=int)
    gStrtIdx = 0
    vStrtIdx = 0

    for l in range(m, 1, -1):
        gEndIdx = gStrtIdx + 2**(l - 2) - 1
        vEndIdx = vStrtIdx + 2**(l - 1) - 1

        gRow = np.ones(2**(l - 2), dtype=int)
        intMod = 2**l
        for idx in range(1, 2**(l - 2)):
            temp = (gRow[idx - 1] * 5) % intMod
            gRow[idx] = min(intMod - temp, temp)

        gRows[gStrtIdx:gEndIdx + 1] = gRow
        rowV = np.concatenate(([1], np.flip(gRow[1:])))
        doubled = np.concatenate((rowV, rowV))
        rowVects[vStrtIdx:vEndIdx + 1] = 2**(m - l) * doubled

        gStrtIdx = gEndIdx + 1
        vStrtIdx = vEndIdx + 1

    rowVects[-1] = 2**(m - 1)

    # ----------------------------------------------------------------------
    # Set up prodV
    # ----------------------------------------------------------------------
    prodV = np.ones((2**m - 1, 1))
    prodV = prodV * rhovectorNx1

    # Initial 1D case
    rowV = rowVects / 2**m
    rowV = 1 + (rowV * (rowV - 1) + 1 / 6)
    prodV = prodV * rowV[:, None]

    # Set up k0
    k0 = 7 / 6

    # ----------------------------------------------------------------------
    # Begin search
    # ----------------------------------------------------------------------
    h = np.ones(d, dtype=int)

    for hComp in range(2, d + 1):
        WSSD = np.zeros(2**(m - 2))

        gamma = 1 / (hComp**2)
        omega = lambda x: 1 + gamma * (x * (x - 1) + 1 / 6)

        k0 = k0 * (1 + gamma / 6)

        curIdx2 = 0
        prodIdx1 = 0
        for l in range(m, 1, -1):
            nextIdx2 = curIdx2 + 2**(l - 2) - 1
            prodIdx2 = prodIdx1 + 2**(l - 2) - 1

            curRow = gRows[curIdx2:nextIdx2 + 1]
            col = curRow / 2**l
            fftCol = omega(col)

            pCol = prodV[prodIdx1:prodIdx2 + 1, 0]

            wVector = 2 * np.fft.ifft(np.fft.fft(fftCol) * np.fft.fft(pCol)).real
            numrep = 2**(m - l)
            WSSD = WSSD + np.tile(wVector, numrep)

            curIdx2 = nextIdx2 + 1
            prodIdx1 = prodIdx2 + 2**(l - 2) + 1

        WSSD = WSSD + omega(1 / 2) * prodV[-1, 0]
        WSSD = WSSD + N * k0 - N * (N + 1) / 2

        bestIdx = int(np.argmin(WSSD))
        bestWSSD = float(WSSD[bestIdx])
        newH = int(gR[bestIdx])

        # Avoid duplicates
        while newH in h:
            WSSD[bestIdx] = np.inf
            bestIdx = int(np.argmin(WSSD))
            bestWSSD = float(WSSD[bestIdx])
            newH = int(gR[bestIdx])

        h[hComp - 1] = newH

        rowV = (newH * rowVects) % 2**m
        rowV = rowV / 2**m
        rowV = omega(rowV)
        prodV = prodV * rowV[:, None]

    return h