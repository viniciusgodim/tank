import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt

N = 50
H = 1
reynolds = 1000
WD = 0.25
nIterations = 30
pRelax = 0.5
uRelax = 0.5
vRelax = 0.5
uBC = [0, 0, 0, 0]
vBC = [0, 0, 0, 0]


vRange = np.array((3 + N) * [False] + (N * [True] +
                                       2 * [False]) * (N - 1) + (N + 1) * [False])
uRange = np.array(
    (3 + N - 1) * [False] + ((N - 1) * [True] + 2 * [False]) * N + N * [False])
pRange = np.array((3 + N - 2) * [False] + ((N - 2) *
                                           [True] + 2 * [False]) * (N - 2) + (N - 1) * [False])

Xp = np.linspace(1 / (2 * N), 1 - 1 / (2 * N), N)
X = np.append(np.append(0, Xp), 1)

Yp = np.linspace(H - H / (2 * N), H / (2 * N), N)
Y = np.append(np.append(H, Yp), 0)

Rf = np.linspace(0, 1, N + 1)

Ar = 2 * Rf * H / N
Ar = np.tile(Ar, (N, 1))
ArP = Ar[:, range(1, N)]

Az = []
for i in range(len(Rf) - 1):
    Az.append(Rf[i + 1]**2 - Rf[i]**2)

Az = np.tile(Az, (N + 1, 1))
AzP = Az[range(1, N), :]

Vz = AzP * H / N
Vz = Vz.flatten()

d = N / reynolds

P = np.zeros((N, N))
Uface = -np.ones((N, N - 1))*d
Vface = -np.ones((N - 1, N))*d

uBCArray = [np.ones((1, N + 2)) * uBC[0], np.ones((N, 1)) *
            uBC[1], np.ones((N, 1)) * uBC[2], np.ones((1, N + 2)) * uBC[3]]

vBCArray = [np.ones((1, N)) * vBC[0], np.ones((N + 2, 1)) *
            vBC[1], np.ones((N + 2, 1)) * vBC[2], np.ones((1, N)) * vBC[3]]


def movingAverage(x):
    return np.convolve(x, np.ones(2) / 2, mode='valid')


Rfp = movingAverage(Rf)
ArUAvg = np.apply_along_axis(movingAverage, 1, Ar).flatten()
AzUAvg = np.apply_along_axis(movingAverage, 1, Az).flatten()
ArVAvg = np.apply_along_axis(movingAverage, 0, Ar).flatten()
AzVAvg = np.apply_along_axis(movingAverage, 0, Az).flatten()

ArAux = Ar[range(1, N - 1), :][:, range(1, N)]
AzAux = Az[range(1, N), :][:, range(1, N - 1)]

Azflat = AzAux.flatten()
Arflat = ArAux.flatten()


def hybridScheme(f, d):
    return np.array([max(f[i], d[i] + f[i] / 2, 0) for i in range(4)])


def upwindScheme(f, d):
    return np.array([d[i] + max(f[i], 0) for i in range(4)])


def uSolve(uFace, vFace, p):
    uNode = np.apply_along_axis(movingAverage, 1, uFaceHorizontalBounded)
    vCorner = np.apply_along_axis(movingAverage, 1, vFaceVerticalBounded)
    uNodeLinear = uNode.flatten()
    vCornerLinear = vCorner.flatten()
    A = np.zeros((N * (N - 1), (N + 2) * (N + 1)))
    for i in range(N * (N - 1)):
        l = i // (N - 1)
        area = np.array([AzUAvg[i], ArUAvg[i + l],
                         ArUAvg[i + l + 1], AzUAvg[i + N - 1]])
        D = area * d
        radius = Rf[i % (N - 1) + 1]
        F = np.array([-vCornerLinear[i], uNodeLinear[i + l], -
                      uNodeLinear[i + 1 + l], vCornerLinear[i + N - 1]])
        wBC = np.ones(4)
        if i % (N - 1) == 0:
            wBC[1] = uBC[1]
        if (i + 1) % (N - 1) == 0:
            wBC[2] = uBC[2]
        if i + 1 > (N - 1) * (N - 1):
            wall = False
            if radius > WD:
                D[3] *= 2
                wBC[3] = uBC[3]
                wall = True
        a = upwindScheme(F, D)
        ap = -(sum(a) - sum(F)) / uRelax
        aW = wBC * a
        if i + 1 <= N - 1:
            aW = [0, 0, 0, -1]
            ap = 1
        if i + 1 > (N - 1) * (N - 1) and not wall:
            aW = [-1, 0, 0, 0]
            ap = 1
        A[i, i + 1 + 2 * l] = aW[0]
        A[i, i + 1 + 2 * l + N] = aW[1]
        A[i, i + 2 + 2 * l + N] = ap
        A[i, i + 3 + 2 * l + N] = aW[2]
        A[i, i + 3 + 2 * l + 2 * N] = aW[3]
    AU = A[:, uRange]
    BU = A[:, ~uRange]
    BU = -np.sum(BU, axis=1)
    ApU = -np.diagonal(AU)
    ApU = np.reshape(ApU, (-1, N - 1))
    BU = BU + (np.diff(p, axis=1) * ArP).flatten() - \
        (1 - uRelax) * (ApU * uFace).flatten()
    for i in range(N * (N - 1)):
        radius = Rf[i % (N - 1) + 1]
        wall = False
        if radius > WD:
            wall = True
        if i + 1 <= N - 1:
            BU[i] = 0
        if i + 1 > (N - 1) * (N - 1) and not wall:
            BU[i] = 0
    uLinear = solve(AU, BU)
    uFaceNew = np.reshape(uLinear, (-1, N - 1))
    return [uFaceNew, ApU]


def vSolve(uFace, vFace, p):
    uCorner = np.apply_along_axis(movingAverage, 0, uFaceHorizontalBounded)
    vNode = np.apply_along_axis(movingAverage, 0, vFaceVerticalBounded)
    uCornerLinear = uCorner.flatten()
    vNodeLinear = vNode.flatten()
    A = np.zeros((N * (N - 1), (N + 2) * (N + 1)))
    for i in range(N * (N - 1)):
        l = i // N
        area = np.array([AzVAvg[i], ArVAvg[i + l],
                         ArVAvg[i + l + 1], AzVAvg[i + N]])

        D = area * d
        radius = Rfp[i % N]
        F = np.array([-vNodeLinear[i], uCornerLinear[i + l], -
                      uCornerLinear[i + 1 + l], vNodeLinear[i + N]])
        wBC = np.ones(4)
        if (i + 1) % N == 0:
            D[2] *= 2
            wBC[2] = vBC[2]
        if (i + 1) > (N - 2) * N:
            wall = False
            if radius > WD:
                wBC[3] = vBC[3]
                wall = True
        a = upwindScheme(F, D)
        ap = -(sum(a) - sum(F)) / vRelax
        aW = wBC * a
        if i + 1 <= N:
            aW = [0, 0, 0, -1]
            ap = 1
        if i % N == 0:
            aW = [0, 0, -1, 0]
            ap = 1
        if (i + 1) > (N - 2) * N and not wall:
            aW = [-1, 0, 0, 0]
            ap = 1
        A[i, i + 1 + 2 * l] = aW[0]
        A[i, i + 1 + 2 * l + N + 1] = aW[1]
        A[i, i + 2 + 2 * l + N + 1] = ap
        A[i, i + 3 + 2 * l + N + 1] = aW[2]
        A[i, i + 5 + 2 * l + 2 * N] = aW[3]
    AV = A[:, vRange]
    BV = A[:, ~vRange]
    BV = -np.sum(BV, axis=1)
    ApV = -np.diagonal(AV)
    ApV = np.reshape(ApV, (-1, N))
    BV = BV + Vz - (np.diff(p, axis=0) * AzP).flatten() - \
        (1 - vRelax) * (ApV * vFace).flatten()
    for i in range(N * (N - 1)):
        radius = Rfp[i % N]
        wall = False
        if radius > WD:
            wall = True
        if i + 1 <= N:
            BV[i] = 0
        if i % N == 0:
            BV[i] = 0
        if (i + 1) > (N - 2) * N and not wall:
            BV[i] = 0
    vLinear = solve(AV, BV)
    vFaceNew = np.reshape(vLinear, (-1, N))
    return [vFaceNew, ApV]


def pSolve(apU, apV, uFace, vFace):
    apU = apU / ArP
    apV = apV / AzP
    apU = apU[range(1, N - 1), :].flatten()
    apV = apV[:, range(1, N - 1)].flatten()
    uFaceStripped = uFace[range(1, N - 1), :]
    vFaceStripped = vFace[:, range(1, N - 1)]
    q = 1 / apU
    r = 1 / apV
    A = np.zeros(((N - 2)**2, N**2))
    for i in range((N - 2)**2):
        l = i // (N - 2)
        area = np.array([Azflat[i], Arflat[i + l],
                         Arflat[i + 1 + l], Azflat[i + N - 2]])
        radius = Rfp[i % (N - 2) + 1]
        a = np.array([r[i], q[i + l], q[i + 1 + l], r[i + N - 2]])
        a = a * area
        ap = -sum(a)
        if i + 1 <= N - 2:
            a[0] = 0
        if i % (N - 2) == 0:
            ap = 1
            a = [0, 0, -1, 0]
        if (i + 1) % (N - 2) == 0:
            ap = 1
            a = [0, -1, 0, 0]
        if (i + 1) > (N - 2) * (N - 3):
            if radius <= WD:
                a[3] = 0
            else:
                ap = 1
                a = [-1, 0, 0, 0]
        A[i, i + 1 + 2 * l] = a[0]
        A[i, i + 1 + 2 * l + N - 1] = a[1]
        A[i, i + 2 + 2 * l + N - 1] = ap
        A[i, i + 3 + 2 * l + N - 1] = a[2]
        A[i, i + 1 + 2 * l + 2 * N] = a[3]
    AP = A[:, pRange]
    BP = A[:, ~pRange]
    BP = -np.sum(BP, axis=1)
    dU = np.diff(uFaceStripped * ArAux, axis=1).flatten()
    dV = -np.diff(vFaceStripped * AzAux, axis=0).flatten()
    BP = BP + dU + dV
    for i in range((N - 2)**2):
        radius = Rfp[i % (N - 2) + 1]
        wall = False
        if radius > WD:
            wall = True
        if i % (N - 2) == 0:
            BP[i] = 0
        if (i + 1) % (N - 2) == 0:
            BP[i] = 0
        if (i + 1) > (N - 2) * (N - 3) and wall:
            BP[i] = 0
    P = solve(AP, BP)
    P = np.reshape(P, (-1, N - 2))
    P = np.vstack((np.zeros(len(P)), P, np.zeros(len(P))))
    P = np.hstack((P[:, 0].reshape(-1, 1), P, P[:, -1].reshape(-1, 1)))
    for i in range(P.shape[1]):
        radius = Rfp[i]
        if radius > WD:
            P[-1, i] = P[-2, i]
    P[0, 0] = (P[0, 1] + P[1, 0]) / 2
    P[-1, 0] = (P[-2, 0] + P[-1, 1]) / 2
    P[0, -1] = (P[1, -1] + P[0, -2]) / 2
    P[-1, -1] = (P[-2, -1] + P[-1, -2]) / 2
    return P


for i in range(nIterations):
    print(i)
    uFaceHorizontalBounded = np.hstack((uBCArray[1], Uface, uBCArray[2])) * Ar
    vFaceVerticalBounded = np.vstack(([Vface[0, :]], Vface, vBCArray[3])) * Az
    for i in range(vFaceVerticalBounded.shape[1]):
        radius = Rfp[i]
        if radius <= WD:
            vFaceVerticalBounded[-1, i] = vFaceVerticalBounded[-2, i]
    USolution = uSolve(Uface, Vface, P)
    UfaceNew = USolution[0]
    UAp = USolution[1]
    VSolution = vSolve(Uface, Vface, P)
    VfaceNew = VSolution[0]
    VAp = VSolution[1]
    pline = pSolve(UAp, VAp, UfaceNew, VfaceNew)
    P = P + pRelax * pline
    Uface = UfaceNew - ArP * np.diff(pline, axis=1) / UAp
    Vface = VfaceNew + AzP * np.diff(pline, axis=0) / VAp
    print(Vface)

Uplot = np.hstack((uBCArray[1], Uface, uBCArray[2]))
Uplot = np.apply_along_axis(movingAverage, 1, Uplot)
Uplot = np.hstack((uBCArray[1], Uplot, uBCArray[2]))
Uplot = np.vstack(([Uplot[0, :]], Uplot, [Uplot[-1, :]]))

for i in range(Uplot.shape[1] - 2):
    radius = Rfp[i]
    if radius > WD:
        Uplot[-1, i + 1] = uBC[3]

Vplot = np.vstack(([Vface[0, :]], Vface, vBCArray[3]))
Vplot = np.apply_along_axis(movingAverage, 0, Vplot)
Vplot = np.vstack(([Vplot[0, :]], Vplot, [Vplot[-1, :]]))
Vplot = np.hstack(([[i] for i in Vplot[:, 0]], Vplot, vBCArray[2]))

for i in range(Vplot.shape[1]-2):
    radius = Rfp[i]
    if radius > WD:
        Vplot[-1, i + 1] = vBC[3]

Uplot = np.hstack((np.fliplr(Uplot)[:, :-1], -Uplot))
Vplot = np.hstack((np.fliplr(Vplot)[:, :-1], Vplot))

Xflip = -np.flip(X)[:-1]
X = np.append(Xflip, X)

fig, ax = plt.subplots()
ax.quiver(X, Y, Uplot, Vplot)
plt.show()

Xpflip = -np.flip(Xp)[:-1]
Xp = np.append(Xpflip, Xp)
P = np.hstack((np.fliplr(P)[:, :-1], P))

XP, YP = np.meshgrid(Xp, Yp)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(XP, YP, P)
plt.show()
