from math import *
import matplotlib.pyplot as plt
import numpy as np

############### Functions ###############
def mach_def(M, g, p2, p1):
    p1pt = (1+((g-1)/2)*M**2)**(-g/(g-1))
    p1p2 = p1 / p2
    p2pt = (1/p1p2)*p1pt
    M2 = (((((p2pt)**((1-g)/g))-1))*(2/(g-1)))**0.5
    return M2

def prandtl(M, g):
    v = (((g+1)/(g-1))**0.5) * atan ((((g-1)/(g+1))*((M**2) -1))**0.5) - atan(((M**2) -1)**0.5)
    return v

def prandtldiff(M, g):
    a = (g+1)/(g-1)
    dvdm = (M/((a**0.5) * ((((M**2) - 1)/a) + 1) * (((M**2) - 1)/a)**0.5)) - (1/(M*(M**2 - 1)**0.5))
    return dvdm

def mach_angle(M):
    miu = np.arcsin(1/M)
    return miu

def nu_to_m(v, g):
    x0 = 2
    e = 1
    i = 0
    while e > 1e-10 and i < 100:
        xi = x0 - (prandtl(x0, g) - v) / prandtldiff(x0, g)
        e = abs(xi - x0)
        x0 = xi
        i += 1
    return x0

def newalpha(v0, v1, phi0, miu):
    phi = v0 - v1 - phi0
    alpha = miu - phi
    return alpha

def ccw(A, B, C):
    return np.array((C[:, 1]-A[:, 1])*(B[:, 0]-A[:, 0]) > (B[:, 1]-A[:, 1])*(C[:, 0]-A[:, 0]))

def intersect(A, B, C, D):
    return np.logical_and(ccw(A, C, D) != ccw(B, C, D), ccw(A, B, C) != ccw(A, B, D))

def relpress(M, g):
    ppt = (1 + (((g-1) / 2)*(M**2)))**(-g / (g-1))
    return ppt
############### Given constants ###############
N = 15
Me = 2
gam = 1.4
pa = 1
pe = 2*pa
phie = 0
h = 1

############### Defining (some) arrays ###############
phi = np.empty(N+1)
v = np.empty(N+1)
M = np.empty(N+1)
miu = np.empty(N+1)
alpha = np.empty(N+1)

############### Points arrays definition ###############
vpoint = np.empty((N**2, N**2))
vpoint[:] = np.nan
phipoint = np.empty((N**2, N**2))
phipoint[:] = np.nan
Mpoint = np.empty((N**2, N**2))
Mpoint[:] = np.nan
miupoint = np.empty((N**2, N**2))
miupoint[:] = np.nan
xpoint = np.empty((N**2, N**2))
xpoint[:] = np.nan
ypoint = np.empty((N**2, N**2))
ypoint[:] = np.nan
linex = np.empty((N**2, 2))
linex[:] = np.nan
liney = np.empty((N**2, 2))
liney[:] = np.nan
xb = np.array([])
yb = np.array([])
Mb = np.array([])
gc = np.array([])
phigc = np.array([])
Mgc = np.array([])

############### Properties of first expansion fan ###############
Ma = mach_def(Me, gam, pa, pe)
miue = mach_angle(Me)
ve = prandtl(Me, gam)
va = prandtl(Ma, gam)
phia = va - ve
dphi = (phia - phie) / (N-1)

for i in range(0, N):
    phi[i] = dphi*(i)
    v[i] = ve - phie + phi[i]
    M[i] = nu_to_m(v[i], gam)
    miu[i] = mach_angle(M[i])
    alpha[i] = miu[i] - phi[i]

############### Properties of first point, corner ###############
for i in range(0, N):
   vpoint[i][0] = v[i]
   phipoint[i][0] = phi[i]
   Mpoint[i][0] = nu_to_m(vpoint[i][0], gam)
   miupoint[i][0] = mach_angle(Mpoint[i][0])
   xpoint[i][0] = 0
   ypoint[i][0] = h
   
############### First reflection, centerline ###############
for j in range(1, N+1):
    for i in range(j-1, N):
        if i+1 == j:
            vpoint[i][j] = vpoint[i][j-1] + phipoint[i][j-1]
            phipoint[i][j] = 0
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            b = 0.5 * (phipoint[i][j-1] - miupoint[i][j-1] + phipoint[i][j] - miupoint[i][j])
            ypoint[i][j] = 0
            xpoint[i][j] = (xpoint[i][j-1] * tan(b) - ypoint[i][j-1]) / (tan(b))
            
        else:
            vpoint[i][j] = 0.5*(vpoint[i-1][j] - phipoint[i-1][j]) + 0.5*(vpoint[i][j-1] + phipoint[i][j-1])
            phipoint[i][j] = 0.5*(phipoint[i][j-1] + phipoint[i-1][j]) + 0.5*(vpoint[i][j-1] - vpoint[i-1][j])
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            a = 0.5 * (phipoint[i-1][j] + miupoint[i-1][j] + phipoint[i][j] + miupoint[i][j])
            b = 0.5 * (phipoint[i][j-1] - miupoint[i][j-1] + phipoint[i][j] - miupoint[i][j])
            xpoint[i][j] = ((xpoint[i][j-1] * tan(b)) - (xpoint[i-1][j] * tan(a)) + ypoint[i-1][j] - ypoint[i][j-1])/(tan(b) - tan(a))
            ypoint[i][j] = (xpoint[i][j] - xpoint[i-1][j])*(tan(a)) + ypoint[i-1][j]

############### Second reflection, jet boundary ###############
for i in range(N, (2*N)):
    for j in range(1, N+1):
        if j == i + 1 - N:
            vpoint[i][j] = prandtl(Ma, gam)
            phipoint[i][j] = - vpoint[i-1][j] + phipoint[i-1][j] + vpoint[i][j] 
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            b = phipoint[i-1][j-1]
            a = 0.5 * (phipoint[i-1][j] + miupoint[i-1][j] + phipoint[i][j] + miupoint[i][j])
            xpoint[i][j] = ((xpoint[i-1][j-1] * tan(b)) - (xpoint[i-1][j] * tan(a)) + ypoint[i-1][j] - ypoint[i-1][j-1])/(tan(b) - tan(a))
            ypoint[i][j] = (xpoint[i][j] - xpoint[i-1][j])*(tan(a)) + ypoint[i-1][j]
        else:
            vpoint[i][j] = 0.5*(vpoint[i][j-1] + vpoint[i-1][j]) + 0.5*(phipoint[i][j-1] - phipoint[i-1][j])
            phipoint[i][j] = 0.5*(phipoint[i][j-1] + phipoint[i-1][j]) + 0.5*(vpoint[i][j-1] - vpoint[i-1][j])
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            a = 0.5 * (phipoint[i-1][j] + miupoint[i-1][j] + phipoint[i][j] + miupoint[i][j])
            b = 0.5 * (phipoint[i][j-1] - miupoint[i][j-1] + phipoint[i][j] - miupoint[i][j])
            xpoint[i][j] = ((xpoint[i][j-1] * tan(b)) - (xpoint[i-1][j] * tan(a)) + ypoint[i-1][j] - ypoint[i][j-1])/(tan(b) - tan(a))
            ypoint[i][j] = (xpoint[i][j] - xpoint[i-1][j])*(tan(a)) + ypoint[i-1][j]

############### Third reflection, centerline ###############            
for j in range(N+1, (2*N)+1):
    for i in range(N, 2*N):
        if i+1 == j:
            vpoint[i][j] = vpoint[i][j-1] + phipoint[i][j-1]
            phipoint[i][j] = 0
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            b = 0.5 * (phipoint[i][j-1] - miupoint[i][j-1] + phipoint[i][j] - miupoint[i][j])
            ypoint[i][j] = 0
            xpoint[i][j] = (xpoint[i][j-1] * tan(b) - ypoint[i][j-1]) / (tan(b))
        else:
            vpoint[i][j] = 0.5*(vpoint[i-1][j] - phipoint[i-1][j]) + 0.5*(vpoint[i][j-1] + phipoint[i][j-1])
            phipoint[i][j] = 0.5*(phipoint[i][j-1] + phipoint[i-1][j]) + 0.5*(vpoint[i][j-1] - vpoint[i-1][j])
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            a = 0.5 * (phipoint[i-1][j] + miupoint[i-1][j] + phipoint[i][j] + miupoint[i][j])
            b = 0.5 * (phipoint[i][j-1] - miupoint[i][j-1] + phipoint[i][j] - miupoint[i][j])
            xpoint[i][j] = ((xpoint[i][j-1] * tan(b)) - (xpoint[i-1][j] * tan(a)) + ypoint[i-1][j] - ypoint[i][j-1])/(tan(b) - tan(a))
            ypoint[i][j] = (xpoint[i][j] - xpoint[i-1][j])*(tan(a)) + ypoint[i-1][j]

############### Fourth reflection, jet boundary,  ###############
for i in range(2*N, (2*N)+1):
    for j in range(N+1, 2*N + 1):
        if j == i + 1 - N:
            vpoint[i][j] = prandtl(Ma, gam)
            phipoint[i][j] = - vpoint[i-1][j] + phipoint[i-1][j] + vpoint[i][j] 
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            b = phipoint[i-1][j-1]
            a = 0.5 * (phipoint[i-1][j] + miupoint[i-1][j] + phipoint[i][j] + miupoint[i][j])
            xpoint[i][j] = ((xpoint[i-1][j-1] * tan(b)) - (xpoint[i-1][j] * tan(a)) + ypoint[i-1][j] - ypoint[i-1][j-1])/(tan(b) - tan(a))
            ypoint[i][j] = (xpoint[i][j] - xpoint[i-1][j])*(tan(a)) + ypoint[i-1][j]
        else:
            vpoint[i][j] = 0.5*(vpoint[i][j-1] + vpoint[i-1][j]) + 0.5*(phipoint[i][j-1] - phipoint[i-1][j])
            phipoint[i][j] = 0.5*(phipoint[i][j-1] + phipoint[i-1][j]) + 0.5*(vpoint[i][j-1] - vpoint[i-1][j])
            Mpoint[i][j] = nu_to_m(vpoint[i][j], gam)
            miupoint[i][j] = mach_angle(Mpoint[i][j])
            a = 0.5 * (phipoint[i-1][j] + miupoint[i-1][j] + phipoint[i][j] + miupoint[i][j])
            b = 0.5 * (phipoint[i][j-1] - miupoint[i][j-1] + phipoint[i][j] - miupoint[i][j])
            xpoint[i][j] = ((xpoint[i][j-1] * tan(b)) - (xpoint[i-1][j] * tan(a)) + ypoint[i-1][j] - ypoint[i][j-1])/(tan(b) - tan(a))
            ypoint[i][j] = (xpoint[i][j] - xpoint[i-1][j])*(tan(a)) + ypoint[i-1][j]

############### Plotting -ve & +ve characteristics and points ###############
pd = 500
for i in range(0, N*3):
    xg = xpoint[i][np.logical_not(np.isnan(xpoint[i]))]
    yg = ypoint[i][np.logical_not(np.isnan(ypoint[i]))]
    Mg = Mpoint[i][np.logical_not(np.isnan(Mpoint[i]))]
    phig = phipoint[i][np.logical_not(np.isnan(phipoint[i]))]
    plt.plot(xg, yg, color = 'green')
    if xg.shape[0] != 0:
        for j in range(0, xg.shape[0]-1):
            a = np.array([])
            xc = np.linspace(xg[j], xg[j+1], pd)
            yc = np.linspace(yg[j], yg[j+1], pd)
            phic = phig[j]
            Mk = Mg[j]
            a = np.hstack((a, [xc[0], yc[0]]))
            a = np.hstack((a, [xc[-1], yc[-1]]))
            if np.shape(gc) == (0,):
                gc = a
                phigc = phic
                Mgc = Mk
            else:
                gc = np.vstack((gc, a))
                phigc = np.vstack((phigc, phic))
                Mgc = np.vstack((Mgc, Mk))
                
            miugc = mach_angle(Mgc)
            alphagc = phigc - miugc
            Mc = np.linspace(Mg[j], Mg[j+1], pd)
            xb = np.hstack((xc[1:-1], xb))
            yb = np.hstack((yc[1:-1], yb))
            Mb = np.hstack((Mc[1:-1], Mb))
   
xt = np.transpose(xpoint)
yt = np.transpose(ypoint)
Mt = np.transpose(Mpoint)
phit = np.transpose(phipoint)
for i in range(1, N*3):
    xj = xt[i][np.logical_not(np.isnan(xt[i]))]
    yj = yt[i][np.logical_not(np.isnan(yt[i]))]
    Mj = Mt[i][np.logical_not(np.isnan(Mt[i]))]
    phig = phit[i][np.logical_not(np.isnan(phit[i]))]
    plt.plot(xj, yj, color = 'red')
    if xj.shape[0] != 0:
        for j in range(0, xj.shape[0]-1):
            a = np.array([])
            xc = np.linspace(xj[j], xj[j+1], pd)
            yc = np.linspace(yj[j], yj[j+1], pd)
            phic = phig[j]
            Mk = Mj[j]
            a = np.hstack((a, [xc[0], yc[0]]))
            a = np.hstack((a, [xc[-1], yc[-1]]))
            if np.shape(gc) == (0,):
                gc = a
                phigc = phic
                Mgc = Mk
            else:
                gc = np.vstack((gc, a))
                phigc = np.vstack((phigc, phic))
                Mgc = np.vstack((Mgc, Mk))
            miugc = mach_angle(Mgc)
            alphagc = phigc - miugc
            Mc = np.linspace(Mj[j], Mj[j+1], pd)
            xb = np.hstack((xc[1:-1], xb))
            yb = np.hstack((yc[1:-1], yb))
            Mb = np.hstack((Mc[1:-1], Mb))
gc0, gc1 = np.hsplit(gc, 2)
gc0x, gc0y = np.hsplit(gc0, 2)
gc1x, gc1y = np.hsplit(gc1, 2)
f = len(gc)

############### Calculating and drawing streamline ###############
xinit = 0
yinit = h / 2
phip = 0
Mpinit = Mgc[0]
Mpnew = Mgc[0]
dl = 0.01
xend = 15
strm = True
while strm:
    init = np.array([[xinit, yinit]])
    sinit = np.full((f, 2), np.array([[xinit, yinit]]))
    sinitx, sinity = np.hsplit(sinit, 2)
    xinit += float(dl*cos(phip))
    yinit += float(dl*sin(phip))
    fin = np.array([[xinit, yinit]])
    sfin = np.full((f, 2), fin)
    sfinx, sfiny = np.hsplit(sfin, 2)
    l = intersect(sinit, sfin, gc0, gc1)
    a = (gc1y-gc0y) / (gc1x-gc0x)
    xint = (sinity - gc0y + (gc0x * a) - (sinitx * np.tan(phip))) / (a - np.tan(phip))
    yint = sinity + (xint - sinitx)*np.tan(phip)
    t = np.nanargmin(np.sqrt((xint - sfinx)**2 + (yint - sfiny)**2))
    if np.any(l) and init[0][0] <= xint[t]:
        xinit = float(xint[t])
        yinit = float(yint[t])
        phip = float(phigc[t])
        Mpnew = float(Mgc[t])
        fin = np.array([[xinit, yinit]])
        gc0x = np.delete(gc0x, t, axis = 0)
        gc0y = np.delete(gc0y, t, axis = 0)
        gc1x = np.delete(gc1x, t, axis = 0)
        gc1y = np.delete(gc1y, t, axis = 0)
        phigc = np.delete(phigc, t, axis = 0)
        alphagc = np.delete(alphagc, t, axis = 0)
        gc0 = np.delete(gc0, t, axis = 0)
        gc1 = np.delete(gc1, t, axis = 0)
        Mgc = np.delete(Mgc, t, axis = 0)
        f = len(gc0x)
    elif xinit >= xend:
        strm = False
        continue
    plt.figure(1)
    plt.plot((np.array([fin[0][0],init[0][0]])), np.array([fin[0][1],init[0][1]]), color = 'black', zorder = 100)
    plt.figure(2)
    pnew = relpress(Mpnew, gam)
    pinit = relpress(Mpinit, gam)
    plt.plot((np.array([fin[0][0],init[0][0]])), (pnew, pinit), color = 'black')
    Mpinit = Mpnew
    if xinit >= xend:
        strm = False
plt.figure(1) 
 
############### Plotting jet boundary and centerline ###############
linex[0] = xpoint[0][0], xpoint[N][1]
liney[0] = ypoint[0][0], ypoint[N][1]
for i in range(1, N+1):
    linex[i] = xpoint[i+N-1][i], xpoint[i+N+1][i+1]
    liney[i] = ypoint[i+N-1][i], ypoint[i+N+1][i+1]
linex[N+1] = xpoint[2*N-1][N], xpoint[2*N][N+1]
liney[N+1] = ypoint[2*N-1][N], ypoint[2*N][N+1]
xct = [0,xpoint[2*N][N+1]]
yct = [0, 0]

############### Plotting entrance points ###############
ye = np.arange(0, h, 0.1)
xe = np.zeros(np.size(ye))
Mex = np.full(np.size(ye), Me)

############### Plotting ###############
for i in range(0, N):
    Mpoint[i][0] = Mpoint[N-1][0]
xpoint1 = xpoint[np.logical_not(np.isnan(xpoint))]
ypoint1 = ypoint[np.logical_not(np.isnan(ypoint))]
Mpoint1 = Mpoint[np.logical_not(np.isnan(Mpoint))]
ypoint2 = np.hstack((ye, ypoint1))
xpoint2 = np.hstack((xe, xpoint1))
Mpoint2 = np.hstack((Mex, Mpoint1))
ypoint2 = np.hstack((yb, ypoint2))
xpoint2 = np.hstack((xb, xpoint2))
Mpoint2 = np.hstack((Mb, Mpoint2))

############### Plotting interpolated values used as a background map ###############
plt.tricontourf(xpoint2, ypoint2, Mpoint2, levels=50, cmap='cool') 
plt.colorbar(label=f'Mach number')

############### Plotting part 2 ###############
plt.xlabel('x [m]')
plt.ylabel('y [m]')
linex1 = linex[np.logical_not(np.isnan(linex))]
liney1 = liney[np.logical_not(np.isnan(liney))]
#p1 = plt.scatter(xpoint1, ypoint1, c = Mpoint1, cmap = 'winter') # point plotting
plt.plot(linex1, liney1, '-', color = 'blue') # jet boundary plotting
plt.plot(xct, yct, '-.', color = 'black') # centerline plotting
plt.axis('equal')
#plt.colorbar(p1, label = 'Mach number')