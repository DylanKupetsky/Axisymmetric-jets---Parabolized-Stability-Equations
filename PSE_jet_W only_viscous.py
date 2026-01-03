
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from numpy import pi,cos,arange,ones,tile,dot,eye,diag
import scipy.linalg.lapack as la

def W(r):
    return (1/2)*(1 + np.tanh((1/4)*(1/.04)*(1/(r) - (r))))

def Wr(r):
    return (1/2)*(-1/r**2 - 1)*(1/(4*.04))*(1/(np.cosh((1/(4*.04))*(1/(r) - r))**2))

def W2(z,r):
    thetaRj = 1/(.03*z + .04)
    return (1/2)*(1 + np.tanh((1/4)*(thetaRj)*(1/r - r)))

def Wr2(z,r):
    thetaRj = 1/(.03*z + .04)
    return (1/2)*(-1/r**2 - 1)*(thetaRj/4)*(1/(np.cosh((1/4)*thetaRj*(1/r - r))**2))

r = np.linspace(0,6,num=200)
plt.plot(r,W(r),label='W(z=0,r)')
plt.plot(r,W2(3,r),label='W(z=3,r)')
plt.plot(r,W2(10,r),label='W(z=10,r)')
plt.legend()
plt.xlabel('r')
plt.show()

def cheb(N):
    '''Chebushev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    x      = cos(pi*arange(0,N+1)/N)
    if N%2 == 0:
        x[N//2] = 0.0 # only when N is even!
    c      = ones(N+1); c[0] = 2.0; c[N] = 2.0
    c      = c * (-1.0)**arange(0,N+1)
    c      = c.reshape(N+1,1)
    X      = tile(x.reshape(N+1,1), (1,N+1))
    dX     = X - X.T
    D      = dot(c, 1.0/c.T) / (dX+eye(N+1))
    D      = D - diag( D.sum(axis=1) )
    return D,x

def chebint(derivmatrix,y,ai,bi): #y is the func to integrate
    b = scipy.linalg.solve(derivmatrix,y)
    return b[bi] - b[ai]

def tensor2matrix(tensor): #Turns a matrix of matrices into a regular matrix
    r, c, mr, mc = tensor.shape
    new_dim = (r*mr,c*mc)
    return np.reshape(np.swapaxes(tensor,1,2), new_dim,order='C')

def LST(N): #Function that will generate matrices A and B that correspond to a spatial LST using the mean functions
    
    derivmatrix, xi = cheb(N)
    xi = np.array(xi)
    xi = (3/2)*xi + (3/2) + .001
    derivmatrix = (2/3)*derivmatrix
    Re = 5000
    omega = .6*np.pi
    zeromat = np.zeros((N + 1, N + 1))
    Wmat = zeromat.copy()
    Wrmat = zeromat.copy()
    for i, rval in enumerate(xi):
        Wmat[i][i] = W(rval)

    Wrmat = zeromat.copy()
    Wrarray = np.zeros(N + 1)
    for i, rval in enumerate(xi):
        Wrmat[i][i] = Wr(rval)
        # Wrarray[i] = Wr(rval)
    # plt.plot(xi,Wrarray)
        
    
    oneOverRmat = zeromat.copy()
    for i, rval in enumerate(xi):
        oneOverRmat[i][i] = 1/rval
        
    rMat = zeromat.copy()
    for i, rval in enumerate(xi):
        rMat[i][i] = rval
        
    oneOverRsMat = zeromat.copy()
    for i, rval in enumerate(xi):
        oneOverRsMat[i][i] = 1/rval**2
        
    
    oneOneA = derivmatrix + oneOverRmat



    oneTwoB = 1j*np.identity(N + 1)

    twoOneA = Wrmat
    twoTwoA = -1j*np.identity(N + 1)*omega #- (1/Re)*(np.matmul(oneOverRmat,derivmatrix) + np.matmul(derivmatrix,derivmatrix))

    twoTwoB = 1j*Wmat

    twoThreeB = 1j*np.identity(N + 1)
    twoTwoC = zeromat.copy() #(1/Re)*np.identity(N + 1)
    
    threeOneA = -1j*np.identity(N + 1)*omega #- (1/Re)*(-oneOverRsMat + np.matmul(oneOverRmat,derivmatrix) + np.matmul(derivmatrix,derivmatrix))
    threeThreeA = derivmatrix.copy()
    
    threeOneB = 1j*Wmat
    threeOneC = (1/Re)*np.identity(N + 1)
    
    A = np.array([[oneOneA, zeromat, zeromat],
                [twoOneA, twoTwoA, zeromat],
                [threeOneA, zeromat, threeThreeA]])

    B = np.array([[zeromat, oneTwoB, zeromat],
                [zeromat, twoTwoB, twoThreeB],
                [threeOneB, zeromat, zeromat]])
    
    C = np.array([[zeromat, zeromat, zeromat],
                [zeromat, twoTwoC, zeromat],
                [threeOneC, zeromat, zeromat]])

    # Minv = np.array([[invDtot, zeromat, zeromat],
                    # [zeromat, np.identity(N + 1), zeromat],
                    # [zeromat, zeromat, invDtot]])
 



    newA = tensor2matrix(A)
    print(la.zgecon(newA,1))
    newB = -tensor2matrix(B)
    newC = tensor2matrix(C)
    # Minv = tensor2matrix(Minv)
    # newA = np.matmul(Minv,newA)
    # newB = np.matmul(Minv,newB)

    newA[2*(N + 1) - 1] = np.zeros(3*(N + 1))
    newA[2*(N + 1) - 1,N + 1:2*(N + 1)] = derivmatrix[-1] #dw/dr(r=0) = 0
    newB[2*(N + 1) - 1] = np.zeros(3*(N + 1))
    newC[2*(N + 1) - 1] = np.zeros(3*(N + 1))
    
    newA[N + 1] = np.zeros(3*(N + 1))
    newA[N + 1][N + 1] = 1 #w(r=inf) = 0
    newB[N + 1] = np.zeros(3*(N + 1))
    newC[N + 1] = np.zeros(3*(N + 1))

    # newA[3*(N + 1) - 1] = np.zeros(3*(N + 1))
    # newA[3*(N + 1) - 1,2*(N + 1):3*(N + 1)] = derivmatrix[-1] 
    # newB[3*(N + 1) - 1] = np.zeros(3*(N + 1)) #dp/dr(r = 0) = 0

    newA[N + 1 - 1] = np.zeros(3*(N + 1))
    newA[N + 1 - 1, N + 1 - 1] = 1
    newB[N + 1 - 1] = np.zeros(3*(N + 1)) #enforce u(r = 0) = 0
    newC[N + 1 - 1] = np.zeros(3*(N + 1))

    newA[0] = np.zeros(3*(N + 1))
    newA[0][0] = 1
    newB[0] = np.zeros(3*(N + 1)) #enforce u(r = inf (4)) = 0
    newC[0] = np.zeros(3*(N + 1))
    
    # newA[N + 1] = np.zeros(3*(N + 1))
    # newA[N + 1][N + 1] = 1
    # newB[N + 1] = np.zeros(3*(N + 1)) #enforce w(r = inf (4)) = 0

    # newA[2*(N + 1)] = np.zeros(3*(N + 1))
    # newA[2*(N + 1)][2*(N + 1)] = 1
    # newB[2*(N + 1)] = np.zeros(3*(N + 1)) #enforce p(r = inf (4)) = 0
    print(np.shape(newA))
    print(np.shape(newB))


    print(la.zgecon(newA,1))

    scipy.io.savemat('Abeta{}c.mat'.format(N + 1), mdict={'arrA_beta{}c'.format(N + 1): newA})
    scipy.io.savemat('Bbeta{}c.mat'.format(N + 1), mdict={'arrB_beta{}c'.format(N + 1): newB})
    scipy.io.savemat('Cbeta{}c.mat'.format(N + 1), mdict={'arrC_beta{}c'.format(N + 1): newC})


Ns = [79,89,99]
for N in Ns:
    LST(N)

#%%

'''
The following script takes the matrices (A80,B80), (A90,B90), (A100,B100) and solves the generalized eigenvalue problem.
The script then filters the eigenvalues that are distant from each other by less than .5 and additionally have real parts between
0 and 1 and imaginary parts between -1 and 0. The value with the largest negative value is chosen. 
The eigenvalue and corresponding eigenfunctions are saved to be loaded again in the next code block.
We choose the eigenfunctions/eigenvalue corresponding to N = 79.
'''
import matlab.engine

eng = matlab.engine.start_matlab()

eng.GEP_script(nargout=0)

#%%
#disturbances are not symmetric but will be made symmetric following the PSE code
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
N = 99
Re = 5000
derivmatrix, xi = cheb(N)

# xi = np.zeros(N + 1)
# for j in range(N + 1):
    # xi[j] = np.cos(j*np.pi/N)
    

xi = np.array(xi)
xi = (3/2)*xi + (3/2) + .001
derivmatrix = (2/3)*derivmatrix
zeromat = np.zeros((N + 1, N + 1))

oneOverRmat = zeromat.copy()
for i, rval in enumerate(xi):
    oneOverRmat[i][i] = 1/rval
rMat = zeromat.copy()
for i, rval in enumerate(xi):
    rMat[i][i] = rval
oneOverRsMat = zeromat.copy()
for i, rval in enumerate(xi):
    oneOverRsMat[i][i] = 1/rval**2
    
plt.style.use('dark_background')
omega = .6*np.pi
beta = scipy.io.loadmat('beta.mat')
beta = beta['beta'][0][0]

# deltaZ = 1/np.real(beta) +.15
deltaZ = .2



z = np.linspace(0,12,num=int(12/deltaZ))
Nz = len(z)
all_eigvals = scipy.io.loadmat('Vtarget.mat')
print(all_eigvals)
all_eigvals = all_eigvals['Vtarget']
u0 = np.array(all_eigvals[0:N + 1])[:,0]
w0 = np.array(all_eigvals[N + 1:2*(N + 1)])[:,0]
p0 = np.array(all_eigvals[2*(N + 1):3*(N + 1)])[:,0]

# xiNeg = np.where(xi < 0)[0]
# xiPos = np.where(xi > 0)[0]
# u0[xiNeg] = -u0[xiPos[::-1]]
# w0[xiNeg] = w0[xiPos[::-1]] #Make solution symmetric
# p0[xiNeg] = p0[xiPos[::-1]]


plt.plot(xi,np.real(u0),label='Re')
plt.xlabel("r")
# plt.ylabel('Re(u0)')
# plt.show() #make seperate plots
plt.plot(xi,np.imag(u0),label='Im')
plt.xlabel("r")
plt.ylabel('u0')
plt.legend()
plt.show()

plt.plot(xi,np.real(w0),label='Re')
plt.xlabel("r")
# plt.ylabel('Re(w0)')
# plt.show()
plt.plot(xi,np.imag(w0),label='Im')
plt.xlabel("r")
plt.ylabel('w0')
plt.show()

plt.plot(xi,np.real(p0),label='Re')
plt.xlabel('r')
# plt.ylabel('Re(p0)')
# plt.show()
plt.plot(xi,np.imag(p0),label='Im')
plt.xlabel('r')
plt.ylabel('p0')
plt.show()

normu = np.max(np.sqrt(np.real(u0)**2 + np.imag(u0)**2))
numu = np.sqrt(np.real(u0)**2 + np.imag(u0)**2)
plt.plot(xi,numu/normu)
plt.xlabel('r')
plt.ylabel('|u0|/max(|u0|)')
plt.title('Normalized Radial perturbation amplitude for z/R_j = 0')
plt.show()

normu = np.max(np.sqrt(np.real(w0)**2 + np.imag(w0)**2))
numu = np.sqrt(np.real(w0)**2 + np.imag(w0)**2)
plt.plot(xi,numu/normu)
plt.xlabel('r')
plt.ylabel('|w0|/max(|w0|)')
plt.title('Normalized Axial perturbation amplitude for z/R_j = 0')
plt.show()

normu = np.max(np.sqrt(np.real(p0)**2 + np.imag(p0)**2))
numu = np.sqrt(np.real(p0)**2 + np.imag(p0)**2)
plt.plot(xi,numu/normu)
plt.xlabel('r')
plt.ylabel('|p0|/max(|p0|)')
plt.title('Normalized Pressure perturbation amplitude for z/R_j = 0')
plt.show()
all_eigvals = np.concatenate((u0,np.concatenate((w0,p0))))

all_eigvals = (10**(-1))*all_eigvals #Ensures disturbances are at least 10^-2 seperated from mean flow (10^0)
# %%

'''
Now we have the initial data and can proceed with the PSE
We will delete the previous definitions of W,Wr,Wz
'''
try:
    W
except NameError:
    print('W undefined')
else:
    del W
try:
    Wr
except NameError:
    print('Wr undefined')
else:
    del Wr
try:
    Wz
except NameError:
    print('Wz undefined')
else:
    del Wz

'''
Define the mean flow as a function of r, z. These are functions
we can define for all r and z, so we can go ahead and create the
functions in the form of matrices.
'''
# W = np.zeros((len(z),N+1))
# def W2(z,r):
#     thetaRj = 1/(.03*z + .04)
#     return (1/2)*(1 + np.tanh((1/4)*(thetaRj)*(1/r**2 - r**2)))

Wz = np.zeros((Nz,N + 1))
W = np.zeros((Nz,N + 1))
Wr = np.zeros((Nz,N + 1))
# def Wr2(z,r):
#     thetaRj = 1/(.03*z + .04)
#     return (-1/2)*(1/r**3 + r)*thetaRj*(1/(np.cosh((1/4)*thetaRj*(1/r**2 - r**2))**2))

def Wz2(z,r):
    thetaRj = 1/(.03*z + .04)
    thetaRjz = -.03/(.03*z + .04)**2
    return (thetaRjz/4)*(1/r - r)*(1/(np.cosh((thetaRj/4)*(1/r - r))))

for i, zval in enumerate(z):
    for j, xival in enumerate(xi):
        W[i][j] = W2(zval,xival)
        Wr[i][j] = Wr2(zval,xival)
        Wz[i][j] = Wz2(zval,xival)
# %%
def K(alpha,j): #j is the index of z
    global omega
    global W
    global derivmatrix
    Tmat = np.identity(N + 1,dtype=complex)
    for i, Wval in enumerate(W[j]):
        Tmat[i][i] = 1j*(alpha*Wval - omega) #j is imaginary number, not index
    return Tmat

# def G(Re):
#     return (1/Re)*(oneOverRsMat + np.matmul(oneOverRmat,derivmatrix) + np.matmul(derivmatrix,derivmatrix))

# def T(Re):
#     return (1/Re)*(np.matmul(oneOverRmat,derivmatrix) + np.matmul(derivmatrix,derivmatrix))

def L(alpha,j,Re):
    global omega
    global W
    global Wr
    global Wz
    zeromat = np.zeros((N + 1, N + 1),dtype = complex)
    Tm = K(alpha, j) - (1/Re)*(-(alpha**2)*np.identity(N + 1,dtype=complex) + np.matmul(derivmatrix,derivmatrix))
    Tm = Tm + (1/Re)*(oneOverRsMat)
    oneOne = Tm.copy()
    oneTwo = zeromat.copy()
    oneThree = derivmatrix.copy()
    twoOne = derivmatrix + oneOverRmat
    twoTwo = 1j*alpha*np.identity(N + 1,dtype = complex)
    twoThree = zeromat.copy()
    Wrvec = Wr[j]
    threeOne = np.identity(N + 1,dtype = complex)
    for i, Wrval in enumerate(Wrvec):
        threeOne[i][i] = Wrval
    Wzvec = Wz[j]
    # threeTwo = np.identity(N + 1,dtype = complex)
    Tm2 = K(alpha, j) - (1/Re)*(-(alpha**2)*np.identity(N + 1,dtype=complex) + np.matmul(derivmatrix,derivmatrix)) - (1/Re)*np.matmul(oneOverRmat,derivmatrix) #uncomment if using streamwise visc.
    # Tm = np.identity(N + 1, dtype = complex)
    for i, Wzval in enumerate(Wzvec):
        Tm2[i][i] = Tm2[i][i] + Wzval
    threeTwo = Tm2.copy()
    threeThree = 1j*alpha*np.identity(N + 1,dtype = complex)
    
    Lmat = np.array([[oneOne, oneTwo, oneThree],
                     [twoOne, twoTwo, twoThree],
                     [threeOne, threeTwo, threeThree]],dtype = complex)
    Lmat = tensor2matrix(Lmat)
    ### BOUNDARY CONDITIONS!! ###
    #on u#
    Lmat[N + 1 - 1] = np.zeros(3*(N + 1))
    Lmat[N + 1 - 1, N + 1 - 1] = 1 #u(r=0) = 0
    Lmat[0] = np.zeros(3*(N + 1))
    Lmat[0][0] = 1 #u(r=inf) = 0
    #on w#
    # Lmat[2*(N + 1) - 1] = np.zeros(3*(N + 1))
    # Lmat[2*(N + 1) - 1,N + 1:2*(N + 1)] = derivmatrix[-1] #dw/dr(r=0) = 0
    Lmat[N + 1] = np.zeros(3*(N + 1))
    Lmat[N + 1][N + 1] = 1 #w(r=inf) = 0
    #on p#
    # Lmat[3*(N + 1) - 1] = np.zeros(3*(N + 1))
    # Lmat[3*(N + 1) - 1,2*(N + 1):3*(N + 1)] = derivmatrix[-1] #dp/dr(r=0) = 0
    ### BOUNDARY CONDITIONS!! ###
    return Lmat

def M(j,alpha,Re):
    global W
    zeromat = np.zeros((N + 1, N + 1),dtype=complex)
    Wvec = W[j]
    oneOne = np.identity(N + 1,dtype=complex)
    secondterm = (1/Re)*2*1j*np.identity(N + 1,dtype=complex)*alpha
    for i, Wval in enumerate(Wvec):
        oneOne[i][i] = Wval - secondterm[i][i]
    oneTwo = zeromat.copy()
    oneThree = zeromat.copy()
    twoOne = zeromat.copy()
    twoTwo = np.identity(N + 1,dtype=complex)
    twoThree = zeromat.copy()
    threeOne = zeromat.copy()
    threeTwo = oneOne.copy() #zeromat.copy() if no 1/Re(w) terms #oneOne.copy() if including streamwise viscosity
    threeThree = zeromat.copy() #change to complex identity to include dp/dz
    Mmat = np.array([[oneOne, oneTwo, oneThree],
                     [twoOne, twoTwo, twoThree],
                     [threeOne, threeTwo, threeThree]],dtype=complex)
    Mmat = tensor2matrix(Mmat)
    ## BOUNDARY CONDITIONS!##
    #on u#
    Mmat[N + 1 - 1] = np.zeros(3*(N + 1)) #u(r=0) = 0
    Mmat[0] = np.zeros(3*(N + 1)) #u(r=inf) = 0
    #on w#
    # Mmat[2*(N + 1) - 1] = np.zeros(3*(N + 1)) #dw/dr(r=0) = 0
    Mmat[N + 1] = np.zeros(3*(N + 1)) #w(r = inf) = 0
    #on p#
    # Mmat[3*(N + 1) - 1] = np.zeros(3*(N + 1)) #dp/dr(r = 0) = 0
    ## BOUNDARY CONDITIONS!##
    return Mmat

def Nm(Re):
    zeromat = np.zeros((N + 1, N + 1),dtype=complex)
    oneOne = -(1j/Re)*np.identity(N + 1,dtype = complex)
    threeTwo = oneOne.copy() #oneOne.copy() if including viscosity along streamwise
    Nmat = np.array([[oneOne, zeromat, zeromat],
                     [zeromat, zeromat, zeromat],
                     [zeromat, threeTwo, zeromat]])
    Nmat = tensor2matrix(Nmat)
    ## BOUNDARY CONDITIONS! ##
    #on u#
    Nmat[N + 1 - 1] = np.zeros(3*(N + 1)) #u(r=0) = 0
    Nmat[0] = np.zeros(3*(N + 1)) #u(r=inf) = 0
    #on w#
    # Nmat[2*(N + 1) - 1] = np.zeros(3*(N + 1)) #dw/dr(r=0) = 0
    Nmat[N + 1] = np.zeros(3*(N + 1)) #w(r = inf) = 0
    #on p#
    # Nmat[3*(N + 1) - 1] = np.zeros(3*(N + 1))
    ## BOUNDARY CONDITIONS! ##
    return Nmat
# %%
rint = np.concatenate((xi,xi))
# rint = np.concatenate((rint,xi))

#To get an idea of how to code in general, get q_j+1^0 and alpha_j+1^1
#For this viscous case, we will not include p in the norm, only u and w
qprev = all_eigvals.copy()

A = deltaZ*L(alpha=beta,j=1,Re=Re) + M(j=1,alpha=beta,Re=Re) #For the first case, we let alpha_j+1^0 = alpha_j so the N matrix is not here
b = np.matmul(M(j=1,alpha=beta,Re=Re),qprev)

qnext = scipy.linalg.solve(A,b)

mag = np.real(qnext[0:2*(N + 1)])**2 + np.imag(qnext[0:2*(N + 1)])**2
denom = scipy.integrate.simpson(mag,x=rint)

top = np.real(qnext[0:2*(N + 1)])**2 + np.imag(qnext[0:2*(N + 1)])**2 - np.conjugate(qnext[0:2*(N + 1)])*qprev[0:2*(N + 1)]
#scipy.integrate.simpson only returns floats, so integration between Re and Im is done seperately
topRe = np.real(top)
topIm = np.imag(top)
num = scipy.integrate.simpson(topRe,x=rint) + 1j*scipy.integrate.simpson(topIm,x=rint)

intg = (1j/deltaZ)*num/denom
alphanew = beta - intg

relError = (np.real(alphanew - beta)**2 + np.imag(alphanew - beta)**2)/(np.real(alphanew)**2 + np.imag(alphanew)**2)

# %%
alphavec = [beta]
condvec = []
q = np.zeros((3*len(xi),len(z)),dtype=complex)
q[:,0] = all_eigvals
q[:,1] = qnext
for j in range(len(z) - 1):
    m = 0
    relError = 1 #this is here simply so that when j = j + 1, the while loop is activated. Otherwise we could calculate 
    while relError > 10**(-5):
        alphaold = alphanew.copy()
        if m == 0:
            print(alphaold - alphavec[j])
        A = deltaZ*L(alphaold,j=j + 1,Re=Re) + M(j = j + 1,alpha=alphaold,Re=Re) #+ (alphaold - alphavec[j])*Nm(Re=Re)
        b = np.matmul(M(j = j + 1,alpha=alphaold,Re=Re),q[:,j])
        qnext = scipy.linalg.solve(A,b)
        
        mag = np.real(qnext[0:2*(N + 1)])**2 + np.imag(qnext[0:2*(N + 1)])**2
        denom = scipy.integrate.simpson(mag,x=rint)

        top = np.real(qnext[0:2*(N + 1)])**2 + np.imag(qnext[0:2*(N + 1)])**2 - (np.real(qnext[0:2*(N + 1)]) - 1j*np.imag(qnext[0:2*(N + 1)]))*q[:,j][0:2*(N + 1)]
        topRe = np.real(top)
        topIm = np.imag(top)
        num = scipy.integrate.simpson(topRe,x=rint) + 1j*scipy.integrate.simpson(topIm,x=rint) 
        intg = (1j/deltaZ)*num/denom
        
        alphanew = alphaold - intg
        
        relError = (np.real(alphanew - alphaold)**2 + np.imag(alphanew - alphaold)**2)/(np.real(alphanew)**2 + np.imag(alphanew)**2)
        print('Current alpha = {}, m = {}, Relative error = {}'.format(alphanew,m,relError))
        m += 1
    print('Convergence complete')
    q[:,j + 1] = qnext
    alphavec.append(alphanew)
    condvec.append(np.linalg.cond(L(alphanew,j=j + 1,Re=Re)*deltaZ + M(j=j+1,alpha=alphanew,Re=Re) + (alphanew - alphavec[j])*Nm(Re=Re)))
# %%
plt.style.use('dark_background')
def η(index):
    global z
    global alphavec
    if index >= 1:
        s = 0
        for i in range(0,index):
            s += deltaZ*(10**(0))*(alphavec[i + 1] + alphavec[i])*.5
        return s
    else:
        return 0

ηvec = np.zeros(len(z),dtype = complex)
for zindex in range(len(z)):
    ηvec[zindex] = η(zindex) #Get η such that η_z = alpha(z)
    
plt.plot(z,np.real(ηvec),'.')
plt.xlabel('z')
plt.ylabel('Re(η(z))')
plt.show()
plt.plot(z,np.imag(ηvec),'.')
plt.xlabel('z')
plt.ylabel('Im(η(z))')
plt.show()

u = np.transpose(q[0:N + 1,:])
w = np.transpose(q[N + 1:2*(N + 1),:])
p = np.transpose(q[2*(N + 1):3*(N + 1),:])
uh = u.copy()
wh = w.copy()
ph = p.copy()


#Now that we solved the system for r > 0, we want to create
#graphs which are axisymmetric. For u, the solution
#will be its negative on the "negative r" side since positive u values on the
#"negative r" side should point to positive r values, which
#points in the direction we would normally think of as the
#negative x-direction. Hopefully that makes sense.
#w/p values will be symmetric across r = 0.

uflip = -u.copy()
wflip = w.copy()
pflip = p.copy()

uold = np.concatenate((uflip,np.flip(u,axis=1)),axis=1)
wold = np.concatenate((wflip,np.flip(w,axis=1)),axis=1) 
pold = np.concatenate((pflip,np.flip(p,axis=1)),axis=1)

unew = np.zeros((len(u),len(uold[0]) - 1))
wnew = np.zeros((len(w),len(wold[0]) - 1))
pnew = np.zeros((len(p),len(pold[0]) - 1))
for j, zitem in enumerate(unew):
    for i in range(len(zitem)):
        if i < N + 1:
            unew[j][i] = uold[j][i]
            wnew[j][i] = wold[j][i]
            pnew[j][i] = pold[j][i]
        if i >= N + 1:
            unew[j][i] = uold[j][i + 1]
            wnew[j][i] = wold[j][i + 1]
            pnew[j][i] = pold[j][i + 1]

u = unew.copy()
w = wnew.copy()
p = pnew.copy()

qp = np.zeros((3*2*len(xi) - 1,len(z)),dtype=complex)
up = np.zeros((2*len(xi) - 1,len(z)),dtype=complex)
wp = np.zeros((2*len(xi) - 1,len(z)),dtype=complex)
pp = np.zeros((2*len(xi) - 1,len(z)),dtype=complex)
    
for j in range(len(z)):
    # qp[:,j] = q[:,j]*np.exp(1j*ηvec[j])
    up[:,j] = unew[j]*np.exp(1j*ηvec[j])
    wp[:,j] = wnew[j]*np.exp(1j*ηvec[j]) #create q'
    pp[:,j] = pnew[j]*np.exp(1j*ηvec[j])

pp = np.transpose(pp)

xieven = ((3)/(np.pi))*np.arccos((2/3)*(xi - .001) - 1) #some graphs require evenly spaced points
xieven = np.concatenate((-np.flip(xieven),xieven))
xievencop = np.zeros(len(xieven) - 1)

for i in range(len(xievencop)):
    if i < N + 1:
        xievencop[i] = xieven[i]
    if i >= N + 1:
        xievencop[i] = xieven[i + 1]

xixi, zz = np.meshgrid(xievencop,z)

import re
fig, ax = plt.subplots(1,1)
cont = ax.contour(xixi,zz,p,levels=25)
h1,l1 = cont.legend_elements("p")
h1 = h1[::8]
l1 = l1[::8]
floats = []
for i in range(len(h1)):
    match = re.findall("[+-]?\d+\.\d+", l1[i])[0]
    floats.append(match)

l1newlist = []
for i in range(len(l1)):
    l1new = np.round(float(floats[i]),3)
    l1new = '$p = %.3f$'%l1new
    l1newlist.append(l1new)
    
ax.legend(h1,l1newlist)
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_title('p hat')
plt.show()

import re
fig, ax = plt.subplots(1,1)

cont = ax.contour(xixi,zz,pp,levels=25)
h1,l1 = cont.legend_elements("pp")
h1 = h1[::8]
l1 = l1[::8]
floats = []
for i in range(len(h1)):
    match = re.findall("[+-]?\d+\.\d+", l1[i])[0]
    floats.append(match)

l1newlist = []
for i in range(len(l1)):
    l1new = np.round(float(floats[i]),3)
    l1new = '$p = %.3f$'%l1new
    l1newlist.append(l1new)
    
ax.legend(h1,l1)
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_title("p'")

# %%
Wab = np.zeros((Nz,2*(N + 1) - 1))
def Wabs(z,r):
    thetaRj = 1/(.03*z + .04)
    return (1/2)*(1 + np.tanh((1/4)*(thetaRj)*(1/(np.abs(r)) - np.abs(r))))

for i, zval in enumerate(z):
    for j, rval in enumerate(xievencop):
        Wab[i][j] = Wabs(zval,rval)
# xieven = 6*(np.arccos(xi/3)/np.pi - 1/2)
# xixi, zz = np.meshgrid(xieven,z)

# u0 = np.transpose(q[0:N + 1,:])
# w0 = np.transpose(q[N + 1:2*(N + 1),:])
# u = u0.copy()
# xiNeg = np.where(xi < 0)
# u[:,xiNeg] = -u[:,xiNeg]
# plt.quiver(zz,xixi,w0 + W,u,color='white')
# plt.xlabel('z')
# plt.ylabel('r/R_j')
# plt.show()
#%%
#Streamline code
import matplotlib as mpl
# u = q[:]
## Set colormap for the colorbar
cmap = mpl.cm.rainbow
# u = np.transpose(np.real(u))
# w = np.transpose(np.real(w))
#### Use norm to set you vmax and vmin
norm = mpl.colors.Normalize(vmin=0, vmax=1)
from matplotlib import animation
veldata = np.concatenate((u,w),axis=1,dtype = complex)
for j in range(len(z)):
    veldata[j] = veldata[j]*np.exp(1j*(ηvec[j]))
# veldata = veldata*10**(-2)
qt = np.tile(veldata,(70,1,1))
t = np.linspace(0,20,num=70)
for index, time in enumerate(t):
    qt[index] = np.real(veldata)*np.exp(1j*(-omega*time))
    
fig, ax = plt.subplots(1,1)
ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xlabel('r')
ax.set_ylabel('z')
stream = ax.streamplot(xixi,zz,np.real(u),np.real(w) + Wab,density=(6,3),linewidth=.8,color=Wab + np.real(w),cmap='autumn',norm=norm,arrowstyle='-')

fig.colorbar(stream.lines)


def plotattime(i):
    ax.clear()
    u = np.zeros((len(z),len(xievencop)))
    w = np.zeros((len(z),len(xievencop)))
    for j in range(len(z)):
        u[j] = np.real(qt[i][j,0:2*(N + 1) - 1])
        w[j]= np.real(qt[i][j,2*(N + 1) - 1:4*(N + 1) - 2])
    print('%i/69 done'%i)
    # ax.title('t = {}'.format(i))
    ax.set_xlabel('r')
    ax.set_ylabel('z')
    index.set_text('time = %.3f s' % t[i])
    stream = ax.streamplot(xixi,zz,u,w + Wab,density=(6,3),linewidth=.8,color=Wab + np.real(w),cmap='autumn',norm=norm,arrowstyle='-')
    return stream,

index = ax.annotate('t = 0', xy=(0.75, 0.95), xycoords='axes fraction')
ani = animation.FuncAnimation(fig, func=plotattime, frames=np.arange(0, 70,1), blit=False,repeat=True)
plt.show()
writergif = animation.PillowWriter(fps=80)
ani.save('Streamline_jet_axisymmetric_visc_t0to20pi.gif',writer=writergif)
# %%
plt.style.use('default')
vellevels = np.transpose(np.sqrt(up**2 + wp**2))
# for j in range(len(z)):
    # vellevels[j] = vellevels[j]*np.exp(1j*(ηvec[j]))
    
fig, ax = plt.subplots(1,1)
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_title("q'")

# xixi, zz = np.meshgrid(xi,z)
# print(np.shape(vellevels))
# print(np.shape(xixi))
ax.contour(xixi,zz,Wab + vellevels,levels=500)
# %%
