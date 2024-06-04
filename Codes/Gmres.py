import numpy as np
import sys
import scipy as sp
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from scipy.linalg import kron
import scipy.sparse.linalg as spla
import time
np.set_printoptions(threshold=sys.maxsize)
#assembly of Ax=b from -u''(x) = f(x) on ]0,1[ and u(0)=u(1)=0  
#using central finite differences and the exact solution u(x) = (x(1-x))^2
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.callbacks = []
        self.internal_list = []
    def append(self, elem):
        self.internal_list.append(elem)
    def getList(self):
        return self.internal_list
    def __call__(self, rk=None):
       self.callbacks.append(rk)
       self.internal_list.append(rk)
       self.niter += 1
def setup_1d_poisson(n):
    h = 1./(n+1.)
    nn = n
    A = 2.*sp.eye(n)-np.diag(np.ones(n-1), 1)-np.diag(np.ones(n-1),-1)
    A = csc_matrix(A)
    xexact = np.zeros(n)
    x0 = np.zeros(n)
    b = np.zeros(n)
    for i in range(1,n-1):
        xi = i*h
        b[i] = -2 + 12*xi-12*xi**2
        b[i] = b[i] * h**2
        xexact[i] = (xi*(1-xi))**2
#    print(A,b)
    return A,b,xexact,x0,nn

def setup_2d_poisson(n):
    # compute mesh width from n (number of interior points per dimension)
    h = 1 / (n+1)
    nn = n*n
    # assemble coefficient matrix A, quick and dirty 
    I = np.eye(n)
    row_ind = np.arange(1,n,1)
    col_ind = np.arange(0,n-1,1)
    data = np.ones(n-1)
    E = csc_matrix((data,(row_ind, col_ind)), shape=[n,n])
    D = E+E.T-2*I
    A = -(np.kron(D,I)+np.kron(I,D))
    A = csc_matrix(A)
    # initialise remaining vectors
    xexact = np.zeros(n*n)
    x0 = np.zeros(n*n)
    b = np.zeros(n*n)
    # compute RHS and exact solution (inner points only)
    for i in range(0,n):
      for j in range(0,n):
        xij = i*h
        yij = j*h
        b[(j-1)*n+i] = -2*(6*xij**2-6*xij+1)*(yij-1)**2*yij**2 - 2*(xij-1)**2*xij**2*(6*yij**2-6*yij+1)
        b[(j-1)*n+i] = b[(j-1)*n+i] * (h*h)
        xexact[(j-1)*n+i] = (xij*(1-xij)*yij*(1-yij))**2
    return [A,b,xexact,x0,nn]

def setup_2d_convdiff(n,central_differences):
    h = 1/(n+1)
    nn = n*n
    a = lambda x,y:  20.*np.exp(3.5*(x**2+y**2))
    dadx = lambda x,y: 140.*x*np.exp(3.5*(x**2+y**2))
    A = np.zeros((nn,nn))

    #A = csc_matrix(A)
    for i in range(0,n):
        xij = i*h
        for j in range(0,n):
            yij = j*h
            k = (j-1)*n + i
            kiP = (j-1)*n + i + 1
            kiM = (j-1)*n + i - 1
            kjP = j*n + i
            kjM = (j-2)*n + i
            A[k][k] = A[k][k]+4.
            if j<n :
                A[k][kjP] = A[k][kjP]-1
            
            if j>1:
                A[k][kjM] = A[k][kjM]-1

            if i < n:
                A[k][kiP] = A[k][kiP]-1

            if i>1:
                A[k][kiM] = A[k][kiM]-1
            
            if central_differences:
                if i<n:
                    A[k][kiP] = A[k][kiP]+a(xij,yij)*(h/2)

                if i > 1:
                    A[k][kiM] = A[k][kiM]-a(xij,yij)*(h/2)
            else:
                if i>1:
                    A[k][kiM] = A[k][kiM]-a(xij,yij)*(h)
                
                A[k][k] = A[k][k]+a(xij,yij)*(h)
    b = np.zeros(nn)
    xexact = np.zeros(nn)
    for i in range(0,n):
        xij = i*h
        for j in range(0,n):
            yij = j*h
            k = (j-1)*n+1
            u = (xij*(1-xij)*yij*(1-yij))**2
            dudx = 2*yij**2*(1-yij)**2*xij*(1-xij)*(1 - 2*xij)

            dudxx = 2*yij**2 * (1-yij)**2 * ( 6*xij**2 - 6*xij + 1);
            dudyy = 2*xij**2*(1-xij)**2*( 6*yij**2 - 6*yij + 1);

            b[k] = - dudxx - dudyy + 0.5*dadx(xij,yij) * u + a(xij,yij) * dudx;
           
            b[k] = b[k] * (h*h);

            xexact[k] = u;
    x0 = np.zeros((nn))
    A = csc_matrix(A)
    return A,b,xexact,x0,nn

def GMRES(A, b, x0, m, tol, maxiter):
    n = A.shape[0]
    v = np.zeros((n, m + 1))
    h = np.zeros((m + 1, m))
    c = np.zeros(m)
    s = np.zeros(m)
    alpha = np.zeros(m)
    gamma = np.zeros(m + 1)
    x = np.copy(x0)
    r = b - A @ x
    rnorm = np.linalg.norm(r)
    res = []
    #res.append(rnorm)
    if rnorm < tol:
        print('Solution found: The solution is the initial solution')
        return x0, res, 0

    v[:, 0] = r / rnorm
    gamma[0] = rnorm

    for k in range(min(maxiter, m)):
        w = A @ v[:, k]
        for i in range(k + 1):
            h[i, k] = np.dot(w, v[:, i])
            w -= h[i, k] * v[:, i]
        h[k + 1, k] = np.linalg.norm(w)
        
        if h[k + 1, k] != 0 and k < m - 1:
            v[:, k + 1] = w / h[k + 1, k]

        # Apply Givens rotations to H to maintain upper Hessenberg form
        for i in range(k):
            temp = c[i] * h[i, k] + s[i] * h[i + 1, k]
            h[i + 1, k] = -s[i] * h[i, k] + c[i] * h[i + 1, k]
            h[i, k] = temp

        # Calculate new Givens rotation
        beta = np.sqrt(h[k, k]**2 + h[k + 1, k]**2)
        #if beta != 0:
        c[k] = h[k, k] / beta
        s[k] = h[k + 1, k] / beta
        h[k, k] = beta
        gamma[k + 1] = -s[k] * gamma[k]
        gamma[k] = c[k] * gamma[k]

        res.append(abs(gamma[k + 1]))

        if res[-1] <= tol:
            print('Solution found at iteration:', k + 1)
            break



    # Solve the upper triangular system to get the solution
    y = np.zeros(k + 1)
    for i in range(k, -1, -1):
        y[i] = gamma[i]
        for j in range(i + 1, k +1):
            y[i] -= h[i, j] * y[j]
        y[i] /= h[i, i]

    x += v[:, :k + 1] @ y
    return x, res, k+1
    # for i in range(k,-1,-1):
    #     temp = 0
    #     for j in range(i+1, k+1):
    #         temp = temp + h[i,j]*alpha[j]
    #         alpha[i] = (1/h[i,i]) * (gamma[i]- temp)
    # for i in range(0,k+1):
    #     x = x + alpha[i]*v[:,i]


def GMRESM(A, b, x0, m, tol, maxiter):
    n = np.shape(A)[0]
    k = 0
    x = np.zeros(n)
    x = x0
    r = np.zeros(n)
    r = b - A @ x0
    res = []
    rnorm = np.linalg.norm(r)
    res.append(rnorm)
    
    while k<=maxiter and (rnorm >= tol):
        x, r, ks = GMRES(A, b, x, m, tol, m)
        # for i in range(ks):
        #     res.append(r[i])
        res.extend(r)
        k += ks
        r = b - A @ x 
        rnorm = np.linalg.norm(r)
        res.append(rnorm)
        #print(rnorm, end='\r')
    
    print(k)
    return x, res, k
def PDGMRESM(A,b,x0,m,tol,maxiter):
    n = np.shape(A)[0]
    k = 0
    i = 0
    x = np.zeros(n)
    x = x0  
    r = np.zeros(n)
    r = b - A @ x0
    res = []
    rnorm = np.linalg.norm(r)
    res.append(rnorm)
    m_initial = int(m/2)
    m_min = 1
    d = 3
    m_next = m_initial
    alpha_p = 9
    alpha_d = -3
    rnorm_minus1= rnorm
    rnorm_current=rnorm
    rnorm_minus2=rnorm
    while k<=maxiter and (rnorm_current >= tol):
        m_old = m_next
        if i > 3:
            m_next = m_old+int(alpha_p*rnorm_current/rnorm_minus1+alpha_d*(rnorm_current-rnorm_minus2)/(2*rnorm_minus1))
            rnorm_minus2 = rnorm_minus1
            rnorm_minus1 = rnorm_current
        elif(i>2):
            m_next = m_old+int(alpha_p*rnorm_current/rnorm_minus1)
            rnorm_minus2 = rnorm_minus1
            rnorm_minus1 = rnorm_current
        else:
            rnorm_minus1 = rnorm_current
        if m_next < m_min :
            if m_initial<=m-d:
                m_next=m_initial+d
            m_next=m_initial
        if m_next > m:
            m_next = m*m
        x, r, ks = GMRES(A, b, x, m_next, tol, m_next)
        for i in range(ks):
              res.append(r[i])
        k += ks
        r = b - A @ x 
        rnorm_current = np.linalg.norm(r)
        i+=1
    print(k)#rnorm, end='\r')
        
    print(k)
    return x,res,k

counter = gmres_counter()
n =32
tol = 1e-8
fig,(ax) = plt.subplots(1)

A,b,xexact,x0,nn = setup_2d_convdiff(n,central_differences=False)
k = nn
print(k)
t = []
steps =[4,8,16,24,32]
for i in steps:
    
    # counter.internal_list = []
    # counter.niter = 0
    start = time.time()
    sol,res,it = PDGMRESM(A,b,x0,i,tol,k)
    #x,info= spla.gmres(A,b,xexact,tol=1e-8,callback=counter,restart=i)
    end = time.time()
    t.append(end-start)

    #ylist = counter.getList()
    #ax.plot(range(counter.niter),ylist,label='m:''%s '%i, linewidth=2)
    ax.plot(res ,label='PDGMRES(m):''%s '%i,linewidth=3,linestyle='--')
    # sol,res,it = GMRESM(A,b,x0,i,tol,k)
    # ax.plot(res ,label='GMRES(m):''%s '%i,linewidth=3)
    print(i)



#xl, info = spla.lgmres(A,b,x0,atol=1e-8,inner_m=8)#,callback=counter)
#plt.plot(res)


sol,res,it = GMRES(A,b,x0,k,tol,k)
# start = time.time()
ax.plot(res ,label='GMRES',linewidth=3,linestyle=':')
# end = time.time()
#
#ax.plot(res1,color='black', linewidth=2)
plt.legend()
plt.ylim(1e-9,1e-1)
plt.xlim(0.)
plt.yscale('log')

#print(np.linalg.norm(x-sol))
plt.show()
plt.plot(steps,t)
plt.scatter(steps,t)
plt.yscale('log')
plt.xticks(steps)
plt.show()
#print(counter.niter)