import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
import time

from mpl_toolkits.mplot3d import axes3d
np.set_printoptions(threshold=sys.maxsize)
from scipy.sparse.linalg import lgmres
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
    h = 1./(n)
    nn = n
    A = 2.*np.eye(n)-np.diag(np.ones(n-1), 1)-np.diag(np.ones(n-1),-1)
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
    # Compute mesh width from n (number of interior points per dimension)
    h = 1 / (n +1)
    nn = n * n
    
    # Assemble coefficient matrix A
    I = np.eye(n)
    row_ind = np.arange(1, n, 1)
    col_ind = np.arange(0, n-1, 1)
    data = np.ones(n-1)
    E = csc_matrix((data, (row_ind, col_ind)), shape=[n, n])
    D = E + E.T - 2 * I
    A = -(np.kron(D, I) + np.kron(I, D))
    A = csc_matrix(A)
    
    # Initialize remaining vectors
    xexact = np.zeros(nn)
    x0 = np.zeros(nn)
    b = np.zeros(nn)
    
    # Compute RHS and exact solution (inner points only)
    for i in range(n):
        for j in range(n):
            xij = (i + 1) * h
            yij = (j + 1) * h
            b[j * n + i] = -2 * (6 * xij**2 - 6 * xij + 1) * (yij - 1)**2 * yij**2 - \
                           2 * (xij - 1)**2 * xij**2 * (6 * yij**2 - 6 * yij + 1)
            b[j * n + i] = b[j * n + i] * (h * h)
            xexact[j * n + i] = (xij * (1 - xij) * yij * (1 - yij))**2
    
    return [A, b, xexact, x0, nn]

def setup_2d_convdiff(n, central_differences):
    h = 1 / (n + 1)
    nn = n * n
    dimstr = f"{n}x{n}"

    a = lambda x, y: 20 * np.exp(3.5 * (x**2 + y**2))
    dadx = lambda x, y: 140 * x * np.exp(3.5 * (x**2 + y**2))

    A = np.zeros((nn, nn))

    for i in range(1, n + 1):
        xij = i * h  # x coordinate
        for j in range(1, n + 1):
            yij = j * h  # y coordinate
            k = (j - 1) * n + i  # center index (linear)
            kiP = (j - 1) * n + i + 1  # i+1
            kiM = (j - 1) * n + i - 1  # i-1
            kjP = j * n + i  # j+1
            kjM = (j - 2) * n + i  # j-1

            # center part of FD stencil (x & y) : - 4 u(i,j)
            A[k - 1, k - 1] += 4

            # d^2/dy^2 \approx ( u(i,j+1) - 2 u(i,j) + u(i,j-1) ); - 2 u(i,j) included in center part
            if j < n:
                A[k - 1, kjP - 1] -= 1
            if j > 1:
                A[k - 1, kjM - 1] -= 1

            # d^2/dx^2 \approx ( u(i+1,j) - 2 u(i,j) + u(i-1,j) ); - 2 u(i,j) included in center part
            if i < n:
                A[k - 1, kiP - 1] -= 1
            if i > 1:
                A[k - 1, kiM - 1] -= 1

            # 0.5*da/dx u
            A[k - 1, k - 1] += 0.5 * dadx(xij, yij) * (h**2)

            # a du / dx
            if central_differences:  # central differences
                if i < n:
                    A[k - 1, kiP - 1] += a(xij, yij) * (h / 2)
                if i > 1:
                    A[k - 1, kiM - 1] -= a(xij, yij) * (h / 2)
            else:  # backward differences
                if i > 1:
                    A[k - 1, kiM - 1] -= a(xij, yij) * h
                A[k - 1, k - 1] += a(xij, yij) * h

    b = np.zeros(nn)
    xexact = np.zeros(nn)
    
    # exact function (x(1-x)y(1-y))^2
    for i in range(1, n + 1):
        xij = i * h
        for j in range(1, n + 1):
            yij = j * h
            k = (j - 1) * n + i  # linear index
            
            u = (xij * (1 - xij) * yij * (1 - yij))**2
            dudx = 2 * yij**2 * (1 - yij)**2 * xij * (1 - xij) * (1 - 2 * xij)
            dudxx = 2 * yij**2 * (1 - yij)**2 * (6 * xij**2 - 6 * xij + 1)
            dudyy = 2 * xij**2 * (1 - xij)**2 * (6 * yij**2 - 6 * yij + 1)

            b[k - 1] = -dudxx - dudyy + 0.5 * dadx(xij, yij) * u + a(xij, yij) * dudx
            b[k - 1] *= h * h

            xexact[k - 1] = u

    x0 = np.zeros(nn)
    A = csc_matrix(A)

    return A, b, xexact, x0, nn

def GMRES(A, b, x0, m, tol, maxiter):
    n = A.shape[0]
    v = np.zeros((n, m + 1))
    h = np.zeros((m + 1, m))
    c = np.zeros(m)
    s = np.zeros(m)
    alpha = np.zeros(m)
    gamma = np.zeros(m + 1)
    x = np.copy(x0)
    r = b - A.dot(x)
    rnorm = np.linalg.norm(r)
    res = []
    #res.append(rnorm)
    if rnorm < tol:
        print('Solution found: The solution is the initial solution')
        return x0, res, 0

    v[:, 0] = r / rnorm
    gamma[0] = rnorm

    for k in range(min(maxiter, m)):
        w = A.dot(v[:, k])
        for i in range(k+1):
            h[i, k] = np.dot(w, v[:, i])
            w -= h[i, k] * v[:, i]
        h[k + 1, k] = np.linalg.norm(w)
        

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
        v[:,k+1] = (1/h[k+1,k])*w
        if res[-1] <= tol or k == maxiter-1:
            for i in range(k,-1,-1):
                temp = 0
                for j in range(i+1,k+1):
                    temp = temp + h[i,j]*alpha[j]
                alpha[i] = (1/h[i,i])*(gamma[i]-temp)
            for i in range(k+1):
                x = x+alpha[i]*v[:,i]
            #print('Solution found at iteration:', k + 1)
            break     
        
    return x , res , k+1,v



def GMRESM(A, b, x0, m, tol, maxiter):
    n = np.shape(A)[0]
    k = 0
    x = np.copy(x0)
    r = b - A @ x0
    res = []
    rnorm = np.linalg.norm(r)
    res.append(rnorm)
    v = np.zeros((n,m+1))
    while k<=maxiter and (rnorm > tol):
        x, r, ks,v = GMRES(A, b, x, m, tol, m)
        res.extend(r)
        k += ks
        rnorm = res[-1]
    return x, res, k, v

def LGMRES(A, b, x0, m, tol, max_iter):
    n = A.shape[0]
    i_outer = 0
    r = [b - A @ x0]
    x1 = np.copy(x0)
    rnorm = np.linalg.norm(r[-1])
    res = [rnorm]
    it = 0
    z = np.zeros((n, 3))

    def LGMRESMinner(A, b, x, m, k_extend, tol, z, restart, max_iter):
        p = m + k_extend
        v = np.zeros((n, p + 1))
        ws = np.zeros((n, p))
        h = np.zeros((p + 1, p))
        c = np.zeros(p)
        s = np.zeros(p)
        alpha = np.zeros(p)
        gamma = np.zeros(p + 1)
        inner_res = []
        inner_r = b - A @ x
        y = np.zeros(n)
        inner_rnorm = np.linalg.norm(inner_r)
        v[:, 0] = inner_r / inner_rnorm
        gamma[0] = inner_rnorm

        if inner_rnorm < tol:
            return x, inner_res, z, 0

        for k in range(p):
            if k < m:
                w = A @ v[:, k]
            else:
                w = A @ z[:, (k - m)]
            for i in range(k + 1):
                h[i, k] = np.dot(w, v[:, i])
                w -= h[i, k] * v[:, i]
            h[k + 1, k] = np.linalg.norm(w)

            # Apply Givens rotations to H to maintain upper Hessenberg form
            for i in range(k):
                temp = c[i] * h[i, k] + s[i] * h[i + 1, k]
                h[i + 1, k] = -s[i] * h[i, k] + c[i] * h[i + 1, k]
                h[i, k] = temp

            # Calculate new Givens rotation
            beta = np.sqrt(h[k, k] ** 2 + h[k + 1, k] ** 2)
            c[k] = h[k, k] / beta
            s[k] = h[k + 1, k] / beta
            h[k, k] = beta
            gamma[k + 1] = -s[k] * gamma[k]
            gamma[k] = c[k] * gamma[k]

            v[:, k + 1] = w / h[k + 1, k]
            inner_res.append(abs(gamma[k + 1]))
            if inner_res[-1]<=tol or k == p - 1:
                ws[:, :m] = v[:, :m]
                for q in range(k_extend):
                    ws[:, m + q] = z[:, q]

                for i in range(k, -1, -1):
                    temp = 0
                    for j in range(i + 1, k + 1):
                        temp += h[i, j] * alpha[j]
                    alpha[i] = (gamma[i] - temp) / h[i, i]

                y = alpha[:]
                for i in range(k_extend - 1,0,-1):
                    z[:, i] = z[:, i-1]
                z[:, 0] = ws @ y

                x += z[:, 0]
                inner_r = b - A @ x
                inner_res.append(np.linalg.norm(inner_r))
                break

        return x, inner_res, z, k + 1

    while res[-1] > tol and i_outer <1000:
        r1 = []
        if i_outer == 0:
            x1, r1,k1,v = GMRES(A, b, x0, m + 3, tol, m+3)
            res.extend(r1)
            z  = v[:,-4:-1]
            z[:, 0] = x1 - x0
            it += k1
        elif i_outer <=3:
            x1, r1, z, k1 = LGMRESMinner(A, b, x1, m, i_outer - 1, tol, z, i_outer, 10000)
            res.extend(r1)
            it += k1
        else:
            x1, r1, z, k1 = LGMRESMinner(A, b, x1, m, 3, tol, z, i_outer, 10000)
            res.extend(r1)
            it += k1
        i_outer += 1

    return x1, res, it


def alphaGMRESM(A, b, x0, m, tol, maxiter):
    n = np.shape(A)[0]
    k = 0
    x = np.zeros(n)
    x = x0
    r = np.zeros(n)
    r = b - A @ x0
    res = []
    rnorm = np.linalg.norm(r)
    res.append(rnorm)
    cr = 1
    max_cr = .99
    min_cr = .175

    d = 3
    m_max = m
    m_old = m
    i = 0
    while k<=maxiter and (rnorm > tol):
        if(cr>max_cr or k == 0):
            m = m_max
        elif(cr<min_cr):
            m = m_old
        else:
            if((m_old-d)>=4):
                m = m_old-d
            else:
                m = m_max
        x, r, ks,_ = GMRES(A, b, x, m, tol, m)
        res.extend(r)
        k += ks
        res_last = rnorm
        rnorm = res[-1]
        cr = rnorm/res_last
        m_old = m
    return x, res, k

def PDGMRESM(A,b,x0,m,tol,maxiter):
    m_max =m*m
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
    while k<=maxiter and (rnorm_current > tol):
        m_old = m_next
        if i > 3:
            m_next = m_old+int(np.floor(alpha_p*rnorm_current/rnorm_minus1+alpha_d*(rnorm_current-rnorm_minus2)/(2*rnorm_minus1)))
            rnorm_minus2 = rnorm_minus1
            rnorm_minus1 = rnorm_current
        elif(i>2):
            m_next = m_old+int(np.floor(alpha_p*rnorm_current/rnorm_minus1))
            rnorm_minus2 = rnorm_minus1
            rnorm_minus1 = rnorm_current
        else:
            rnorm_minus1 = rnorm_current
        if m_next < m_min :
            if m_next<=m-d:
                m_next=m_initial+d
            m_next=m_initial
        if m_next > m_max:
            m_next = m_max
        x, r, ks,_ = GMRES(A, b, x, m_next, tol, m_next)
        res.extend(r)
        k += ks
        rnorm_current = res[-1]
        i+=1
    return x,res,k


counter = gmres_counter()
n =128
tol = 1e-8
fig,(ax) = plt.subplots(1)

A,b,xexact,x0,nn = setup_2d_convdiff(n,central_differences=False)
k = nn
print(k)

t = []
t1 = []
t2 = []
t3 = []
t4 = []
steps =[4,8,16,24,32,64]
l = 0
for i in steps:
    
    color = next(ax._get_lines.prop_cycler)['color']
    start = time.time()
    sol1,res,it = PDGMRESM(A,b,x0,i,tol,k)
    end = time.time()
    t.append(end-start)
    ax.plot(res ,label='PDGMRES(m):''%s '%i,linewidth=2,linestyle='--',color = color)
    print('PDGMRES(m) TIME: ', t[l])
    
    start = time.time()
    sol,r,it = alphaGMRESM(A,b,x0,i,tol,k)
    end = time.time()
    t1.append(end-start)
    ax.plot(r,label='alphaGMRES(m):''%s '%i,linewidth=2,linestyle='-.',color = color)
    print('alphaGMRES(m) TIME: ', t1[l])
    
    start = time.time()
    sol1,res,it = LGMRES(A,b,x0,i-3,tol,k)
    end = time.time()
    t2.append(end-start)
    ax.plot(res ,label='LGMRES(m,3):''%s '%i,linewidth=2,linestyle='dotted',color = color)
    print('LGMRES(m) TIME: ', t2[l])
    
    start = time.time()
    Lsol = lgmres(A,b,x0,inner_m=i,tol=tol,outer_k=3,callback=counter)
    end = time.time()
    print('ERROR: ',np.linalg.norm(sol-Lsol[0]))
    t4.append(end-start)
    print('scipyLGMRES(m) TIME: ', t4[l])

    start = time.time()
    sol,res,it,_ = GMRESM(A,b,x0,i,tol,k)
    end = time.time()
    t3.append(end-start)
    ax.plot(res ,label='GMRES(m):''%s '%i,linewidth=2,color = color)
    print('GMRES(m) TIME: ', t3[l])
    print('m: ',i)
    l += 1




start = time.time()
sol,res,it,_ = GMRES(A,b,x0,k,tol,k)


end = time.time()
print('GMRES TIME: ',end-start)
timeee = end-start
print(timeee)
ax.plot(res ,label='GMRES',linewidth=3,linestyle='-')

plt.legend()

plt.xlim(0.)
plt.yscale('log')
plt.ylabel('residual norm')
plt.xlabel('iterations')
plt.show()
gmrestime = []
for i in range(len(steps)):
    gmrestime.append(timeee)

plt.scatter(steps,gmrestime,label='GMRES',color='blue')
plt.plot(steps,gmrestime,color='blue')

plt.scatter(steps,t,label='PDGMRES',color='orange')
plt.plot(steps,t,color='orange')

plt.scatter(steps,t1,label='alphaGMRES(m)',color='black')
plt.plot(steps,t1,color='black')

plt.scatter(steps,t2,label='LGMRES(m,3)',color='red')
plt.plot(steps,t2,color='red')

plt.scatter(steps,t4,label='scipyLGMRES(m,3)',color='cyan')
plt.plot(steps,t4,color='cyan')

plt.scatter(steps,t3,label='GMRES(m)',color='green')
plt.plot(steps,t3,color='green')
#plt.yscale('log')
plt.xticks(steps)
plt.ylabel('time in s')
plt.xlabel('m')
plt.legend()
plt.show()
x = np.linspace(start = 0, stop=1,num = n)
x,y = np.meshgrid(x,x)
b = np.reshape(sol,(n,n))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x,y, b, cmap='viridis')  # x1 for X, x1 for Y, sol for Z

# Labeling and showing the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Surface Plot 2D ConvDiff Problem')
plt.show()
#print(counter.niter)