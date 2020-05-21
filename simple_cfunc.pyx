# cython: language_level=3

import numpy as np
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.interpolate import interp1d


################################################## Parameter List ##################################################

####### Constants #######

cdef double c_light = 299792458
cdef double G = 6.67430e-11
cdef double h_bar = 1.054571817e-34
cdef double pi = 3.1415926535897

cdef double MeV_J = 1.602176634e-13

cdef double m_p = 1.67262192369e-27
cdef double m_n = 1.67492749804e-27
cdef double m_e = 9.1093837015e-31
cdef double m_u = 1.883531475e-28
cdef double m_sun = 1.9884e30

cdef double e0 = 5.346e35

####### Saturation Density #######

cdef double n0 = 0.155e45

####### Taylor Coefficients #######

cdef double E_sat = -15.9 * MeV_J
cdef double E_sym = 31.7 * MeV_J
cdef double K_sat = 240 * MeV_J
cdef double L_sym = 58.9 * MeV_J

cdef double Q_sat = 0 * MeV_J
cdef double K_sym = 0 * MeV_J
cdef double Q_sym = 0 * MeV_J

####### Global Variable #######

cdef double delta_global = 0
cdef double ep_e_global = 0
cdef double ep_u_global = 0
cdef double m_0 = 0
cdef double p_0 = 0
cdef double r_0 = 0
cdef double y_0 = 2
cdef double tp = -1/3
cdef double lam_0 = 0
cdef double cross_point = 0

global Core   # used to store Core EOS

####### Flow Control #######

cycle_control_indicator = True
vs_control_indicator = True
radius_control_indicator = True
mass_control_indicator = True

####### List for Data Storage #######

p_for_coreEOS = []
rho_for_coreEOS = []

####### Filename #######

file_name = 'default'
EOSfile = 'default'
Odata_name = 'default'
Idata_name = 'default'
trans_data_name = 'default'





##################################################### Asymmetry #####################################################


cdef double n_l(double delta, double e_sym, double m_l):   # number density of leptons

    cdef double K, term, sq_term

    if e_sym <= 0:
        return 0

    K = 3 * pi**2 * h_bar**3 * c_light**3
    term = (4 * delta * e_sym)**2 - (m_l * c_light**2)**2
    if term < 0:
        sq_term = 0
    else:
        sq_term = term ** (3/2)

    return sq_term / K


cdef double beta_eq(double delta, double x_0):   # Condition of beta-equilibrium

    cdef double e_sym, n_p, n_e, n_u

    e_sym = E_sym + L_sym * x_0 + (1 / 2) * K_sym * x_0 ** 2 + (1 / 6) * Q_sym * x_0 ** 3
    n_p = n0 * (3 * x_0 + 1) * (1 - delta) / 2
    n_e = n_l(delta, e_sym, m_e)

    if 4 * delta * e_sym < m_u * c_light**2:
        return n_p - n_e
    else:
        n_u = n_l(delta, e_sym, m_u)
        return n_p - n_e - n_u


cdef double delta_of_x(double x_0):   # delta at some x

    cdef double output

    output = (fsolve(beta_eq, np.array([0.8]), args=x_0))[0]
    return output


################################################### EOS(rho(p_0)) ###################################################


# Crust-core Transition Equation
cdef double esat(double x):   # Calculation of saturation energy from x

    cdef double output = E_sat + 0.5 * K_sat * x ** 2 + 1 / 6 * Q_sat * x ** 3

    return output


cdef double esym(double x):   # Calculation of symmetry energy from x

    cdef double output = E_sym + L_sym * x + 0.5 * K_sym * x ** 2 + 1 / 6 * Q_sym * x ** 3

    return output


cdef double crust_core(double x):   # Equation of crust-core transition 

    cdef double d, output

    d = delta_of_x(x)
    output =  (1/9) * (3 * x + 1)**2 * derivative(esat, x, dx=abs(x) * 1e-3, n=2, order=5) \
        + (2/3) * (3 * x + 1) * derivative(esat, x, dx=abs(x) * 1e-3, order=5) \
        + d**2 * ((1/9) * (3 * x + 1)**2 * derivative(esym, x, dx=abs(x) * 1e-3, n=2, order=5) \
                + (2/3) * (3 * x + 1) * derivative(esym, x, dx=abs(x) * 1e-3, order=5) \
                - (2/esym(x)) * ((3 * x + 1)/3 * derivative(esym, x, dx=abs(x) * 1e-3, order=5))**2)

    return output


def set_t_point():   # Solution of 'x' of turning point
    global tp
    tp = -1/3
    tp = (fsolve(crust_core, -0.16))[0]



# Sly4EOS(Crust EOS, fitted from existing data)
inner_p = np.loadtxt('inner_p.txt')
inner_rho = np.loadtxt('inner_rho.txt')

outer_p = np.loadtxt('outer_p.txt')
outer_rho = np.loadtxt('outer_rho.txt')

Para1 = np.loadtxt('Para1.txt')
Para2 = np.loadtxt('Para2.txt')
Para3 = np.loadtxt('Para3.txt')
Para4 = np.loadtxt('Para4.txt')
ParaO = np.loadtxt('ParaO.txt')


def fitfunc(x, B, C, D, E, F, G, H):
    return B*x**0.4 + C*x**0.8 + D*x**1.2 + E*x**1.6 + F*x**2.4 + G*x**3.2 + H


cpdef double CrustEOS(double p):   # Crust EOS
   
    cdef double p1, p2, y1, y2

    if p > inner_p[50]:
        return fitfunc(p, *Para1)

    elif inner_p[50] >= p >= inner_p[49]:
        p1 = inner_p[49]
        p2 = inner_p[50]
        y1 = fitfunc(p1, *Para2)
        y2 = fitfunc(p2, *Para1)
        return (y2 - y1)/(p2 - p1) * p + (y1 * p2 - y2 * p1)/(p2 - p1)

    elif inner_p[50] > p > inner_p[25]:
        return fitfunc(p, *Para2)

    elif inner_p[25] >= p >= inner_p[24]:
        p1 = inner_p[24]
        p2 = inner_p[25]
        y1 = fitfunc(p1, *Para3)
        y2 = fitfunc(p2, *Para2)
        return (y2 - y1)/(p2 - p1) * p + (y1 * p2 - y2 * p1)/(p2 - p1)

    elif inner_p[24] > p > inner_p[15]:
        return fitfunc(p, *Para3)

    elif inner_p[15] >= p >= inner_p[14]:
        p1 = inner_p[14]
        p2 = inner_p[15]
        y1 = fitfunc(p1, *Para4)
        y2 = fitfunc(p2, *Para3)
        return (y2 - y1)/(p2 - p1) * p + (y1 * p2 - y2 * p1)/(p2 - p1)

    elif inner_p[14] > p > 6.90926e28:
        return fitfunc(p, *Para4)

    elif 6.90926e28 >= p >= outer_p[-1]:
        p1 = outer_p[-1]
        p2 = 6.90926e28
        y1 = fitfunc(p1, *ParaO)
        y2 = fitfunc(p2, *Para4)
        return (y2 - y1)/(p2 - p1) * p + (y1 * p2 - y2 * p1)/(p2 - p1)

    elif outer_p[-1] > p > outer_p[1]:
        return fitfunc(p, *ParaO)

    elif outer_p[1] >= p >= outer_p[0]:
        p0 = outer_p[1]
        y0 = fitfunc(p0, *ParaO)
        return y0/p0 * p

    else:
        return 0


# Core EOS(interpolation)
def set_CoreEOS():

    global Core

    Core = interp1d(p_for_coreEOS, rho_for_coreEOS, 'cubic')
    

# Total EOS
cpdef double rho(p_0):

    cdef double output

    if p_0 > cross_point:
        output = Core(p_0)
        return output

    else:
        output = CrustEOS(p_0)
        return output


############################################### P-rho Relationship ###############################################

cdef double ev_b(double x, double d):   # specific energy of baryons

    return esat(x) + esym(x) * d**2


cdef double ep_l(double n, double m):   # specific energy of leptons
    
    cdef double K, k, t, sq, output

    K = (m**4 * c_light**5) / (8 * pi**2 * h_bar**3)
    if n < 0:
        k = (3 * (-n) * pi ** 2) ** (1 / 3)
        k = -k
    else:
        k = (3 * n * pi ** 2) ** (1 / 3)
    t = (h_bar * k) / (m * c_light)
    sq = (1 + t**2)**0.5

    output = K * (t * sq * (1 + 2 * t**2) - np.log(t + sq))

    return output


cdef double p_b(double x, double d):   # pressure of baryons
    
    cdef double output = (n0/3) * (3*x+1)**2 * \
    			(K_sat*x + 0.5*Q_sat*x**2 + (L_sym + K_sym*x + 0.5*Q_sym*x**2)*d**2)

    return output


cdef double p_l(double x, double d):   # pressure of leptons

    global ep_e_global, ep_u_global

    cdef double ne, nu, ke, ku, Te1, Te2, Tu1, Tu2, p_e, p_u

    ne = n_l(d, esym(x), m_e)

    if ne < 0:
        ke = 0
    else:
        ke = (3 * ne * pi ** 2) ** (1 / 3)

    Te1 = ne * ((h_bar * ke * c_light) ** 2 + m_e ** 2 * c_light ** 4) ** 0.5
    Te2 = ep_l(ne, m_e)
    p_e = Te1 - Te2
    p_u = 0

    ep_e_global = Te2

    if 4 * d * esym(x) > m_u * c_light**2:
        nu = n_l(d, esym(x), m_u)

        if nu < 0:
            ku = 0
        else:
            ku = (3 * nu * pi ** 2) ** (1 / 3)

        Tu1 = nu * ((h_bar * ku * c_light) ** 2 + m_u ** 2 * c_light ** 4) ** 0.5
        Tu2 = ep_l(nu, m_u)
        p_u = Tu1 - Tu2

        ep_u_global = Tu2

    return p_e + p_u


cpdef double x_result_p(double x, double p):   # equation used for calculate x from P

    global delta_global

    cdef double d, pb, pl

    d = delta_of_x(x)

    pb = p_b(x, d)
    pl = p_l(x, d)

    delta_global = d

    return p - pb - pl


def x_p_cal(p_0):
    return (fsolve(x_result_p, 0.2, p_0))[0]



############################################### Caculate P, rho seperately ###########################################


cpdef double x_to_p(double x):   # calculate P from x

    cdef double d, pb, pl, output

    d = delta_of_x(x)

    pb = p_b(x, d)
    pl = p_l(x, d)

    output = pb + pl
    return output


cpdef double x_to_rho(double x):   # calculate rho from x
    
    cdef double d, rho_b, rho_l

    d = delta_of_x(x)

    rho_b = n0 * (3*x+1) * ((1+d)/2 * m_n + (1-d)/2 * m_p + ev_b(x, d) / c_light**2)
    
    n_e = n_l(d, esym(x), m_e)
    n_u = n_l(d, esym(x), m_u)

    rho_l = (ep_l(n_e, m_e) + ep_l(n_u, m_u)) / c_light**2
    return rho_b + rho_l


def set_p_rho_list():   # set the value of p_for_coreEOS, rho_for_coreEOS for specific model

    global p_for_coreEOS, rho_for_coreEOS

    p_for_coreEOS = []
    rho_for_coreEOS = []

    cdef double x_temp

    inx = np.linspace(-1/3, 3, 3000)

    for i in range(3000):

        x_temp = inx[i]

        p_for_coreEOS.append(x_to_p(x_temp))
        rho_for_coreEOS.append(x_to_rho(x_temp))




###################################################### Cross Point ##################################################


cpdef double crossP(p):

    cdef double a, b

    a = CrustEOS(p)
    b = Core(p)

    return a-b


def set_cross_point():

    global cross_point

    cross_point = 0

    cross_point = (fsolve(crossP, 5e31))[0]




################################################# Solving TOV Equation #############################################

# TOV Equation
cdef double M(double rho, double r):

    cdef double output = 4 * pi * r ** 2 * rho

    return output


cdef double P(double m, double rho, double p, double r):
    
    cdef double A = -(G * m * rho) / r ** 2 * (1 + p / (c_light ** 2 * rho)) * (1 + (4 * pi * r ** 3 * p) \
     / (c_light ** 2 * m)) / (1 - (2 * G * m) / (c_light ** 2 * r))
    
    return A


# Speed of Sound
cdef double Vs(double p):

    cdef double k, v

    k = derivative(rho, p, dx=1e-3 * p, order=5)

    if k < 0:
        return -1   # k mustn't be smaller than 0, return error code
    else: 
        v = 1 / np.sqrt(k)
        return v


# y for Tidal Deformability
cdef double Y(double m, double rho, double p, double y, double r):

    cdef double de_dp, F, Q, temp

    de_dp = c_light**2 / Vs(p) ** 2
    F = (1 - 4 * pi * r**2 * G * (rho * c_light**2 - p) / c_light**4) / (1 - 2 * m * G / (r * c_light**2))
    Q = (4 * pi * G / c_light**4) * (5 * rho * c_light**2 + 9 * p + (rho * c_light**2 + p) * de_dp) / (1 - 2 * m * G / (r * c_light**2)) \
        - 6 * (r**2 - 2 * r * m * G / c_light**2)**(-1) \
        - (4 * G**2 / (r**4 * c_light**8)) * (m * c_light**2 + 4 * pi * r**3 * p)**2 * (1 - 2 * m * G / (r * c_light**2))**(-2)

    temp = y**2 / r + y * F / r + r * Q
    return -temp


# Calculation of Tidal Deformability
cdef double tidal(double r, double m, double y):

    cdef double C, k2, lam

    C = G * m / (r * c_light**2)
    k2 = 8/5 * C**5 * (1 - 2 * C)**2 * (2 + 2 * C * (y - 1) - y) / (2 * C * (6 - 3 * y + 3 * C * (5 * y - 8)) \
                                           + 4 * C**3 * (13 - 11 * y + C * (3 * y - 2) + 2 * C**2 * (1 + y)) \
                                           + 3 * (1 - 2 * C)**2 * (2 - y + 2 * C * (y - 1)) * np.log(1 - 2 * C))
    lam = 2/3 * k2 * C**(-5)
    return lam


# RK4
cdef double RK4_cal(double h):

    global m_0, p_0, r_0, y_0, cycle_control_indicator, vs_control_indicator, mass_control_indicator

    cdef double ro, m_1, p_1, y_1, m_2, p_2, y_2, m_3, p_3, y_3, m_4, p_4, y_4, m, p, y, r 

    # Calculate k1
    ro = rho(p_0)
    m_1 = M(ro, r_0)
    p_1 = P(m_0, ro, p_0, r_0)
    y_1 = Y(m_0, ro, p_0, y_0, r_0)

    # Calculate k2
    ro = rho(p_0 + 0.5 * h * p_1)
    m_2 = M(ro, r_0 + 0.5 * h)
    p_2 = P(m_0 + 0.5 * h * m_1, ro, p_0 + 0.5 * h * p_1, r_0 + 0.5 * h)
    y_2 = Y(m_0 + 0.5 * h * m_1, ro, p_0 + 0.5 * h * p_1, y_0 + 0.5 * h * y_1, r_0 + 0.5 * h)

    # Calculate k3
    ro = rho(p_0 + 0.5 * h * p_2)
    m_3 = M(ro, r_0 + 0.5 * h)
    p_3 = P(m_0 + 0.5 * h * m_2, ro, p_0 + 0.5 * h * p_2, r_0 + 0.5 * h)
    y_3 = Y(m_0 + 0.5 * h * m_2, ro, p_0 + 0.5 * h * p_2, y_0 + 0.5 * h * y_2, r_0 + 0.5 * h)

    # Flow Control 1
    if p_0 + 0.5 * h * p_1 <= 1 or p_0 + 0.5 * h * p_2 <= 1 or p_0 + h * p_3 <= 1:
        cycle_control_indicator = False

    # Calculate k4
    ro = rho(p_0 + h * p_3)
    m_4 = M(ro, r_0 + h)
    p_4 = P(m_0 + h * m_3, ro, p_0 + h * p_3, r_0 + h)
    y_4 = Y(m_0 + h * m_3, ro, p_0 + h * p_3, y_0 + h * y_3, r_0 + h)

    if m_1 < 0 or m_2 < 0 or m_3 < 0 or m_4 < 0:
        cycle_control_indicator = False
        mass_control_indicator = False
        print('*****************************************')
        print('       Something wrong with mass! \n')

    # Boost
    m_0 = m_0 + 1/6 * h * (m_1 + 2 * m_2 + 2 * m_3 + m_4)
    p_0 = p_0 + 1/6 * h * (p_1 + 2 * p_2 + 2 * p_3 + p_4)
    y_0 = y_0 + 1/6 * h * (y_1 + 2 * y_2 + 2 * y_3 + y_4)
    r_0 = r_0 + h

    # Flow Control 2
    if p_0 <= 1: 
        cycle_control_indicator = False

    # Sound Speed control
    if p_0 > cross_point:
        if Vs(p_0) < 0:
            cycle_control_indicator = False
            vs_control_indicator = False
            print('*****************************************')
            print('            Monotony break! \n')
        elif Vs(p_0) > c_light:
            cycle_control_indicator = False
            vs_control_indicator = False
            print('*****************************************')
            print('         Sound speed too large! \n')


cpdef double RK4loop(p):

    global m_0, p_0, r_0, y_0, \
    radius_control_indicator, cycle_control_indicator, vs_control_indicator, mass_control_indicator

    # Initial Conditions
    m_0 = 0.0
    p_0 = p*e0
    r_0 = 0.0001
    y_0 = 2.0

    cycle_control_indicator = True
    radius_control_indicator = True
    vs_control_indicator = True
    mass_control_indicator = True

    # Step Length
    cdef double h1 = 10.0
    cdef double h2 = 1.0

    # Singular Point
    RK4_cal(h1)

    r_0 = r_0 - 0.0001

    while cycle_control_indicator:
        if p_0 >= 5e31:
            RK4_cal(h1)
        else:
            RK4_cal(h2)

        if r_0 > 18000:
            radius_control_indicator = False
            print('*****************************************')
            print('         radius too large! \n')
            break
    
    # print(m_0/m_sun, r_0/1000, tidal(r_0, m_0, y_0))



def f_14(p):   # used in fsolve for the data at 1.4 Msun

    RK4loop(p)

    m = m_0/m_sun

    return m - 1.4


def f_197(p):   # used in fsolve for the data at 1.4 Msun

    RK4loop(p)

    m = m_0/m_sun

    return m - 1.97


def get_m14_data():

    fsolve(f_14, 0.02, maxfev=20, xtol=1e-3)

    output = [m_0/m_sun, r_0/1000, tidal(r_0, m_0, y_0)]

    return output


def get_m197_data():

    fsolve(f_197, 0.04, maxfev=15, xtol=1e-3)

    output = [m_0/m_sun, r_0/1000, tidal(r_0, m_0, y_0)]

    return output


#################################################### Initialization ################################################

def __initialization__():
    set_p_rho_list()
    set_CoreEOS()
    set_cross_point()

################################################# Functional Functions #############################################

# select the EOSs with esym > 0
def select_1st():

    global Q_sat, K_sym, Q_sym, tp, p_for_coreEOS, rho_for_coreEOS

    count = 0
    t_count = 0
    alldata = 19*25*44

    with open(file_name, 'a') as f:
        f.write('[')

    for i in range(19):
        Q_sat = (-130 + 5 * i) * MeV_J
        for j in range(25):
            K_sym = (-120 + 5 * j) * MeV_J
            for k in range(44):
                Q_sym = (70 + 10 * k) * MeV_J

                x = np.linspace(-1/3, 3, 3000)
                esym_value = np.zeros(3000)

                for ii in range(3000):
                    esym_value[ii] = esym(x[ii])

                t_count += 1

                if all(esym_value > 0):
                    output = [-130 + 5 * i, -120 + 5 * j, 70 + 10 * k]
                    with open(file_name, 'a') as f:
                        f.write(str(output))  # Save data
                        f.write(',')

                    count += 1                

                print('{0}/{1} checked, {2} added'.format(t_count, alldata, count))

    with open(file_name, 'a') as f:
        f.write(']')




# 2nd round selection 
# (P at turning point <= P at cross point <= 3.27727e32(Sly4 EOS P(n_0)))
def select_2nd():

    global Q_sat, K_sym, Q_sym, tp

    with open(EOSfile, 'r') as f:
        raw = f.read()
        data = eval(raw)

    count = 0
    t_count = 0
    alldata = np.shape(data)[0]

    with open(file_name, 'a') as f:
        f.write('[')

    for i in data:

        Q_sat = i[0] * MeV_J
        K_sym = i[1] * MeV_J
        Q_sym = i[2] * MeV_J

        #initialize point set for interpolation

        __initialization__()

        set_t_point()   # Caldulate the Crust-Core turning point

        t_count += 1

        if tp < 0: 
            if 0 < x_to_p(tp) and 0 < cross_point:
                if x_to_p(tp) <= cross_point <= 3.27727e32: 

                    with open(file_name, 'a') as f:
                        f.write(str(i))  # Save data
                        f.write(',')

                    count += 1
                
        print('{0}/{1} checked, {2} added'.format(t_count, alldata, count))

    with open(file_name, 'a') as f:
        f.write(']') 


# 3rd round selection
# model should be correct when P = 0.01*e0(mass around 1.1M_sun to 1.4M_sun)
def select_3rd():

    global Q_sat, K_sym, Q_sym

    with open(EOSfile, 'r') as f:
        raw = f.read()
        data = eval(raw)

    with open(file_name, 'a') as f:
        f.write('[')

    count = 0
    t_count = 0
    alldata = np.shape(data)[0]

    for i in data:

        Q_sat = i[0] * MeV_J
        K_sym = i[1] * MeV_J
        Q_sym = i[2] * MeV_J

        #initialize point set for interpolation

        __initialization__()

        RK4loop(0.01)

        t_count += 1

        if vs_control_indicator == True and mass_control_indicator == True:

            count += 1

            with open(file_name, 'a') as f:
                f.write(str(i))
                f.write(',')

        print('{0}/{1} checked, {2} added'.format(t_count, alldata, count))

    with open(file_name, 'a') as f:
        f.write(']')


# 4th round selection by the condition of tidal deformability at 1.4Msun
def select_4th():

    global Q_sat, K_sym, Q_sym

    with open(EOSfile, 'r') as f:
        raw = f.read()
        data = eval(raw)

    with open(file_name, 'a') as f:
        f.write('[')

    with open(Odata_name, 'a') as f:
        f.write('[')

    count = 0
    t_count = 0
    alldata = np.shape(data)[0]

    for i in data:

        Q_sat = i[0] * MeV_J
        K_sym = i[1] * MeV_J
        Q_sym = i[2] * MeV_J

        #initialize point set for interpolation

        __initialization__()

        output = get_m14_data()

        print(output)

        t_count += 1

        if vs_control_indicator == True and mass_control_indicator == True: 

            if 70 < output[2] < 580:

                count += 1

                with open(file_name, 'a') as f:
                    f.write(str(i))
                    f.write(',')

                with open(Odata_name, 'a') as f:
                    f.write(str(output))
                    f.write(',')

        print('{0}/{1} checked, {2} added'.format(t_count, alldata, count))

    with open(file_name, 'a') as f:
        f.write(']')

    with open(Odata_name, 'a') as f:
        f.write(']')


# 5th round selection by the condition of supporting max mass at least 1.97Msun
# compute the NS observable at 1.97Msun
def select_5th():

    global Q_sat, K_sym, Q_sym

    with open(EOSfile, 'r') as f:
        raw = f.read()
        data = eval(raw)

    with open(Idata_name, 'r') as f:
        raw = f.read()
        data14 = eval(raw)

    with open(file_name, 'a') as f:
        f.write('[')

    with open(Odata_name, 'a') as f:
        f.write('[')

    with open(trans_data_name, 'a') as f:
        f.write('[')   

    count = 0
    t_count = 0
    alldata = np.shape(data)[0]

    for i in data:

        Q_sat = i[0] * MeV_J
        K_sym = i[1] * MeV_J
        Q_sym = i[2] * MeV_J

        #initialize point set for interpolation

        __initialization__()

        output = get_m197_data()

        print(output)

        t_count += 1

        if vs_control_indicator == True and mass_control_indicator == True: 

            count += 1

            with open(file_name, 'a') as f:
                f.write(str(i))
                f.write(',')

            with open(Odata_name, 'a') as f:
                f.write(str(output))
                f.write(',')

            with open(trans_data_name, 'a') as f:
                f.write(str(data14[t_count-1]))
                f.write(',')

        print('{0}/{1} checked, {2} added'.format(t_count, alldata, count))

    with open(file_name, 'a') as f:
        f.write(']')

    with open(Odata_name, 'a') as f:
        f.write(']')

    with open(trans_data_name, 'a') as f:
        f.write(']')


def figure_paint():

    global Q_sat, K_sym, Q_sym

    with open(EOSfile, 'r') as f:
        raw = f.read()
        data = eval(raw)

    with open(file_name, 'a') as f:
        f.write('[')

        count = 0

    for i in data:

        Q_sat = i[0] * MeV_J
        K_sym = i[1] * MeV_J
        Q_sym = i[2] * MeV_J

        temp_data = []

        count += 1

        #initialize point set for interpolation
        __initialization__()
        
        for j in range(55):

            if j <= 8: 
                p_temp = 0.001 * j + 0.002
            elif j <= 23:
                p_temp = 0.004 * j - 0.022
            elif j <= 33:
                p_temp = 0.008 * j - 0.106
            else:
                p_temp = 0.012 * j - 0.158

            # Initial condition test
            if p_temp*e0 > np.max(p_for_coreEOS):
                print('*****************************************')
                print('         pressure too large! \n')
                print('         {} model finished. \n'.format(count))
                break

            RK4loop(p_temp)

            output = [m_0/m_sun, r_0/1000, tidal(r_0, m_0, y_0)]
            print(output)

            if vs_control_indicator == False or mass_control_indicator == False:
                print('*****************************************')
                print('         {} model finished. \n'.format(count))
                break

            if output[0] < 0.7:
                print('P = {}e0 counted'.format(p_temp))
                continue

            if radius_control_indicator == True and output[0] >= 0.7:
                print('P = {}e0 counted'.format(p_temp))
                temp_data.append(output)

            if j == 54:
                print('*****************************************')
                print('         {} model finished. \n'.format(count))

        with open(file_name, 'a') as f:
            f.write(str(temp_data))

        if count < np.shape(data)[0]:
            with open(file_name, 'a') as f:
                f.write(',')

    with open(file_name, 'a') as f:
        f.write(']')





############################################### Value setting and getting ###########################################

def set_file_name(input):
    global file_name
    file_name = input


def set_EOSfile_name(input):
    global EOSfile
    EOSfile = input


def set_Odata_name(input):
    global Odata_name
    Odata_name = input


def set_Idata_name(input):
    global Idata_name
    Idata_name = input    


def set_trans_data_name(input):
    global trans_data_name
    trans_data_name = input 


def set_Q_sat(input):
    global Q_sat
    Q_sat = input * MeV_J


def set_K_sym(input):
    global K_sym
    K_sym = input * MeV_J


def set_Q_sym(input):
    global Q_sym
    Q_sym = input * MeV_J


def get_cross_point():
    return cross_point


