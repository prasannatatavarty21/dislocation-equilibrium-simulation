import numpy as np
from scipy.integrate import IntegrationWarning, quad
import math
from tqdm import tqdm
import warnings
import multiprocessing
import csv

warnings.filterwarnings("ignore", category=IntegrationWarning)

# ==========================================
# Material Constants and Simulation Parameters
# ==========================================

mu = 26.97                                                                # Shear modulus
nu = 0.33                                                                 # Poisson's ratio
a_bar = 0.6                                                               # Precipitate semi-axis a
b_bar = 0.35                                                              # Precipitate semi-axis b

# Geometric positioning parameters

x_0_bar = 8                                                               # Precipitate center offset in x-direction
y_0_bar = 8                                                               # Precipitate center offset in y-direction
h_bar = 20
h_tilda = 1

# Parameters for the simulation -- Change as required

d_bar_values = [0.25, 0.5, 1, 1.75]                                       # Initial horizontal position of dislocation
eigen_values = [0.01, 0.05, 0.1, 0.4]                                     # Eigenstrain values
omega_values = [math.radians(0), math.radians(0.25), math.radians(0.5)]   # Grain boundary strength in radians

# ==========================================
# Global stiffness matrix
# ==========================================

C_global = np.zeros((2,2,2,2))
for i in range(0,2):
    for j in range(0,2):
        for k in range(0,2):
            for l in range(0,2):
                # Isotropic elasticity tensor construction
                C_global[i][j][k][l] = mu*(((2*nu*(i==j)*(k==l))/(1-2*nu)) + (i==k)*(j==l) + (i==l)*(j==k))

# ==========================================
# Multiprocessing
# ==========================================

def worker_wrapper(args):
    p_val, d, e, o = args
    no_last, with_last = calculate_Delta_Eel(p_val, False, d, e, o)
    return no_last, with_last

# ==========================================
# Mathematical Functions
# ==========================================

def integrand_q1(theta, p_bar, i, j, k, l, d_bar, eigen, omega):
    """
    Core integrand for the stress field of precipitate 1.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    zi = cos_theta if i==0 else sin_theta
    zj = cos_theta if j==0 else sin_theta
    zk = cos_theta if k==0 else sin_theta
    zl = cos_theta if l==0 else sin_theta

    delta_ij = 1 if i==j else 0
    delta_kl = 1 if k==l else 0

    beta = np.sqrt((a_bar*cos_theta)**2 + (b_bar*sin_theta)**2)

    gamma = (d_bar-x_0_bar)*cos_theta + (p_bar-y_0_bar)*sin_theta         # For precipitate in Quadrants 1
    # gamma = (d_bar+x_0_bar)*cos_theta + (p_bar-y_0_bar)*sin_theta       # For precipitate in Quadrants 2
    
    zz_inv_ij = (1/mu)* (delta_ij - (zi*zj)/(2*(1-nu)))
    
    # Piecewise boundary check for the inclusion
    if(beta < gamma):
        return ((-a_bar*b_bar)/2*np.pi)*zz_inv_ij*zk*zl*(1/beta**2)*(1- (np.abs(gamma)/np.sqrt(gamma**2 - beta**2)))
    else:
        return ((-a_bar*b_bar)/2*np.pi)*zz_inv_ij*zk*zl*(1/beta**2)

def integrand_q3(theta, p_bar, i, j, k, l, d_bar, eigen, omega):
    """
    Core integrand for the stress field of precipitate 2.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    zi = cos_theta if i==0 else sin_theta
    zj = cos_theta if j==0 else sin_theta
    zk = cos_theta if k==0 else sin_theta
    zl = cos_theta if l==0 else sin_theta

    delta_ij = 1 if i==j else 0
    delta_kl = 1 if k==l else 0

    beta = np.sqrt((a_bar*cos_theta)**2 + (b_bar*sin_theta)**2)

    gamma = (d_bar+x_0_bar)*cos_theta + (p_bar+y_0_bar)*sin_theta         # For precipitate in Quadrants 3
    # gamma = (d_bar-x_0_bar)*cos_theta + (p_bar+y_0_bar)*sin_theta       # For precipitate in Quadrants 4
    
    zz_inv_ij = (1/mu)* (delta_ij - (zi*zj)/(2*(1-nu)))
    
    if(beta < gamma):
        return ((-a_bar*b_bar)/2*np.pi)*zz_inv_ij*zk*zl*(1/beta**2)*(1- (np.abs(gamma)/np.sqrt(gamma**2 - beta**2)))
    else:
        return ((-a_bar*b_bar)/2*np.pi)*zz_inv_ij*zk*zl*(1/beta**2)

def calculate_sigma_c(p_bar, precipitate, d_bar, eigen, omega):
    """
    Calculates the constrained stress tensor (sigma_c) for a given precipitate.
    """
    D = np.zeros((2,2,2,2))
    quad_cache = {} # Dictionary to store previously calculated integrals
    
    # 1. Evaluate the D tensor integrals
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                for l in range(0,2):
                    # Sort indices to spot mathematically identical tensor components
                    cache_key = (tuple(sorted((i, j))), tuple(sorted((k, l))))
                    
                    if cache_key not in quad_cache:
                        if precipitate == 1:
                            val, _ = quad(integrand_q1, 0, 2*np.pi, args=(p_bar, i, j, k, l, d_bar, eigen, omega), limit=10000)
                        else:
                            val, _ = quad(integrand_q3, 0, 2*np.pi, args=(p_bar, i, j, k, l, d_bar, eigen, omega), limit=10000)
                        quad_cache[cache_key] = val
                    
                    D[i][j][k][l] = quad_cache[cache_key]

    # 2. Compute the S (Eshelby) tensor
    S = np.zeros((2,2,2,2))
    for i in range(0,2):
        for j in range(0,2):
            for m in range(0,2):
                for n in range(0,2):
                    term1 = 0
                    term2 = 0
                    for k in range(0,2):
                        for l in range(0,2):
                            # Multiply by the pre-computed global stiffness tensor
                            term1 += C_global[l][k][m][n]*D[i][k][l][j]
                            term2 += C_global[l][k][m][n]*D[j][k][l][i]
                    S[i][j][m][n] = -0.5*(term1+term2)

    # 3. Define the eigenstrain tensor
    e_star = [[eigen, 0],
              [0,     0]]                                                 # Change the eigenstrain tensor form as required

    # 4. Compute constrained strain (e_c)
    e_c = np.zeros((2,2))
    for i in range(0,2):
        for j in range(0,2):
            s_term = 0
            for k in range(0,2):
                for l in range(0,2):
                    s_term += S[i][j][k][l]*e_star[k][l]
            e_c[i][j] = s_term

    # 5. Contract stiffness and strain to get stress
    sigma_c = np.einsum('ijkl, kl', C_global, e_c)
    return sigma_c

def calculate_Delta_Eel(P_BAR, ignoreLastTerm, d_bar, eigen, omega):
    """
    Calculates the total change in elastic energy (Delta Eel).
    """
    # Analytical baseline terms
    Delta_Eel = (2*P_BAR**2)/ (d_bar**2 + P_BAR**2)
    Delta_Eel = Delta_Eel + math.log(d_bar**2/(d_bar**2 + P_BAR**2))
    Delta_Eel = Delta_Eel + 2*h_bar*omega*d_bar*math.log((d_bar**2 + (h_tilda-P_BAR)**2)/(d_bar**2 + (h_tilda+P_BAR)**2))

    sigma_xy = []
    x_vals = np.linspace(0, P_BAR, 25)

    for i in x_vals:
        sigma_xy_p1 =  calculate_sigma_c(i, precipitate= 1, d_bar=d_bar, eigen=eigen, omega=omega)[0][1]
        sigma_xy_p2 =  calculate_sigma_c(i, precipitate= 2, d_bar=d_bar, eigen=eigen, omega=omega)[0][1]
        sigma_xy.append(sigma_xy_p1 + sigma_xy_p2)

    # Subtract the complex interaction integral
    Delta_Eel_With_LastTerm = Delta_Eel - (8*h_bar*np.pi*(1-nu)/mu)*gauss_quad_approx_AUC(0, P_BAR, 25, d_bar, eigen, omega)
    
    return 0.5*Delta_Eel, 0.5*Delta_Eel_With_LastTerm

def gauss_quad_approx_AUC(a, b, n, d_bar, eigen, omega):
    """
    Approximates the Area Under the Curve (AUC) for the interaction energy using n-point Gauss-Legendre quadrature.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    approximation = 0
    
    # Map standard nodes from [-1, 1] to integration interval [a, b]
    nodes = 0.5*(nodes + 1)*(b - a) + a

    for i in range(n):
        sigma_xy_p1 = calculate_sigma_c(nodes[i], precipitate= 1, d_bar=d_bar, eigen=eigen, omega=omega)[0][1]
        sigma_xy_p2 = calculate_sigma_c(nodes[i], precipitate= 2, d_bar=d_bar, eigen=eigen, omega=omega)[0][1]
        approximation = approximation + weights[i]*(sigma_xy_p1 + sigma_xy_p2)
        
    return approximation*0.5*(b - a)

# ==========================================
# Main Execution Loop
# ==========================================

def run_simulations():
    """
    Runs the parameter sweeps across multiple CPU cores and streams the raw array data directly to a CSV file.
    """
    t = 0
    num_cores = multiprocessing.cpu_count()
    print(f"Initializing multiprocessing pool with {num_cores} cores...")
    pool = multiprocessing.Pool(processes=num_cores)

    # Open CSV for data export
    with open("Data.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['d_bar', 'eigen', 'omega', 'P_BAR', 'Delta_Eel', 'Delta_Eel_no_last_term'])

        # Sweep through parameter space
        for i, d in enumerate(d_bar_values):
            for j, e in enumerate(eigen_values):
                for k, o in enumerate(omega_values):
                    P_BAR = np.linspace(0, 25, 250)
                    Delta_Eel = np.zeros(len(P_BAR))
                    Delta_Eel_no_last_term = np.zeros(len(P_BAR))

                    t += 1
                    total_simulations = len(d_bar_values) * len(omega_values) * len(eigen_values)
                    print(f"\n--- Running Combination {t}/{total_simulations} ---")
                    print(f"d_bar={d}, eigen={e}, omega={math.degrees(o):.1f}°")

                    args_list = [(p, d, e, o) for p in P_BAR]
                    
                    # Distribute P_BAR array calculations across CPU pool
                    results = list(tqdm(pool.imap(worker_wrapper, args_list, chunksize=5), total=len(P_BAR), desc='Parallel compute'))

                    # Reassemble results
                    for l, (no_last, with_last) in enumerate(results):
                        Delta_Eel_no_last_term[l] = no_last
                        Delta_Eel[l] = with_last

                    # Write all arrays directly to CSV for this parameter combination
                    writer.writerow([d, e, o, P_BAR, np.array(Delta_Eel), np.array(Delta_Eel_no_last_term)])

    # Clean up multiprocessing pool
    pool.close()
    pool.join()
    
    # Corrected the filename in this print statement to match the open() call above
    print("\n✅ All simulations completed and saved to Data.csv")

if __name__ == '__main__':
    run_simulations()