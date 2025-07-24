using OptimalControl
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
using MINPACK
using ForwardDiff
using LinearAlgebra


Tmax = 60                                  # Maximum thrust in Newtons
cTmax = 3600^2 / 1e6; T = Tmax * cTmax     # Conversion from Newtons to kg x Mm / h²
mass0 = 1500                               # Initial mass of the spacecraft
β = 1.42e-02                               # Engine specific impulsion
μ = 5165.8620912                           # Earth gravitation constant
P0 = 11.625                                # Initial semilatus rectum
ex0, ey0 = 0.75, 0                         # Initial eccentricity
hx0, hy0 = 6.12e-2, 0                      # Initial ascending node and inclination
L0 = π                                     # Initial longitude
Pf = 42.165                                # Final semilatus rectum
exf, eyf = 0, 0                            # Final eccentricity
hxf, hyf = 0, 0                            # Final ascending node and inclination
ε = 1e-1                             # Regularization parameter for logarithmic barrier
tf= 15

asqrt(x; ε=1e-9) = sqrt(sqrt(x^2 + ε^2))  # sqrt lissée pour AD
function F0(x)
    P, ex, ey, hx, hy, L = x
    pdm = asqrt(P / μ)
    cl = cos(L)
    sl = sin(L)
    w = 1 + ex * cl + ey * sl
    F = zeros(eltype(x), 6) # Use eltype to allow overloading for AD
    F[6] = w^2 / (P * pdm)
    return F
end

function F1(x)
    P, ex, ey, hx, hy, L = x
    pdm = asqrt(P / μ)
    cl = cos(L)
    sl = sin(L)
    F = zeros(eltype(x), 6)
    F[2] = pdm *   sl
    F[3] = pdm * (-cl)
    return F
end

function F2(x)
    P, ex, ey, hx, hy, L = x
    pdm = asqrt(P / μ)
    cl = cos(L)
    sl = sin(L)
    w = 1 + ex * cl + ey * sl
    F = zeros(eltype(x), 6)
    F[1] = pdm * 2 * P / w
    F[2] = pdm * (cl + (ex + cl) / w)
    F[3] = pdm * (sl + (ey + sl) / w)
    return F
end

function F3(x)
    P, ex, ey, hx, hy, L = x
    pdm = asqrt(P / μ)
    cl = cos(L)
    sl = sin(L)
    w = 1 + ex * cl + ey * sl
    pdmw = pdm / w
    zz = hx * sl - hy * cl
    uh = (1 + hx^2 + hy^2) / 2
    F = zeros(eltype(x), 6)
    F[2] = pdmw * (-zz * ey)
    F[3] = pdmw *   zz * ex
    F[4] = pdmw *   uh * cl
    F[5] = pdmw *   uh * sl
    F[6] = pdmw *   zz
    return F
end

Lf = 3π                                      # Estimation of final longitude
x0 = [P0, ex0, ey0, hx0, hy0, L0]            # Initial state
xf = [Pf, exf, eyf, hxf, hyf, Lf]            # Final state
x(t) = x0 + (xf - x0) * t / tf               # Linear interpolation
u = [0.1, 0.5, 0.]                        # Initial guess for the control
nlp_init = (state=x, control=u) # Initial guess for the NLP

function min_tf()
    @def ocp begin
        tf ∈ R, variable
        t ∈ [0, tf], time
        x = (P, ex, ey, hx, hy, L) ∈ R⁶, state
        u ∈ R³, control
        x(0) == x0
        x[1:5](tf) == xf[1:5]
        mass = mass0 - β * T * t
        mass ≥ 0.1 * mass0  # Contrainte de masse positive
        ẋ(t) == F0(x(t)) + T / mass * (u₁(t) * F1(x(t)) + u₂(t) * F2(x(t)) + u₃(t) * F3(x(t)))
        u₁(t)^2 + u₂(t)^2 + u₃(t)^2 ≤ 1
        tf → min
    end
    return ocp
end

function min_conso(e)
    @def ocp begin
        t ∈ [0, tf], time
        x = (P, ex, ey, hx, hy, L) ∈ R⁶, state
        u ∈ R³, control
        x(0) == x0
        x[1:5](tf) == xf[1:5]
        mass = mass0 - β * T * t
        mass ≥ 0.1 * mass0  
        u_norm = sqrt(u₁(t)^2 + u₂(t)^2 + u₃(t)^2)
        ẋ(t) == F0(x(t)) + T / mass * (u₁(t) * F1(x(t)) + u₂(t) * F2(x(t)) + u₃(t) * F3(x(t)))
        1e-4 ≤ u_norm ≤ 0.99  
        ∫(u_norm - e * log(u_norm) - e * log(1.0 - u_norm)) → min
    end
    return ocp
end

ocp1 = min_tf()
nlp_init_tf = (state=x, control=u, variable=tf)
try
    sol_tf = solve(ocp1; init=nlp_init_tf, grid_size=100, print_level=5)
    println("✓ Problème de temps minimal résolu !")
    
    # Extraire le temps optimal et ajuster
    tf_optimal = variable(sol_tf)
    println("Temps optimal trouvé: ", tf_optimal, " h")
    
    # Ajuster le temps final
    global tf = tf_optimal
    
    
    # Initialisation à partir de la solution de temps minimal
    nlp_init_reg = (state = state(sol_tf), control = control(sol_tf))
    
    eps_values = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    current_init = nlp_init_reg
    current_sol = sol_tf
    
    for (i, e) in enumerate(eps_values)
        println("  Résolution avec ε = ", e)
        ocp_reg = min_conso(e)
        try
            current_sol = solve(ocp_reg; init=current_init, grid_size=200, print_level=3)
            current_init = current_sol  # Utiliser cette solution pour l'itération suivante
            println(" Convergé avec ε = ", e)
        catch err
            println(" Échec avec ε = ", e, " : ", err)
            break
        end
    end
    
    # Affichage final
    if current_sol !== nothing
        println("\nSolution finale obtenue !")
        
        # Tracés
        plt1 = plot(current_sol; control=:norm, size=(800, 300), layout=:group)
        display(plt1)
        
        t_grid = time_grid(current_sol)
        u_vals = control(current_sol)
        plt2 = plot(t_grid, norm.(u_vals.(t_grid)); 
                    label="‖u‖", xlabel="t (h)", ylabel="Norme du contrôle",
                    title="Évolution de la norme du contrôle")
        display(plt2)
        
        println("Temps de transfert final: ", tf, " h")
        println("Norme max du contrôle: ", maximum(norm.(u_vals.(t_grid))))
    end
    
catch err
    println("Échec du problème de temps minimal: ", err)    
    ocp_direct = min_conso(1e-2)  
    nlp_init_fallback = (state=x, control=u)
    
    try
        sol_direct = solve(ocp_direct; init=nlp_init_fallback, grid_size=100, print_level=5)
        println("Solution directe trouvée !")
        plot(sol_direct; control=:norm, size=(800, 300), layout=:group)
    catch err2
        println("Échec : ", err2)
    end
end