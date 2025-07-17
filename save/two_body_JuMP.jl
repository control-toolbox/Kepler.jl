using JuMP
using Ipopt
using Plots
using LinearAlgebra

# ===================== Données du problème =====================
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
ε = 1e-1                                   # Regularization parameter for logarithmic barrier
tf = 20

N = 100  # Nombre de points de discrétisation
dt = tf / (N-1)

# ===================== États initiaux et finaux =====================
Lf = 3π                                      # Estimation of final longitude
x0 = [P0, ex0, ey0, hx0, hy0, L0]            # Initial state
xf = [Pf, exf, eyf, hxf, hyf, Lf]            # Final state

# Grille de temps
t_grid = range(0, tf, length=N)

# ===================== Modèle JuMP =====================
model = JuMP.Model(Ipopt.Optimizer)
JuMP.set_silent(model)

# Variables d'état [P, ex, ey, hx, hy, L] aux N points de temps
JuMP.@variable(model, P[1:N])
JuMP.@variable(model, ex[1:N])
JuMP.@variable(model, ey[1:N])
JuMP.@variable(model, hx[1:N])
JuMP.@variable(model, hy[1:N])
JuMP.@variable(model, L[1:N])

# Variables de contrôle [u1, u2, u3] aux N points de temps
JuMP.@variable(model, u1[1:N])
JuMP.@variable(model, u2[1:N])
JuMP.@variable(model, u3[1:N])

# Variable auxiliaire pour la norme du contrôle
JuMP.@variable(model, u_norm[1:N] >= 1e-6)

# ===================== Contraintes =====================

# Conditions initiales
JuMP.@constraint(model, P[1] == x0[1])
JuMP.@constraint(model, ex[1] == x0[2])
JuMP.@constraint(model, ey[1] == x0[3])
JuMP.@constraint(model, hx[1] == x0[4])
JuMP.@constraint(model, hy[1] == x0[5])
JuMP.@constraint(model, L[1] == x0[6])

# Conditions finales (seulement les 5 premiers états)
JuMP.@constraint(model, P[N] == xf[1])
JuMP.@constraint(model, ex[N] == xf[2])
JuMP.@constraint(model, ey[N] == xf[3])
JuMP.@constraint(model, hx[N] == xf[4])
JuMP.@constraint(model, hy[N] == xf[5])

# Contraintes sur la norme du contrôle
for i in 1:N
    JuMP.@constraint(model, u1[i]^2 + u2[i]^2 + u3[i]^2 <= u_norm[i]^2)
    JuMP.@constraint(model, 1e-3 <= u_norm[i]^2 <= 1)
    JuMP.@constraint(model, u_norm[i] >= 1e-6)  # Éviter u_norm = 0 pour le log
    JuMP.@constraint(model, u_norm[i] <= 1.0 - 1e-6)  # Éviter u_norm = 1 pour le log
end

# Dynamique (discrétisée avec Euler) - CORRECTION: calcul direct dans JuMP
for i in 1:N-1
    # Masse au point i
    mass_i = mass0 - β * T * t_grid[i]
    
    # Variables d'état au point i
    P_i = P[i]
    ex_i = ex[i]
    ey_i = ey[i]
    hx_i = hx[i]
    hy_i = hy[i]
    L_i = L[i]
    
    # Variables de contrôle au point i
    u1_i = u1[i]
    u2_i = u2[i]
    u3_i = u3[i]
    
    # Calcul direct des équations de Gauss dans JuMP
    # sqrt lissé pour éviter les problèmes de différentiation
    asqrt_P = sqrt(sqrt(P_i^2 + 1e-18))  # P/μ avec sqrt lissé
    pdm = asqrt_P / sqrt(μ)
    
    cl = cos(L_i)
    sl = sin(L_i)
    w = 1 + ex_i * cl + ey_i * sl
    
    # F0 contribution
    dL_dt_F0 = w^2 / (P_i * pdm)
    
    # F1 contribution
    dex_dt_F1 = pdm * sl
    dey_dt_F1 = pdm * (-cl)
    
    # F2 contribution
    dP_dt_F2 = pdm * 2 * P_i / w
    dex_dt_F2 = pdm * (cl + (ex_i + cl) / w)
    dey_dt_F2 = pdm * (sl + (ey_i + sl) / w)
    
    # F3 contribution
    pdmw = pdm / w
    zz = hx_i * sl - hy_i * cl
    uh = (1 + hx_i^2 + hy_i^2) / 2
    
    dex_dt_F3 = pdmw * (-zz * ey_i)
    dey_dt_F3 = pdmw * zz * ex_i
    dhx_dt_F3 = pdmw * uh * cl
    dhy_dt_F3 = pdmw * uh * sl
    dL_dt_F3 = pdmw * zz
    
    # Dynamique totale: ẋ = F0 + T/mass * (u1*F1 + u2*F2 + u3*F3)
    thrust_factor = T / mass_i
    
    dP_dt = thrust_factor * u2_i * dP_dt_F2
    dex_dt = thrust_factor * (u1_i * dex_dt_F1 + u2_i * dex_dt_F2 + u3_i * dex_dt_F3)
    dey_dt = thrust_factor * (u1_i * dey_dt_F1 + u2_i * dey_dt_F2 + u3_i * dey_dt_F3)
    dhx_dt = thrust_factor * u3_i * dhx_dt_F3
    dhy_dt = thrust_factor * u3_i * dhy_dt_F3
    dL_dt = dL_dt_F0 + thrust_factor * u3_i * dL_dt_F3
    
    # Discrétisation d'Euler
    JuMP.@constraint(model, P[i+1] == P[i] + dt * dP_dt)
    JuMP.@constraint(model, ex[i+1] == ex[i] + dt * dex_dt)
    JuMP.@constraint(model, ey[i+1] == ey[i] + dt * dey_dt)
    JuMP.@constraint(model, hx[i+1] == hx[i] + dt * dhx_dt)
    JuMP.@constraint(model, hy[i+1] == hy[i] + dt * dhy_dt)
    JuMP.@constraint(model, L[i+1] == L[i] + dt * dL_dt)
end

# ===================== Fonction objectif avec barrière logarithmique =====================
# Objectif: ∫(u_norm - ε * (log(u_norm) + log(1 - u_norm))) dt
JuMP.@objective(model, Min, 
    sum(u_norm[i] - ε * (log(u_norm[i]) + log(max(1e-10, 1 - u_norm[i]))) 
        for i in 1:N) * dt)

# ===================== Initialisation =====================
for i in 1:N
    t_frac = (i-1) / (N-1)
    
    # Initialisation des états (interpolation linéaire)
    JuMP.set_start_value(P[i], x0[1] + t_frac * (xf[1] - x0[1]))
    JuMP.set_start_value(ex[i], x0[2] + t_frac * (xf[2] - x0[2]))
    JuMP.set_start_value(ey[i], x0[3] + t_frac * (xf[3] - x0[3]))
    JuMP.set_start_value(hx[i], x0[4] + t_frac * (xf[4] - x0[4]))
    JuMP.set_start_value(hy[i], x0[5] + t_frac * (xf[5] - x0[5]))
    JuMP.set_start_value(L[i], x0[6] + t_frac * (xf[6] - x0[6]))
    
    # Initialisation des contrôles
    JuMP.set_start_value(u1[i], 0.1)
    JuMP.set_start_value(u2[i], 0.5)
    JuMP.set_start_value(u3[i], 0.0)
    JuMP.set_start_value(u_norm[i], 0.5)
end

# ===================== Résolution =====================
println("Résolution du problème JuMP...")
JuMP.optimize!(model)

if JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
    println("Solution trouvée !")
else
    println("Problème de convergence: ", JuMP.termination_status(model))
end

# ===================== Extraction des résultats =====================
P_opt = JuMP.value.(P)
ex_opt = JuMP.value.(ex)
ey_opt = JuMP.value.(ey)
hx_opt = JuMP.value.(hx)
hy_opt = JuMP.value.(hy)
L_opt = JuMP.value.(L)
u1_opt = JuMP.value.(u1)
u2_opt = JuMP.value.(u2)
u3_opt = JuMP.value.(u3)
u_norm_opt = JuMP.value.(u_norm)

# ===================== Tracés =====================

# Tracé des états
plt_states = plot(layout=(2,3), size=(1200, 600))
plot!(plt_states[1], t_grid, P_opt, label="P", title="P", linewidth=2)
plot!(plt_states[2], t_grid, ex_opt, label="ex", title="ex", linewidth=2)
plot!(plt_states[3], t_grid, ey_opt, label="ey", title="ey", linewidth=2)
plot!(plt_states[4], t_grid, hx_opt, label="hx", title="hx", linewidth=2)
plot!(plt_states[5], t_grid, hy_opt, label="hy", title="hy", linewidth=2)
plot!(plt_states[6], t_grid, L_opt, label="L", title="L", linewidth=2)
display(plt_states)

# Tracé des contrôles
plt_control = plot(layout=(2,2), size=(800, 600))
plot!(plt_control[1], t_grid, u1_opt, label="u1", title="u1", linewidth=2)
plot!(plt_control[2], t_grid, u2_opt, label="u2", title="u2", linewidth=2)
plot!(plt_control[3], t_grid, u3_opt, label="u3", title="u3", linewidth=2)
plot!(plt_control[4], t_grid, u_norm_opt, label="||u||", title="||u||", linewidth=2)
display(plt_control)

# ===================== Résultats =====================
println("\n========== RÉSULTATS JUMP ==========")
println("Statut: ", JuMP.termination_status(model))
println("Temps de transfert: ", tf, " h")
println("Fonction objectif: ", JuMP.objective_value(model))
println("Nombre de points: ", N)
println("Norme max du contrôle: ", maximum(u_norm_opt))
println("Norme min du contrôle: ", minimum(u_norm_opt))
println("États initiaux: ", x0)
println("États finaux: ", [P_opt[end], ex_opt[end], ey_opt[end], hx_opt[end], hy_opt[end], L_opt[end]])
println("Erreur finale: ", norm([P_opt[end], ex_opt[end], ey_opt[end], hx_opt[end], hy_opt[end]] - xf[1:5]))

