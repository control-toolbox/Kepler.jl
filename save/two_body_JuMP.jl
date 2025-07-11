using JuMP
using Ipopt
using Plots
using LinearAlgebra

# ===================== Données du problème =====================
μ = 5165.8620912
ε = 1.0
tf_fixed = 5000.0
N = 100  # Nombre de points de discrétisation



# Transfert d'Hohmann classique : orbite circulaire basse → orbite circulaire haute

q0 = [7000.0, 0.0]
v0 = [0.0, sqrt(μ/7000.0)]   

qf = [42165.0, 0.0]  
vf = [0.0, sqrt(μ/42165.0)]   
θf = π/2 
qf = [42165.0 * cos(θf), 42165.0 * sin(θf)]  # = [-42165, 0]
vf = [-sqrt(μ/42165.0) * sin(θf), sqrt(μ/42165.0) * cos(θf)]  # = [0, -sqrt(μ/42165)]

function cartesian_to_equinoctial_2d(q::Vector, v::Vector, μ)
    r = norm(q)
    v2 = dot(v, v)
    h = q[1]*v[2] - q[2]*v[1]
    e_vec = (v2 - μ/r) * q - dot(q, v) * v
    e_vec /= μ
    P = h^2 / μ
    ex = e_vec[1]
    ey = e_vec[2]
    θ = atan(q[2], q[1])
    L = θ
    return [P, ex, ey, L]
end

x0 = cartesian_to_equinoctial_2d(q0, v0, μ)
xf = cartesian_to_equinoctial_2d(qf, vf, μ)

# Grille de temps
t_grid = range(0, tf_fixed, length=N)
dt = tf_fixed / (N-1)

# ===================== Modèle JuMP =====================
model = JuMP.Model(Ipopt.Optimizer)  # Spécifier JuMP.Model explicitement
JuMP.set_silent(model)  # Utiliser JuMP.set_silent

# Variables d'état [P, ex, ey, L] aux N points de temps
JuMP.@variable(model, P[1:N])
JuMP.@variable(model, ex[1:N])
JuMP.@variable(model, ey[1:N])
JuMP.@variable(model, L[1:N])

# Variables de contrôle [ur, ut] aux N points de temps
JuMP.@variable(model, ur[1:N])
JuMP.@variable(model, ut[1:N])

# ===================== Contraintes =====================

# Conditions initiales
JuMP.@constraint(model, P[1] == x0[1])
JuMP.@constraint(model, ex[1] == x0[2])
JuMP.@constraint(model, ey[1] == x0[3])
JuMP.@constraint(model, L[1] == x0[4])

# Conditions finales
JuMP.@constraint(model, P[N] == xf[1])
JuMP.@constraint(model, ex[N] == xf[2])
JuMP.@constraint(model, ey[N] == xf[3])
JuMP.@constraint(model, L[N] == xf[4])

# Contraintes sur les contrôles
for i in 1:N
    JuMP.@constraint(model, ur[i]^2 + ut[i]^2 <= ε^2)
end

# Contraintes physiques
for i in 1:N
    JuMP.@constraint(model, P[i] >= 100.0)  # P > 0
end

# Dynamique (équations de Gauss discrétisées avec Euler)
for i in 1:N-1
    # Variables intermédiaires au point i
    w_i = 1 + ex[i] * cos(L[i]) + ey[i] * sin(L[i])
    sqrt_p_mu_i = sqrt(P[i] / μ)
    sqrt_mu_p_i = sqrt(μ / P[i])
    
    # Équations de Gauss
    dP_dt = 2 * P[i] * sqrt_p_mu_i * ut[i]
    dex_dt = sqrt_p_mu_i * (sin(L[i]) * ur[i] + ((w_i + 1) * cos(L[i]) + ex[i]) * ut[i] / w_i)
    dey_dt = sqrt_p_mu_i * (-cos(L[i]) * ur[i] + ((w_i + 1) * sin(L[i]) + ey[i]) * ut[i] / w_i)
    dL_dt = sqrt_mu_p_i * (w_i / P[i])^2
    
    # Discrétisation d'Euler
    JuMP.@constraint(model, P[i+1] == P[i] + dt * dP_dt)
    JuMP.@constraint(model, ex[i+1] == ex[i] + dt * dex_dt)
    JuMP.@constraint(model, ey[i+1] == ey[i] + dt * dey_dt)
    JuMP.@constraint(model, L[i+1] == L[i] + dt * dL_dt)
end

# ===================== Fonction objectif =====================
JuMP.@objective(model, Min, sum(ur[i]^2 + ut[i]^2 for i in 1:N) * dt)

# ===================== Initialisation =====================
for i in 1:N
    t_frac = (i-1) / (N-1)
    JuMP.set_start_value(P[i], x0[1] + t_frac * (xf[1] - x0[1]))
    JuMP.set_start_value(ex[i], x0[2] + t_frac * (xf[2] - x0[2]))
    JuMP.set_start_value(ey[i], x0[3] + t_frac * (xf[3] - x0[3]))
    JuMP.set_start_value(L[i], x0[4] + t_frac * (xf[4] - x0[4]))
    JuMP.set_start_value(ur[i], 0.01)
    JuMP.set_start_value(ut[i], 0.01)
end

# ===================== Résolution =====================
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
L_opt = JuMP.value.(L)
ur_opt = JuMP.value.(ur)
ut_opt = JuMP.value.(ut)

# Conversion en coordonnées cartésiennes
q1_jump = P_opt .* cos.(L_opt) ./ (1 .+ ex_opt .* cos.(L_opt) .+ ey_opt .* sin.(L_opt))
q2_jump = P_opt .* sin.(L_opt) ./ (1 .+ ex_opt .* cos.(L_opt) .+ ey_opt .* sin.(L_opt))

# ===================== Tracés =====================

# Tracé statique de la trajectoire
plt_traj = plot(q1_jump, q2_jump, 
    title="Transfert orbital 2D - Solution JuMP", 
    xlabel="x (km)", ylabel="y (km)", 
    legend=false, aspect_ratio=:equal,
    linewidth=2, color=:blue)

# Points initial et final
scatter!(plt_traj, [q0[1]], [q0[2]], color=:green, markersize=8, label="Initial")
scatter!(plt_traj, [qf[1]], [qf[2]], color=:red, markersize=8, label="Final")

# Cercles d'orbite
θ = 0:0.1:2π
r_init = norm(q0)
r_final = norm(qf)
plot!(plt_traj, r_init .* cos.(θ), r_init .* sin.(θ), linestyle=:dash, color=:green, alpha=0.5)
plot!(plt_traj, r_final .* cos.(θ), r_final .* sin.(θ), linestyle=:dash, color=:red, alpha=0.5)

display(plt_traj)

# Tracé des contrôles
plt_control = plot(t_grid, ur_opt, label="ur (radial)", linewidth=2)
plot!(plt_control, t_grid, ut_opt, label="ut (tangentiel)", linewidth=2)
plot!(plt_control, title="Contrôles optimaux", xlabel="Temps (s)", ylabel="Contrôle")
display(plt_control)

# Tracé des états
plt_states = plot(t_grid, P_opt, label="P", linewidth=2)
plot!(plt_states, t_grid, ex_opt, label="ex", linewidth=2)
plot!(plt_states, t_grid, ey_opt, label="ey", linewidth=2)
plot!(plt_states, title="États équinoctiaux", xlabel="Temps (s)")
display(plt_states)

# Animation
println("Création de l'animation...")
plt_anim = plot(; xlim=(minimum(q1_jump)*1.1, maximum(q1_jump)*1.1), 
                  ylim=(minimum(q2_jump)*1.1, maximum(q2_jump)*1.1), 
                  title="Transfert orbital 2D - Animation JuMP", 
                  xlabel="x (km)", ylabel="y (km)",
                  legend=false, aspect_ratio=:equal)

# Cercles de référence
plot!(plt_anim, r_init .* cos.(θ), r_init .* sin.(θ), linestyle=:dash, color=:green, alpha=0.3)
plot!(plt_anim, r_final .* cos.(θ), r_final .* sin.(θ), linestyle=:dash, color=:red, alpha=0.3)

# Points initial et final
scatter!(plt_anim, [q0[1]], [q0[2]], color=:green, markersize=8)
scatter!(plt_anim, [qf[1]], [qf[2]], color=:red, markersize=8)

# Animation
@gif for i in 1:N
    push!(plt_anim, q1_jump[i], q2_jump[i])
end every max(1, N ÷ 50)

println("Animation créée !")

# ===================== Résultats =====================
println("\n========== RÉSULTATS JUMP ==========")
println("Statut: ", JuMP.termination_status(model))
println("Temps de transfert: ", tf_fixed, " s")
println("Fonction objectif: ", JuMP.objective_value(model))
println("Nombre de points: ", N)
println("Position initiale: ", q0)
println("Position finale: ", qf)
println("Norme max du contrôle: ", maximum(sqrt.(ur_opt.^2 .+ ut_opt.^2)))