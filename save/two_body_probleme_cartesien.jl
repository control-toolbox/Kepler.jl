using JuMP
using Ipopt
using Plots
using LinearAlgebra

# ===================== Données du problème =====================
μ = 5165.8620912
ε = 1.0
tf_fixed = 50000
N = 100

# Transfert d'Hohmann : orbite basse → orbite haute
q0 = [7000.0, 0.0]
v0 = [0.0, sqrt(μ/7000.0)]

θf = π/3  # 90 degrés
qf = [42165.0 * cos(θf), 42165.0 * sin(θf)]  # = [0, 42165]
vf = [-sqrt(μ/42165.0) * sin(θf), sqrt(μ/42165.0) * cos(θf)]  # = [-sqrt(μ/42165), 0]

println("Position initiale: ", q0)
println("Position finale: ", qf)
println("Vitesse initiale: ", v0)
println("Vitesse finale: ", vf)

# Grille de temps
t_grid = range(0, tf_fixed, length=N)
dt = tf_fixed / (N-1)


# ===================== Modèle JuMP avec coordonnées cartésiennes =====================
model = JuMP.Model(Ipopt.Optimizer)
JuMP.set_silent(model)

# Variables d'état cartésiennes [x, y, vx, vy] aux N points de temps
JuMP.@variable(model, x_pos[1:N])
JuMP.@variable(model, y_pos[1:N])
JuMP.@variable(model, vx[1:N])
JuMP.@variable(model, vy[1:N])

# Variables de contrôle [ux, uy] aux N points de temps
JuMP.@variable(model, ux[1:N])
JuMP.@variable(model, uy[1:N])

# ===================== Contraintes =====================

# Conditions initiales
JuMP.@constraint(model, x_pos[1] == q0[1])
JuMP.@constraint(model, y_pos[1] == q0[2])
JuMP.@constraint(model, vx[1] == v0[1])
JuMP.@constraint(model, vy[1] == v0[2])

# Conditions finales
JuMP.@constraint(model, x_pos[N] == qf[1])
JuMP.@constraint(model, y_pos[N] == qf[2])
JuMP.@constraint(model, vx[N] == vf[1])
JuMP.@constraint(model, vy[N] == vf[2])

# Contraintes sur les contrôles
for i in 1:N
    JuMP.@constraint(model, ux[i]^2 + uy[i]^2 <= ε^2)
end

# Contrainte de distance minimale (éviter collision)
for i in 1:N
    JuMP.@constraint(model, x_pos[i]^2 + y_pos[i]^2 >= 1000.0^2)
end

# Dynamique cartésienne (équations de Newton avec gravité + contrôle)
for i in 1:N-1
    # Distance au centre
    r_i = sqrt(x_pos[i]^2 + y_pos[i]^2)
    
    # Accélérations gravitationnelles
    ax_grav = -μ * x_pos[i] / (r_i^3)
    ay_grav = -μ * y_pos[i] / (r_i^3)
    
    # Accélérations totales (gravité + contrôle)
    ax_total = ax_grav + ux[i]
    ay_total = ay_grav + uy[i]
    
    # Intégration d'Euler
    JuMP.@constraint(model, x_pos[i+1] == x_pos[i] + dt * vx[i])
    JuMP.@constraint(model, y_pos[i+1] == y_pos[i] + dt * vy[i])
    JuMP.@constraint(model, vx[i+1] == vx[i] + dt * ax_total)
    JuMP.@constraint(model, vy[i+1] == vy[i] + dt * ay_total)
end

# ===================== Fonction objectif =====================
JuMP.@objective(model, Min, sum(ux[i]^2 + uy[i]^2 for i in 1:N) * dt)

# ===================== Initialisation =====================
for i in 1:N
    t_frac = (i-1) / (N-1)
    
    # Interpolation linéaire pour les positions
    JuMP.set_start_value(x_pos[i], q0[1] + t_frac * (qf[1] - q0[1]))
    JuMP.set_start_value(y_pos[i], q0[2] + t_frac * (qf[2] - q0[2]))
    
    # Interpolation linéaire pour les vitesses
    JuMP.set_start_value(vx[i], v0[1] + t_frac * (vf[1] - v0[1]))
    JuMP.set_start_value(vy[i], v0[2] + t_frac * (vf[2] - v0[2]))
    
    # Contrôles initiaux petits
    JuMP.set_start_value(ux[i], 0.01)
    JuMP.set_start_value(uy[i], 0.01)
end

# ===================== Résolution =====================
println("Résolution avec JuMP (coordonnées cartésiennes)...")
JuMP.optimize!(model)

if JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
    println("Solution trouvée !")
else
    println("Problème de convergence: ", JuMP.termination_status(model))
end

# ===================== Extraction des résultats =====================
x_opt = JuMP.value.(x_pos)
y_opt = JuMP.value.(y_pos)
vx_opt = JuMP.value.(vx)
vy_opt = JuMP.value.(vy)
ux_opt = JuMP.value.(ux)
uy_opt = JuMP.value.(uy)

# ===================== Tracés =====================

# Tracé statique de la trajectoire
plt_traj = plot(x_opt, y_opt, 
    title="Transfert orbital 2D - Coordonnées cartésiennes", 
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
plt_control = plot(t_grid, ux_opt, label="ux", linewidth=2)
plot!(plt_control, t_grid, uy_opt, label="uy", linewidth=2)
plot!(plt_control, title="Contrôles optimaux", xlabel="Temps (s)", ylabel="Contrôle")
display(plt_control)

# Animation
println("Création de l'animation...")
plt_anim = plot(; xlim=(minimum(x_opt)*1.1, maximum(x_opt)*1.1), 
                  ylim=(minimum(y_opt)*1.1, maximum(y_opt)*1.1), 
                  title="Transfert orbital 2D - Animation", 
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
    push!(plt_anim, x_opt[i], y_opt[i])
end every max(1, N ÷ 50)

println("Animation créée !")

# ===================== Résultats =====================
println("\n========== RÉSULTATS CARTÉSIENS ==========")
println("Statut: ", JuMP.termination_status(model))
println("Temps de transfert: ", tf_fixed, " s")
println("Fonction objectif: ", JuMP.objective_value(model))
println("Position initiale: ", q0)
println("Position finale: ", qf)
println("Norme max du contrôle: ", maximum(sqrt.(ux_opt.^2 .+ uy_opt.^2)))