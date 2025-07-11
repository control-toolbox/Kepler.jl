using OptimalControl
using CTBase
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
using ForwardDiff
using MINPACK
using LinearAlgebra

# ===================== Données du problème =====================

μ = 5165.8620912
ε = 1.0
tf_fixed = 5000.0
α = 0.9
q0 = [7000.0, 0.0]
v0 = [0.0, sqrt(μ/7000.0)]  
θf = π/3
qf = [42165.0 * cos(θf), 42165.0 * sin(θf)]
vf = [-sqrt(μ/42165.0) * sin(θf), sqrt(μ/42165.0) * cos(θf)]

x0_cart = [q0[1], q0[2], v0[1], v0[2]]
xf_cart = [qf[1], qf[2], vf[1], vf[2]]

x_cart(t) = x0_cart + (xf_cart - x0_cart) * t / tf_fixed
u_cart(t) = [0.01, 0.01]
nlp_init = (state = x_cart, control = u_cart)

# ===================== OCP cartésien 2D =====================

@def ocp_cart begin
    t ∈ [0, tf_fixed], time
    x ∈ R⁴, state    
    u ∈ R², control  

    x(0) == x0_cart
    x(tf_fixed) == xf_cart

    x_pos = x₁(t)
    y_pos = x₂(t)
    vx = x₃(t)
    vy = x₄(t)

    ux = u₁(t)
    uy = u₂(t)

    r = sqrt(x_pos^2 + y_pos^2)

    # Dynamique cartésienne (Newton + contrôle)
    ax_grav = -μ * x_pos / r^3
    ay_grav = -μ * y_pos / r^3

    ẋ(t) == [vx, vy, ax_grav + ux, ay_grav + uy]

    # Contraintes
    u₁(t)^2 + u₂(t)^2 ≤ ε^2
    x₁(t)^2 + x₂(t)^2 ≥ 1000.0^2  # Distance minimale

    ∫(u₁(t)^2 + u₂(t)^2) → min
end

# ===================== Résolution NLP =====================
println("Résolution NLP cartésienne...")
nlp_sol_cart = OptimalControl.solve(ocp_cart; init=nlp_init, grid_size=60, print_level=1)
plot(nlp_sol_cart, vars=:x, title="États cartésiens")

# ===================== Fonctions pour le shooting cartésien =====================

function F0_cart(x)
    x_pos, y_pos, vx, vy = x
    r = sqrt(x_pos^2 + y_pos^2)
    ax_grav = -μ * x_pos / r^3
    ay_grav = -μ * y_pos / r^3
    F = zeros(eltype(x), 4)
    F[1] = vx
    F[2] = vy
    F[3] = ax_grav
    F[4] = ay_grav
    return F
end

function F1_cart(x)
    F = zeros(eltype(x), 4)
    F[3] = 1.0  # Contrôle ux agit sur acceleration x
    return F
end

function F2_cart(x)
    F = zeros(eltype(x), 4)
    F[4] = 1.0  # Contrôle uy agit sur acceleration y
    return F
end

function u_cart_optimal(x, p)
    H1 = p' * F1_cart(x)  # ∂H/∂ux
    H2 = p' * F2_cart(x)  # ∂H/∂uy
    u = [H1, H2]
    u_norm = sqrt(u[1]^2 + u[2]^2)
    if u_norm > 1e-12
        u = u / u_norm  # Normalisation pour ||u|| ≤ 1
    end
    return u
end

# Flow pour le shooting cartésien
fr_cart = Flow(ocp_cart, u_cart_optimal; alg=Rodas5())

function shoot_cart(ξ::Vector{T}) where {T}
    p0 = ξ
    xf_computed, pf = fr_cart(0, x0_cart, p0, tf_fixed)
    s = zeros(T, 4)
    s[1:4] = xf_computed[1:4] - xf_cart[1:4]
    return s
end

# ===================== Initialisation du shooting =====================

p0_init_cart = nlp_sol_cart.costate(0)
ξ_cart = p0_init_cart

# Fonctions pour le solveur
jshoot_cart(ξ) = ForwardDiff.jacobian(shoot_cart, ξ)
shoot_cart!(s, ξ) = (s[:] = shoot_cart(ξ); nothing)
jshoot_cart!(js, ξ) = (js[:] = jshoot_cart(ξ); nothing)

# ===================== Résolution du shooting =====================
println("Résolution du problème de tir cartésien...")
bvp_sol_cart = fsolve(shoot_cart!, jshoot_cart!, ξ_cart; show_trace=true)
println("Solution BVP cartésienne: ", bvp_sol_cart)

# Extraction
p0_optimal_cart = bvp_sol_cart.x
tf_optimal = tf_fixed

# ===================== Tracé de la trajectoire =====================

# CORRECTION: Utiliser OrdinaryDiffEq directement au lieu de Flow
println("Intégration de la trajectoire optimale...")

# Définir le système d'équations différentielles complet (état + coétat)
function hamiltonian_system!(du, u, p, t)
    # u = [x, y, vx, vy, px, py, pvx, pvy]
    x_state = u[1:4]
    p_costate = u[5:8]
    
    # Calcul du contrôle optimal
    u_ctrl = u_cart_optimal(x_state, p_costate)
    
    # Dynamique de l'état
    x_pos, y_pos, vx, vy = x_state
    r = sqrt(x_pos^2 + y_pos^2)
    ax_grav = -μ * x_pos / r^3
    ay_grav = -μ * y_pos / r^3
    
    du[1] = vx
    du[2] = vy
    du[3] = ax_grav + u_ctrl[1]
    du[4] = ay_grav + u_ctrl[2]
    
    # Dynamique du coétat (simplifiée, à adapter selon le Hamiltonien)
    # Pour l'instant, on suppose que le coétat évolue lentement
    px, py, pvx, pvy = p_costate
    
    # ∂H/∂x = -ṗx, etc.
    # Calcul approximatif des dérivées du Hamiltonien
    r3 = r^3
    r5 = r^5
    
    # Gradient de l'accélération gravitationnelle
    ∂ax_∂x = -μ/r3 + 3*μ*x_pos^2/r5
    ∂ax_∂y = 3*μ*x_pos*y_pos/r5
    ∂ay_∂x = 3*μ*y_pos*x_pos/r5
    ∂ay_∂y = -μ/r3 + 3*μ*y_pos^2/r5
    
    du[5] = -pvx * ∂ax_∂x - pvy * ∂ay_∂x  # -∂H/∂x
    du[6] = -pvx * ∂ax_∂y - pvy * ∂ay_∂y  # -∂H/∂y
    du[7] = -px * ∂ax_∂x - py * ∂ay_∂x  # -∂H/∂vx
    du[8] = -px * ∂ax_∂y - py * ∂ay_∂y  # -∂H/∂vy
end

# Problème d'ODE avec état et coétat
prob_cart = ODEProblem(hamiltonian_system!, [x0_cart; p0_init_cart], (0.0, tf_optimal))

# Résolution
sol_cart = solve(prob_cart, Rodas5())

# Extraction des états
t_shoot_cart = sol_cart.t
x_shoot = sol_cart[1, :]
y_shoot = sol_cart[2, :]
vx_shoot = sol_cart[3, :]
vy_shoot = sol_cart[4, :]
p1_shoot = sol_cart[5, :]
p2_shoot = sol_cart[6, :]
pvx_shoot = sol_cart[7, :]
pvy_shoot = sol_cart[8, :]
N_shoot_cart = length(t_shoot_cart)

# ===================== Tracés et animation =====================

# Tracé statique
plt_traj_cart = plot(x_shoot, y_shoot, 
    title="Transfert orbital 2D - Shooting cartésien", 
    xlabel="x (km)", ylabel="y (km)", 
    legend=false, aspect_ratio=:equal,
    linewidth=2, color=:blue)

# Points initial et final
scatter!(plt_traj_cart, [q0[1]], [q0[2]], color=:green, markersize=8, label="Initial")
scatter!(plt_traj_cart, [qf[1]], [qf[2]], color=:red, markersize=8, label="Final")

# Cercles d'orbite
θ = 0:0.1:2π
r_init = norm(q0)
r_final = norm(qf)
plot!(plt_traj_cart, r_init .* cos.(θ), r_init .* sin.(θ), linestyle=:dash, color=:green, alpha=0.5)
plot!(plt_traj_cart, r_final .* cos.(θ), r_final .* sin.(θ), linestyle=:dash, color=:red, alpha=0.5)

display(plt_traj_cart)

# Tracé des vitesses
plt_vel = plot(t_shoot_cart, vx_shoot, label="vx", linewidth=2)
plot!(plt_vel, t_shoot_cart, vy_shoot, label="vy", linewidth=2)
plot!(plt_vel, title="Vitesses", xlabel="Temps (s)", ylabel="Vitesse (km/s)")
display(plt_vel)

# Animation
println("Création de l'animation cartésienne...")
plt_anim_cart = plot(; xlim=(minimum(x_shoot)*1.1, maximum(x_shoot)*1.1), 
                       ylim=(minimum(y_shoot)*1.1, maximum(y_shoot)*1.1), 
                       title="Transfert orbital 2D - Shooting cartésien", 
                       xlabel="x (km)", ylabel="y (km)",
                       legend=false, aspect_ratio=:equal)

# Cercles de référence
plot!(plt_anim_cart, r_init .* cos.(θ), r_init .* sin.(θ), linestyle=:dash, color=:green, alpha=0.3)
plot!(plt_anim_cart, r_final .* cos.(θ), r_final .* sin.(θ), linestyle=:dash, color=:red, alpha=0.3)

# Points initial et final
scatter!(plt_anim_cart, [q0[1]], [q0[2]], color=:green, markersize=8)
scatter!(plt_anim_cart, [qf[1]], [qf[2]], color=:red, markersize=8)

# Animation
@gif for i in 1:N_shoot_cart
    push!(plt_anim_cart, x_shoot[i], y_shoot[i])
end every N_shoot_cart ÷ min(N_shoot_cart, 100)

println("Animation cartésienne créée !")

# ===================== Résultats =====================
println("\n========== RÉSULTATS CARTÉSIENS ==========")
println("Temps de transfert: ", tf_fixed, " s")
println("Position initiale: ", q0)
println("Position finale: ", qf)
println("Vitesse initiale: ", v0)
println("Vitesse finale: ", vf)
println("Convergence BVP: ", bvp_sol_cart.converged)