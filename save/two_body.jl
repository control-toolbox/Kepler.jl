using OptimalControl
using CTBase
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
using ForwardDiff
using LinearAlgebra

# Problem definition
μ = 5165.8620912                           # Earth gravitation constant
M = 1500                               # Initial mass of the satelite
ϵ = 1                                   # Poussée max
q0 = [7000.0, 0.0]         # Initial position (km) - 2D
v0 = [0.0, 7.5]            # Initial velocity (km/s) - 2D
qf = [42165.0, 0.0]        # Target position (km) - 2D
vf = [0.0, 3.07]           # Target velocity (km/s) - 2D
x0 = vcat(q0, v0)
xf = vcat(qf, vf)
tf_guess = 20000.0          # Time guess in seconds (5.5 hours)
α = 0.1                    # Paramètre de régularisation barrière logarithmique

# =============== INITIAL GUESS ===============
x(t) = x0 + (xf - x0) * t / tf_guess
u(t) = [0.01, 0.01] # initial control guess - 2D

nlp_init = (state = x, control = u, variable = tf_guess)

# =============== OCP DEFINITION ===============
@def ocp begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x ∈ R⁴, state
    u ∈ R², control
    x(0) == x0
    x(tf) == xf
    r = sqrt(x₁(t)^2 + x₂(t)^2) + 1e-9  # Avoid division by zero
    a1 = -μ * x₁(t) / r^3
    a2 = -μ * x₂(t) / r^3
    ẋ(t) == [x₃(t), x₄(t), a1 + u₁(t), a2 + u₂(t)]
    u₁(t)^2 + u₂(t)^2 ≤ 1

    u_norm = sqrt(u₁(t)^2 + u₂(t)^2) + 1e-12  # Évite log(0)
    u_complement = max(1e-12, 1 - u_norm)  # Évite log(≤0)
    ∫(u_norm - (1-α)*log(u_norm) - (1-α)*log(u_complement)) → min
end

# =============== SOLVE ===============
nlp_sol = OptimalControl.solve(ocp; init=nlp_init, grid_size=50, print_level=3)
plot(nlp_sol, vars=:x, title="2-body Controlled Transfer", control=:norm)

# Affichage des résultats
println("Temps final optimal : ", nlp_sol.variable, " secondes")
println("Statut du solveur : ", nlp_sol.solver.status)

# Plot de la trajectoire dans le plan (x,y)
x_traj = nlp_sol.state.(nlp_sol.times)
positions = [x[1:2] for x in x_traj]
plot([p[1] for p in positions], [p[2] for p in positions], 
     title="Trajectoire orbitale 2D", xlabel="x (km)", ylabel="y (km)",
     label="Trajectoire", linewidth=2)
scatter!([q0[1], qf[1]], [q0[2], qf[2]], 
         label=["Position initiale" "Position finale"], markersize=8)
