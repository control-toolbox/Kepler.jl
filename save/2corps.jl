using OptimalControl
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
#using MINPACK
using ForwardDiff
using LinearAlgebra

## Problem definition

μ = 5165.8620912                           # Earth gravitation constant
M = 1500                               # Initial mass of the satelite
ϵ = 1                                   #Possé max
q0 = [7000.0, 0.0, 0.0]     # Initial position (km)
v0 = [0.0, 7.5, 0.0]        # Initial velocity (km/s)
qf = [42165.0, 0.0, 0.0]    # Target position (km) (e.g. GEO)
vf = [0.0, 3.07, 0.0]       # Target velocity (km/s)
x0 = vcat(q0, v0)
xf = vcat(qf, vf)
tf_guess = 20000.0          # Time guess in seconds (5.5 hours)
α = 0.01                # Paramètre de régularisation barrière logarithmique



#Fonction who calculate dx
function F0(x::Vector{T}, u::Vector{T}) where T
    q = x[1:3]
    v = x[4:6]
    r3 = norm(q)^3 + 1e-9 
    a_grav = -μ * q / r3
    dx = similar(x)
    dx[1:3] .= v
    dx[4:6] .= a_grav + u
    return dx
end


# =============== INITIAL GUESS ===============
x(t) = x0 + (xf - x0) * t / tf_guess
u(t) = [0.01, 0.01, 0.01] # initial control guess


nlp_init = (state = x, control = u, variable = tf_guess)


# =============== OCP DEFINITION ===============
@def ocp begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x ∈ R⁶, state
    u ∈ R³, control
    x(0) == x0
    x(tf) == xf
    r = sqrt(x₁(t)^2 + x₂(t)^2 + x₃(t)^2) + 1e-9  # Avoid division by zero
    a1 = -μ * x₁(t) / r^3
    a2 = -μ * x₂(t) / r^3
    a3 = -μ * x₃(t) / r^3
    ẋ(t) == [x₄(t), x₅(t), x₆(t), a1 + u₁(t), a2 + u₂(t), a3 + u₃(t)]
    u₁(t)^2 + u₂(t)^2 + u₃(t)^2 ≤ 1
    # Fonction objectif avec régularisation pour éviter les problèmes de domaine
    u_norm = sqrt(u₁(t)^2 + u₂(t)^2 + u₃(t)^2) + 1e-12  # Évite log(0)
    u_complement = max(1e-12, 1 - u_norm)  # Évite log(≤0)
    ∫(u_norm - (1-α)*log(u_norm) - (1-α)*log(u_complement)) → min
end


# =============== SOLVE ===============

nlp_sol = OptimalControl.solve(ocp; init=nlp_init, grid_size=50, print_level=3)
plot(nlp_sol, vars=:x, title="2-body Controlled Transfer")
