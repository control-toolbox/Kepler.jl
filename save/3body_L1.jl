using OptimalControl
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
#using MINPACK
using ForwardDiff
using LinearAlgebra

ε = 0.01 

## Problem definition

G = 6.61e10-11
m₁ = 5.972e24          # Mass of the Earth
m₂ = 7.348e22          # Mass of the Moon
μ = m₂ / (m₁ + m₂)     # Ratio of primary masses
M = 1000               # Mass of the satellite (kg)
# !!!!!!!!!!!!
γ_max = 1              # Poussée max
T_max = 1              # Maximal thrust
# !!!!!!!!!!!!

# 1st body is located at (-μ, 0)
# 2nd body is located at (1-μ, 0)

# Kepler law speed in circular movement: v² = G * M / d
d_earth = 0.1
d_moon = 0.1

v_earth_circular = sqrt(G * m₁ / d_earth)
v_moon_circular = sqrt(G * m₂ / d_moon)

q0 = [-μ - d_earth, 0.0]   # Initial position (km) (e.g. GEO Earth)
v0 = [0, v_earth_circular]       # Initial velocity (km/s)
qf = [1 - μ + d_moon, 0.0]    # Target position (km) (e.g. GEO Moon)
vf = [0.0, -v_moon_circular]       # Target velocity (km/s)
x0 = vcat(q0, v0)
xf = vcat(qf, vf)

tf_guess = 20000.0          # Time guess in seconds (5.5 hours)

# =============== INITIAL GUESS ===============
x(t) = x0 + (xf - x0) * t / tf_guess
u(t) = [0.01, 0.01] # initial control guess


nlp_init = (state = x, control = u, variable = tf_guess)

# =============== OCP DEFINITION ===============
@def ocp begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x ∈ R⁴, state
    u ∈ R², control
    x(0) == x0
    x(tf) == xf
    r₁₃ = sqrt((x₁(t) + μ)^2 + x₂(t)^2) + 1e-9  # Avoid division by zero
    r₂₃ = sqrt((x₁(t) - 1 + μ)^2 + x₂(t)^2) + 1e-9  # Avoid division by zero
    F0 = [  x₃(t), 
            x₄(t), 
            2*x₄(t) + x₁(t) - ((1 - μ) / r₁₃^3)*(x₁(t) + μ) - (μ / r₂₃^3)*(x₁(t) - 1 + μ),
            -2*x₃(t) + x₂(t) - ((1 - μ) / r₁₃^3)*x₂(t) - (μ / r₂₃^3)*x₂(t) ]
    F1 = [0, 0, 1, 0]
    F2 = [0, 0, 0, 1]
    ẋ(t) == F0 + (T_max/M)*F1*u1(t) + (T_max/M)*F2*u₂(t)
    u₁(t)^2 + u₂(t)^2 ≤ 1
    # Fonction objectif avec régularisation pour éviter les problèmes de domaine
    u_norm = sqrt(u₁(t)^2 + u₂(t)^2) + 1e-12  # Évite log(0)
    u_complement = max(1e-12, 1 - u_norm)  # Évite log(≤0)
    ∫(u_norm - (1-ε)*log(u_norm) - (1-ε)*log(u_complement)) → min
end


# =============== SOLVE ===============

nlp_sol = OptimalControl.solve(ocp; init=nlp_init, grid_size=50, print_level=3)
plot(nlp_sol, vars=:x, title="3-body Controlled Transfer")


