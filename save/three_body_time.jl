using OptimalControl
using CTBase
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
using ForwardDiff
using LinearAlgebra

## Problem definition (CR3BP)

μ = 0.0121505856              # Earth-Moon reduced mass ratio
ϵ = 1.0                       # Maximum thrust magnitude (normalized)
q0 = [1.0 - μ - 0.01, 0.0]    # Initial position (near Earth)
v0 = [0.0, 0.0]               # Initial velocity
qf = [μ + 0.05, 0.0]          # Final position (near Moon)
vf = [0.0, 0.0]               # Final velocity
x0 = vcat(q0, v0)
xf = vcat(qf, vf)
tf_guess = 5.0               # Initial guess for final time

# Gradient of the effective potential
function grad_omega(q)
    x, y = q
    r1 = sqrt((x + μ)^2 + y^2) + 1e-6
    r2 = sqrt((x - 1 + μ)^2 + y^2) + 1e-6
    dΩdx = x - (1 - μ)*(x + μ)/r1^3 - μ*(x - 1 + μ)/r2^3
    dΩdy = y - (1 - μ)*y/r1^3 - μ*y/r2^3
    return [dΩdx, dΩdy]
end


x(t) = x0 + (xf - x0) * t / tf_guess
u(t) = [0.01, 0.01] 
nlp_init = (state = x, control = u, variable = tf_guess)

@def ocp begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x = (q1, q2, v1, v2) ∈ R⁴, state
    u ∈ R², control
    x(0) == x0
    x(tf) == xf
    r1 = sqrt((q1(t) + μ)^2 + q2(t)^2) + 1e-6
    r2 = sqrt((q1(t) - 1 + μ)^2 + q2(t)^2) + 1e-6
    dΩ1 = q1(t) - (1 - μ)*(q1(t) + μ)/r1^3 - μ*(q1(t) - 1 + μ)/r2^3
    dΩ2 = q2(t) - (1 - μ)*q2(t)/r1^3 - μ*q2(t)/r2^3
    ẋ(t) == [v1(t), v2(t), 2v2(t) + dΩ1 + ϵ*u₁(t), -2v1(t) + dΩ2 + ϵ*u₂(t)]
    u₁(t)^2 + u₂(t)^2 ≤ 1
    tf → min
end

nlp_sol = OptimalControl.solve(ocp; init=nlp_init, grid_size=100, print_level=3)
plot(nlp_sol, vars=:x, title="CR3BP - Minimum Time Transfer")
