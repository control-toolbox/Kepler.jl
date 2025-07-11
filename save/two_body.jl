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
tf_fixed = 5000.0  # Temps plus court pour tester
α = 0.9

q0 = [7000.0, 0.0]
v0 = [0.0, 7.5]
qf = [42165.0, 0.0]
vf = [0.0, 3.07]


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

x(t) = x0 + (xf - x0) * t / tf_fixed
u(t) = [0.01, 0.01]
nlp_init = (state = x, control = u)

# ===================== OCP équinoctial 2D =====================

@def ocp begin
    t ∈ [0, tf_fixed], time
    x ∈ R⁴, state
    u ∈ R², control

    x(0) == x0
    x(tf_fixed) == xf

    P = x₁(t)   
    ex = x₂(t)  
    ey = x₃(t)  
    L = x₄(t)   

    ur = u₁(t)
    ut = u₂(t)

    w = 1 + ex * cos(L) + ey * sin(L)
    sqrt_p_mu = sqrt(P / μ)
    sqrt_mu_p = sqrt(μ / P)

    dP = 2 * P * sqrt_p_mu * ut
    dex = sqrt_p_mu * (sin(L) * ur + ((w + 1) * cos(L) + ex) * ut / w)
    dey = sqrt_p_mu * (-cos(L) * ur + ((w + 1) * sin(L) + ey) * ut / w)
    dL = sqrt_mu_p * (w / P)^2

    ẋ(t) == [dP, dex, dey, dL]

    u₁(t)^2 + u₂(t)^2 ≤ ε^2
    x₁(t) ≥ 100.0

    ∫(u₁(t)^2 + u₂(t)^2) → min  # Fonction objectif simplifiée
end

# ===================== Résolution NLP =====================
nlp_sol = OptimalControl.solve(ocp; init=nlp_init, grid_size=60, print_level=1)
plot(nlp_sol, vars=:x, title="Éléments équinoctiaux 2D")
