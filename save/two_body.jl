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
ε = 1e-1                                   # Regularization parameter for logarithmic barrier

asqrt(x; ε=1e-9) = sqrt(sqrt(x^2 + ε^2))
function F0(x)
    P, ex, ey, hx, hy, L = x
    pdm = asqrt(P / μ)
    cl = cos(L)
    sl = sin(L)
    w = 1 + ex * cl + ey * sl
    F = zeros(eltype(x), 6)
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

tf = 15
Lf = 3π
x0 = [P0, ex0, ey0, hx0, hy0, L0]
xf = [Pf, exf, eyf, hxf, hyf, Lf]
x(t) = x0 + (xf - x0) * t / tf
u = [0.1, 0.5, 0.]
nlp_init = (state=x, control=u, variable=tf)

function min_tf()
    @def ocp begin
        tf ∈ R, variable
        t ∈ [0, tf], time
        x = (P, ex, ey, hx, hy, L) ∈ R⁶, state
        u ∈ R³, control
        x(0) == x0
        x[1:5](tf) == xf[1:5]
        mass = mass0 - β * T * t
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
        u_norm = sqrt(u₁(t)^2 + u₂(t)^2 + u₃(t)^2)
        ẋ(t) == F0(x(t)) + T / mass * (u₁(t) * F1(x(t)) + u₂(t) * F2(x(t)) + u₃(t) * F3(x(t)))
        1e-3 ≤ u_norm^2 ≤ 1
        u_g = max(1e-10, 1 - u_norm)
        ∫(u_norm - e * (log(u_norm) + log(u_g))) → min
    end
    return ocp
end

ocp1 = min_tf()
sol = solve(ocp1; init=nlp_init, grid_size=100)
Tmax = 100.0
T = Tmax * cTmax
tf = variable(sol)
ocp = min_conso(ε)
nlp_init = (state = state(sol), control = control(sol))

eps = 1e-1
i = nlp_init
s = sol
ocp = min_conso(eps)
global s = solve(ocp; init=i, grid_size=500)
global i = s


plot(s; label="ε = $eps")

t = time_grid(s)
u = control(s)
plot(t, norm∘u; label="‖u‖", xlabel="t")