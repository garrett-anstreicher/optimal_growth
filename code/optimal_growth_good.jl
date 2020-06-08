using Parameters, Plots

@with_kw struct Primitives
    β::Float64 = .99
    θ::Float64 = .36
    δ::Float64 = .025
    k_grid::Array{Float64,1} = collect(range(0.01, length=1800, stop=45))
    nk::Int64 = length(k_grid)
end

@with_kw struct States
    Z::Array{Float64,2} = [1.25 .2] #[Zg Zb]
    nz::Int64 = length(Z)
    Π::Array{Float64,2} = [.977 .023; .074 .926] #[Pgg Pgb; Pbg Pbb]
end

mutable struct Results # mutable makes it able to update the contents
    val_func::Array{Float64,2}
    pol_func::Array{Float64,2}
end

function Solve_model()
    prim = Primitives()
    st = States()
    val_func, pol_func = zeros(prim.nk,st.nz), zeros(prim.nk,st.nz)
    res = Results(val_func, pol_func)

    error, n = 100, 0
    while error > .0001
        @unpack val_func, pol_func = res
        n+=1 # counter of iter

        v_next, p_next = Bellman(prim, res, st)
        error = sum(abs.(v_next - val_func))+sum(abs.(p_next - pol_func))
        res = Results(v_next, p_next)
    end

    println("Value function converged in ", n, " iteration.")
    return prim, res, st
end

function Bellman(prim::Primitives, res::Results, st::States)
    @unpack β, δ, θ, nk, k_grid = prim
    @unpack Z, nz, Π = st
    @unpack val_func, pol_func = res

    v_next = zeros(nk,nz)
    p_next = zeros(nk,nz)
    for (i_z, z) in enumerate(Z)
        Pr = Π[i_z,:]

        for (i_k, k_now) in enumerate(k_grid)
            max_util = log(0)
            k_next = log(0)
            budget = z*k_now^θ + (1-δ)*k_now

            for (i_kp, k_temp) in enumerate(k_grid)
                c_temp = budget - k_temp

                if c_temp > 0
                    val = log(c_temp) + β*(val_func[i_kp,1]*Pr[1] + val_func[i_kp,2]*Pr[2])
                else
                    val = log(0)
                end

                if val > max_util
                    max_util = val
                    k_next = k_temp
                end

            end

            v_next[i_k,i_z] = max_util
            p_next[i_k,i_z] = k_next
        end
    end
    return v_next, p_next
end

@time prim, res = Solve_model()

Plots.plot(prim.k_grid, res.val_func[:,1], label="Good State")
Plots.plot!(prim.k_grid, res.val_func[:,2], label="Bad State")
Plots.plot!(title="Value Functions", legend=:bottomright, xlabel="Capital")
savefig("/Volumes/GoogleDrive/My Drive/Coding/Julia/PS/q7_plot.png")

#############
