#### Written-from-scratch code for stochastic growth model

using Parameters, Plots

@with_kw struct Primitives
    β::Float64 = 0.99
    θ::Float64 = 0.36
    δ::Float64 = 0.025
    k_grid::Array{Float64, 1} = collect(range(0.01, length = 1000, stop = 45.0))
    nk::Int64 = length(k_grid)
    Π::Matrix{Float64} = [0.977 0.074; 0.023 0.926] # Transpose of original Markov transition matrix
    Z::Vector{Float64} = [1.25, 0.2]
end

mutable struct Results
    val_func::Matrix{Float64}
    pol_func::Matrix{Float64}
end

function Solve_model()
    prim = Primitives()
    val_func = reshape(zeros(2*prim.nk),prim.nk,2)
    pol_func = reshape(zeros(2*prim.nk),prim.nk,2)
    res = Results(val_func, pol_func)

    error = 100
    n = 0

    while error > eps()
        n += 1
        v_next = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.val_func))
        res.val_func = v_next

        if mod(n, 5000) == 0 || error < eps()
            println(" ")
            println("******************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("******************************************")
        end
    end
    prim, res
end

function Bellman(prim::Primitives, res::Results)
    @unpack β, θ, δ, nk, k_grid, Π, Z = prim
    v_next = reshape(zeros(2*nk), nk, 2)
    c = reshape(zeros(2*nk), nk , 2)

    for i_k in 1:nk
        k = k_grid[i_k]
        for j in 1:2
            budget = Z[j]*k^θ + (1-δ) * k
            c[:,j] = budget .- k_grid
            c[findall(c[:,j].<0),j] .= 0
        end
        val = log.(c) .+ β .* res.val_func * Π
        v_next[i_k,1] = maximum(val[:,1])
        v_next[i_k,2] = maximum(val[:,2])
        res.pol_func[i_k,1] = k_grid[findall(val[:,1] .== maximum(val[:,1]))][1]
        res.pol_func[i_k,2] = k_grid[findall(val[:,2] .== maximum(val[:,2]))][1]
    end
    v_next
end

@elapsed prim, res = Solve_model() # If loop over policy function space, it takes almost three times longer to compute.

Plots.plot(prim.k_grid, hcat(res.val_func[:,1], res.val_func[:,2]); label = ["Good" "Bad"], legend = :bottomright)
Plots.plot(prim.k_grid, hcat(res.pol_func[:,1], res.pol_func[:,2]); label = ["Good" "Bad"], legend = :bottomright)
