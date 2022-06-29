###Optimal Savings
using Parameters, Plots

#struct to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount factor
    θ::Float64 = 0.36 #production
    δ::Float64 = 0.025 #depreciation
    Π::Matrix{Float64} = [0.977 0.023; 0.074 0.926]
    Z::Vector{Float64} = [1.25; 0.2]
    k_grid::Array{Float64,1} = collect(range(0.01, length = 1000, stop = 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital grid states
    nz::Int64 = length(Z)
end

mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
end

#function to solve the model
function Solve_model()
    #initialize primitives and results
    prim = Primitives()
    @unpack nk, nz = prim #unpack primitive structure
    val_func = zeros(nk, nz)
    pol_func = zeros(nk, nz)
    res = Results(val_func, pol_func)

    error, n = 100, 0
    while error > eps() #loop until convergence
        n += 1

        v_next = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.val_func))
        res.val_func = v_next

        #println("Current error: ", error)
        if mod(n, 5000) == 0 || error <eps()
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("*************************************************")
        end
    end
    prim, res
end

#Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, Π, Z, nk, nz, k_grid = prim #unpack primitive structure
    v_next = zeros(nk, nz) #preallocate next guess of value function

    for (i_z, z) in enumerate(Z)
        Pr = Π[i_z, :]'

        for (i_k, k) in enumerate(k_grid)
            max_util = -Inf
            k_next = -Inf
            # k = k_grid[i_k] #value of capital
            budget = z*k^θ + (1-δ)*k #budget

            for (i_kp, kp) in enumerate(k_grid) ###loop over choices of k'
                # kp = k_grid[i_kp] #value of k'
                c = budget - kp #consumption
                if c > 0 #check if positive
                    val = log(c) .+ β * (res.val_func[i_kp, 1] * Pr[1] + res.val_func[i_kp, 2] * Pr[2])
                    if val > max_util
                        max_util = val
                        k_next = kp
                    end
                end
            end
            v_next[i_k, i_z] = max_util #update value function
            res.pol_func[i_k, i_z] = k_next
        end
    end
    v_next
end

prim, res = Solve_model() #solve the model.
plot(prim.k_grid, res.val_func, label = ["Good" "Bad"]) #plot value function
plot(prim.k_grid, res.pol_func, label = ["Good" "Bad"]) #plot policy function
