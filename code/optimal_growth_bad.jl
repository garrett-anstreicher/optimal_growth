using Parameters, Plots #read in necessary packages

@with_kw struct Primitives
    β::Float64 = 0.99
    θ::Float64 = 0.36
    δ::Float64 = 0.025
    k_grid::Array{Float64,1} = collect(range(0.01, length=1800, stop=45.0))
    Π::Array{Float64,2} = [0.977 0.023; 0.074 0.926]
    z_grid::Array{Float64, 1} = [1.25, 0.2]
    nk::Int64 = length(k_grid)
    nz::Int64 = length(z_grid)
end


#initialize value function and policy functions, again as globals.
mutable struct Results
    val_func::Array{Float64,2}
    pol_func::Array{Float64,2}
end

#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid, Π, z_grid, nz = prim

    v_next = zeros(nk,nz)

    for i_k = 1:nk, i_z = 1:nz #loop over state space
        max_util = -1e10
        k , z = k_grid[i_k], z_grid[i_z] #value of capital
        budget = z*k^θ + (1-δ)*k #budget

        for i_kp = 1:nk
            kp = k_grid[i_kp] #value of k'
            c = budget - kp #consumption
            if c>0 #check if postiive
                val = log(c) + β * sum(res.val_func[i_kp].*Π[i_z, :]) #compute utility
                if val > max_util #wow new max!
                    max_util = val #reset max value
                    res.pol_func[i_k, i_z] = kp #update policy function
                end
            end
        end
        v_next[i_k, i_z] = max_util #update value function
    end
    return v_next
end

function Solve_model()
    #initialize primitives and results
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk, prim.nz), zeros(prim.nk,prim.nz)
    res = Results(val_func, pol_func)

    error, n = 100, 0
    while error>0.0001 #loop until convergence
        n+=1
        v_next = Bellman(prim, res) #next guess of value function
        error = maximum(abs.(v_next .- res.val_func)) #check for convergence
        res.val_func = v_next #update
        println("Current error: ", error)
    end
    println("Value function converged in ", n, " iterations")
    return prim, res
end

@time prim, res = Solve_model()

#############
