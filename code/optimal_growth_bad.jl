using Parameters, Plots #read in necessary packages

@with_kw struct Primitives
    β::Float64 = 0.99 #discount factor
    θ::Float64 = 0.36 #production
    δ::Float64 = 0.025 #depreciation
    k_grid::Array{Float64,1} = collect(range(0.1, length = 1800, stop= 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital grid states
    z_grid::Array{Float64,1} =  [1.25, 0.2] #productivity state grid
    nz::Int64 = length(z_grid)
    markov::Array{Float64,2} = [0.977 0.023; 0.074 0.926] #markov transition process
end

mutable struct Results
    val_func::Array{Float64,2} #value function
    pol_func::Array{Float64,2} #policy function
end

# Function to solve the model
function Solve_model()
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk, prim.nz), zeros(prim.nk, prim.nz)
    res = Results(val_func, pol_func)

    error, n = 100, 0
    while error>0.0001 #loop until convergence
        n+=1
        v_next, res.pol_func = Bellman(prim, res) #next guess of value function
        error = maximum(abs.(v_next .- res.val_func)) #check for convergence
        res.val_func = v_next #update
        println("Current error: ", error)
    end
    println("Value function converged in ", n, " iterations")
    prim, res
end

# Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid, nz, z_grid, markov = prim #unpack primitive structure
    v_next = zeros(nk,nz) #preallocate next guess of value function

    for i_k = 1:nk, i_z = 1:nz #loop over state space
        candidate_max = -1e10 #something crappy
        k, z = k_grid[i_k], z_grid[i_z] #convert state indices to state values
        budget = z*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

        for i_kp = 1:nk #loop over choice of k'
            kp = k_grid[i_kp] #value of k'
            c = budget - kp #consumption
            if c>0 #check to make sure consumption is positive
                val = log(c) + β * sum(res.val_func[i_kp,:] .* markov[i_z, :]) #compute utility
                if val > candidate_max #check for new max value
                    candidate_max = val #reset max value
                    res.pol_func[i_k, i_z] = kp #update policy function
                end
            end
        end
        v_next[i_k,i_z] = candidate_max #update value function
    end
    v_next, res.pol_func
end

# The code seems to be running a little slowly... not sure if this is a feature of
#   the exercise parameters or issue with my solution. Anyways, it runs correctly
#   after some time.
@elapsed prim, res = Solve_model() #solve the model.
Plots.plot(prim.k_grid, res.val_func) #plot value function
