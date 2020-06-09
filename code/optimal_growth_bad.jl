using Parameters #read in necessary packages

@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate.
    θ::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #capital depreciation
    k_grid::Array{Float64, 1} = collect(range(1.0, length = 1800, stop = 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital elements
    Π = [0.977 0.023; 0.074 0.926]
    z = [1.25; 0.2]
    nz = 2
end
        
mutable struct Results
    val_func::Array{Float64,2}
    pol_func::Array{Float64,2}
end
    
#Bellman operator.
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid, Π, z, nz = prim #unpack primitive structure
    v_next = zeros(nk, nz)
        
    for i_z = 1:nz
        for i_k = 1:nk #loop over state space
            candidate_max = -1e10 #something crappy
            k = k_grid[i_k]#convert state indices to state values
            budget = z[i_z]*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

            for i_kp = 1:nk #loop over choice of k_prime
                kp = k_grid[i_kp]
                c = budget - kp #consumption
                if c>0 #check to make sure that consumption is positive
                    val = log(c) + β * sum(Π[i_z, :] .* res.val_func[i_kp, :])
                    if val>candidate_max #check for new max value
                        candidate_max = val
                        res.pol_func[i_k, i_z] = kp #update policy function
                    end
                end
            end
            v_next[i_k, i_z] = candidate_max #update next guess of value function
        end
    end
    v_next, res.pol_func
end

#function to solve the model
function Solve_model()
    #initialize primitives and results
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk, prim.nz), zeros(prim.nk, prim.nz)
    res = Results(val_func, pol_func)        
        
    error = 100
    n = 0
    tol = 1e-4
    while error>tol
        n+=1
        v_next, pol_func = Bellman(prim, res)
        error = maximum(abs.(res.val_func - v_next)) #reset error term
        res.val_func = v_next #update value function held in results vector
        println(n, "  ",  error)
    end
    println("Value function converged in ", n, " iterations.")
    prim, res
end
#############
prim, res = Solve_model() #solve the model