#=
Original code from Garrett Anstreicher. I've added in some of the code from our
course to help create the structs and model-solving function.
=#

using Parameters, Plots #read in necessary packages

#struct to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount factor
    θ::Float64 = 0.36 #production
    δ::Float64 = 0.025 #depreciation
    k_grid::Array{Float64,1} = collect(range(0.1, length = 1800, stop= 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital grid states
    mk::Array{Float64, 2} = [0.977 0.023; 0.074 0.926] #Markov matrix
    Z::Array{Float64, 1} = [1.25, 0.2] #Z states
    nz::Int64 = length(Z) #Number of Z parameters- dictates Vk/Pk dimension
end

# structs are adjusted to fit dimension of states
mutable struct Results
    val_func::Array{Float64,2} #value function
    pol_func::Array{Float64,2} #policy function
end




#Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, θ, δ, k_grid, nk, mk, Z, nz  = prim #the struct above
    v_next = zeros(nk,nz)

    for i_k = 1:nk #loop over state space
        candidate_max = [-1e10,-1e10] #something crappy
        k = k_grid[i_k]#convert state indices to state values
        budget = Z.*(k^θ) .+ (1-δ)*k #budget given current state. Doesn't this look nice?

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp]
            c = budget .- kp #consumption

            #c[1] should be "good" state, and c[2] "bad"
            if c[1]>0 #check to make sure that consumption is positive
                val = log(c[1]) + β * (res.val_func[i_kp,1]*mk[1,1]
                                        + res.val_func[i_kp,2]*mk[1,2])
                if val>candidate_max[1] #check for new max value
                    candidate_max[1] = val
                    res.pol_func[i_k,1] = kp #update policy function
                end
            end
            if c[2] > 0
                val = log(c[2]) + β * (res.val_func[i_kp,1]*mk[2,1]
                                        + res.val_func[i_kp,2]*mk[2,2])
                if val>candidate_max[2] #check for new max value
                    candidate_max[2] = val
                    res.pol_func[i_k,2] = kp #update policy function
                end
            end

        end
        v_next[i_k,:] = candidate_max #update next guess of value function
    end
    v_next, res.pol_func
end

#Instead of solving the model by line calls, we can solve it with a function
function Solve_model()
    #initialize primitives and results
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk,prim.nz), zeros(prim.nk,prim.nz)
    res = Results(val_func, pol_func)

    error, n = 100, 0
    while error>0.0001 #loop until convergence
        n+=1
        v_next, res.pol_func = Bellman(prim, res) #next guess of value function
        #println("v_next: ",size(v_next),"; val_func: ", size(res.val_func))
        error = maximum(abs.(v_next .- res.val_func)) #check for convergence
        res.val_func = v_next #update

        if n%100 == 0
            println("Current error: ", error)
        end
    end
    println("Value function converged in ", n, " iterations")
    prim, res
end

@elapsed prim, res = Solve_model()
Plots.plot(prim.k_grid, res.val_func, lab = ["Good" "Bad"])
