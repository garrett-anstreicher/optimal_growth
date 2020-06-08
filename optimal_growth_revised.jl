# XIAN WU Problem 7
using Parameters, Plots #read in necessary packages

#global variables instead of structs
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate.
    θ::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #capital depreciation
    k_grid::Array{Float64,1} = collect(range(0.01, length = 1000, stop = 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital elements
    Π::Array{Float64,2} = [0.977 0.023; 0.074 0.926] # transition matrix
    Z::Array{Float64,2} = [1.25 0.2] #[Zg Zb]
end

#initialize value function and policy functions, again as globals.
mutable struct Results
    val_func::Array{Float64,2} #value function
    pol_func::Array{Float64,2}
end

#function to solve the model
function Solve_model()
    #initialize primitives and results
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk,2), zeros(prim.nk,2)
    res = Results(val_func, pol_func)
    error,n = 100, 0
    while error> 1e-4
        n+=1
        v_next, pol_func = Bellman(prim,res)
        error = maximum(abs.(res.val_func .- v_next))#reset error term
        res.val_func = v_next #update value function held in results vector
        println(n, "  ",  error)
    end
    println("Value function converged in ", n, " iterations.")
    prim,res
end


#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid, Π,Z = prim #unpack primitive structure
    @unpack pol_func,val_func =res
    v_next = zeros(nk,2)
    for i_z=1:2
        for i_k = 1:nk #loop over state space
            candidate_max = -1e10  #something crappy
            k = k_grid[i_k]#convert state indices to state values
            budget = Z[i_z] * k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?
            for i_kp = 1:nk #loop over choice of k_prime
                kp = k_grid[i_kp]
                c = budget - kp #consumption
                if c>0 #check to make sure that consumption is positive
                    val = log(c) + β *( val_func[i_kp,1]*Π[i_z,1]+val_func[i_kp,2]*Π[i_z,2])
                    if val>candidate_max #check for new max value
                        candidate_max = val
                        pol_func[i_k,i_z] = kp #update policy function
                    end
                end
            end
            v_next[i_k,i_z] = candidate_max #update next guess of value function
        end
    end
    v_next, pol_func
end
@elapsed prim, res = Solve_model() #solve the model.
Plots.plot(prim.k_grid, res.val_func[:,1],label="Good State") #plot value function
Plots.plot(prim.k_grid, res.val_func[:,2],label="Bad State") #plot value function

#############
