### Ana Sofia Teixeira Teles

using Parameters, Plots #read in necessary packages

#Used structures here:

@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate.
    θ::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #capital depreciation
    k_grid::Array{Float64,1} = collect(range(1.0, length = 1800, stop = 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital elements
    pi::Matrix{Float64} = [0.9770 0.023; 0.074 0.926] #transition matrix
    z::Array{Float64,1} = [1.25, 0.2] # values of Z in good and bad state
    nz::Int64 = 2 #number of states Z
end

#initialize value function and policy functions: changed to structure.

mutable struct Results
    val_func::Matrix{Float64}
    pol_func::Matrix{Float64}
end   

#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(prim::Primitives, res::Results)

    @unpack β, θ, δ, k_grid, nk, pi, z, nz = prim #unpack primitive structure
    v_next = zeros((nk,2)) #preallocate next guess of value function

    for i_z = 1:nz #loop over z state Variable
        zi = z[i_z]

        for i_k = 1:nk #loop over k state space
            candidate_max = -1e10 #something crappy
            k = k_grid[i_k]#convert state indices to state values
            budget = zi*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

            for i_kp = 1:nk #loop over choice of k_prime
                kp = k_grid[i_kp]
                c = budget - kp #consumption

                if c>0 #check to make sure that consumption is positive
                    val = log(c) + β * pi[i_z,:]'*res.val_func[i_kp,:]

                    if val>candidate_max #check for new max value
                        candidate_max = val
                        res.pol_func[i_k, i_z] = kp #update policy function
                    end
                end
            end
            v_next[i_k,i_z] = candidate_max #update next guess of value function
        end
        
    end
    v_next
end

#Create structure with optimization parameters

mutable struct Optim
    error::Float64
    n::Int
end

tol = 1e-4 
function solve_model()

    prim = Primitives()
    val_func, pol_func = zeros((prim.nk,prim.nz)), zeros((prim.nk,prim.nz))
    res = Results(val_func, pol_func)

     o = Optim(100.0,0)   #initialize optim parameters error, n
    while o.error>tol
        o.n+=1
        v_next = Bellman(prim, res)
        o.error = maximum(vec(abs.(v_next - res.val_func))) #reset error term: vec operator makes matrix 1-dim array
        res.val_func = v_next #update value function held in results vector

        println(o.n, "  ",  o.error)
    end

    println("Value function converged in ", o.n, " iterations.")
    res #output value function and policy function

end
#############

solve_model()