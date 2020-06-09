using Parameters, Plots #read in necessary packages

#global variables instead of structs
@with_kw struct Primitives
    β::Float64=0.99 #discount rate.
    θ::Float64=0.36 #capital share
    δ::Float64=0.025 #capital depreciation
    k_grid::Array{Float64,1}=collect(range(.01, length = 1800, stop = 45.0)) #capital grid
    nk::Int64=length(k_grid) #number of capital elements
    z_grid::Array{Float64,1}=[1.25, .2]
    nz::Int64=length(z_grid)
    Markov::Array{Float64,2}=[.977 .023; .074 .926]
end

#initialize value function and policy functions, again as globals.
mutable struct Results
    val_func::Array{Float64,2}
    pol_func::Array{Float64,2}
end

function Solve_model(tol)
    prim=Primitives()
    val_func,pol_func=zeros(prim.nk,prim.nz),zeros(prim.nk,prim.nz)
    res=Results(val_func,pol_func)

    error,n=100,0
    while error>tol #loop until convergence
        n+=1
        v_next=Bellman(prim,res) #next guess of value function
        error=maximum(abs.(v_next .- res.val_func)) #check for convergence
        res.val_func=v_next #update
        println("current error: ",error)
    end
    println("value function converged in ",n," iterations")
    prim,res
end

#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(prim::Primitives, res::Results)
    @unpack β,δ,θ,nk,k_grid,nz,z_grid,Markov=prim #unpack primitive structure
    v_next = zeros(nk,nz) #preallocate next guess of value function

    for i_k=1:nk,i_z=1:nz #loop over state space
        candidate_max=-1e10 #something crappy
        k,z=k_grid[i_k],z_grid[i_z] #convert state indices to state values
        budget=z*k^θ+(1-δ)*k #budget given current state. Doesn't this look nice?

        for i_kp=1:nk #loop over choice of k_prime
            kp=k_grid[i_kp] #value of k'
            c=budget-kp #consumption
            if c>0 #check to make sure that consumption is positive
                val=log(c)+β*sum(res.val_func[i_kp,:].*Markov[i_z, :]) #compute utility
                if val>candidate_max #check for new max value
                    candidate_max=val #reset max value
                    res.pol_func[i_k]=kp #update policy function
                end
            end
        end
        v_next[i_k,i_z]=candidate_max #update next guess of value function
    end
    v_next
end

@elapsed prim,res=Solve_model(1e-4) #solve the model
Plots.plot(prim.k_grid,res.val_func) #plot value function

#############
