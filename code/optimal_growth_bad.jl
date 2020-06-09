using Parameters, Plots #read in necessary packages

#Primitive Structure
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate
    θ::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #depreciation rate
    k_grid::Array{Float64,1} = collect(range(.01, length=1000, stop = 45.0)) #capital grid
    nk = length(k_grid)
    #Need to add in transitions
    Π::Array{Float64,2} = [0.977 0.023; 0.074 0.926]
    sts::Array{Float64,2} = [1.25 0.2]
    nz::Int64 = length(sts)

end

#Results Structure
mutable struct Results
    val_func::Array{Float64,2} #value function
    pol_func::Array{Float64,2} #policy function
end

#Initialize stuff
function Init()
    prim = Primitives()
    val_func = zeros(prim.nk, prim.nz)
    pol_func = zeros(prim.nk, prim.nz)
    res = Results(val_func, pol_func)
    prim, res
end

# Bellman Operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, θ, δ, k_grid, nk, Π, nz,sts = prim
    @unpack val_func, pol_func = res
    v_next = zeros(nk, nz)

    for i_z = 1:nz
        for i_k = 1:nk
            candidate_max = -1e10
            k = k_grid[i_k]
            budget = sts[i_z]*k^θ + (1-δ)*k


            for i_kp = 1:nk
                kp = k_grid[i_kp]
                c = budget - kp
                if c>0
                    val = log(c) + β*sum(Π[i_z,:].*val_func[i_kp,:])
                    if val > candidate_max
                        candidate_max = val
                        res.pol_func[i_k,i_z] = kp
                    end
                end
            end
            v_next[i_k,i_z] = candidate_max
        end
    end
    v_next
end

#Solve
function SolveM(prim::Primitives, res::Results)
    n = 0
    error = 100
    tol= 1e-4
    while error > tol
        n += 1
        v_next = Bellman(prim, res)
        error = maximum(abs.(res.val_func .-v_next))
        res.val_func = v_next
        println(n, "  ",  error)
    end
    println("Value function converged in ", n, " iterations.")
    prim, res
end


prim, res = Init()
@elapsed prim, res = SolveM(prim::Primitives, res::Results)
