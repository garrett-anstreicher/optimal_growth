using Parameters
using Plots

@with_kw struct Primitives
	beta = 0.99 #discount rate
	theta = 0.36 #capital share
	delta = 0.025 #capital depreciation
	z_values = [1.25, 0.2] #productivity values in states
	z_transition = [[0.977, 0.074] [0.023, 0.926]] #productivity stochastic transition matrix
	nz = length(z_values)
	k_grid = collect(range(1.0, length = 1800, stop = 45.0)) #capital grid
	nk = length(k_grid) #number of capital elements
end

mutable struct Results
	val_func::Array{Float64,2} #value function
	pol_func::Array{Float64,2} #policy function
end

function Solve_model(; tol::Float64=1e-4, verbose::Bool=false)
	
	p = Primitives()
	val_func, pol_func = zeros(p.nk, p.nz), zeros(p.nk, p.nz)
	results = Results(val_func, pol_func)

	error = Inf
	i = 1
	while error > tol
		v_next = Bellman(results, p)
		error = maximum(abs.(results.val_func - v_next))
		results.val_func = v_next #update value function; policy function updated in Bellman function
		if verbose
			println(i, ": error ", error)
		end
		i += 1
	end
	return p, results

end

#Bellman operator
function Bellman(res::Results, p::Primitives)

	budget = (p.k_grid .^ p.theta)*p.z_values' .+ ((1-p.delta)*p.k_grid)*ones(p.nz)' #budget given (k,z) (nk x nz)
	cont_val = p.beta * (res.val_func*p.z_transition') #expected continuation value of choosing k' (nk x 1)

	v_next = fill(-Inf, (p.nk, p.nz))
	for i_kp = 1:p.nk #loop over choice of k_prime
		kp = p.k_grid[i_kp]
		c = max.(budget .- kp, 0) #consumption choosing k_prime given (k,z)
		val = log.(c) .+ cont_val[i_kp] #value of choosing k_prime given (k,z) (will be -Inf if c = 0)
		res.pol_func = res.pol_func .* (val .<= v_next) .+ kp*(val .> v_next) #update policy function
		v_next = max.(val, v_next) #updated value function
	end
	return v_next

end

prim, results = Solve_model(verbose=true)

plot1 = plot(prim.k_grid, results.val_func) #plot value function
savefig(plot1, "val_func.png")
plot2 = plot(prim.k_grid, results.pol_func) #plot policy function
savefig(plot2, "pol_func.png")
