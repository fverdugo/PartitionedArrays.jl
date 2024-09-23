"""
	Geometry

	Collect data about the geometry of the problem.

	# Arguments

	- `nx`: points in the x direction for each process
	- `ny`: points in the y direction for each process
	- `nz`: points in the z direction for each process
	- `npx`: parts in the x direction 
	- `npy`: parts in the y direction 
	- `npz`: parts in the z direction 
	- `nnz`: number of non zeroes in sparse matrices of the preconditioner
	- `nrows`: number of rows in each of the sparse matrices of the preconditioner
"""
struct Geometry
	nx::Int64
	ny::Int64
	nz::Int64
	npx::Int64
	npy::Int64
	npz::Int64
	nnz::Vector{Int64}
	nrows::Vector{Int64}
end

"""
	restrict_operator(nx, ny, nz) -> f2c

	Creates a mapping for vector x to a coarse vector x corresponding to a 
		mapping for a matrix to a matrix that is half the size in all directions.

	# Arguments

	- `nx`: points in the x direction for each process
	- `ny`: points in the y direction for each process
	- `nz`: points in the z direction for each process

	# Output

	- `f2c`: fine to coarse mapping vector.
"""
function restrict_operator(nx, ny, nz)
	@assert (nx % 2 == 0) && (ny % 2 == 0) && (nz % 2 == 0)
	nxc = div(nx, 2)
	nyc = div(ny, 2)
	nzc = div(nz, 2)
	local_number_of_rows = nxc * nyc * nzc
	f2c = zeros(Int32, local_number_of_rows)
	for izc in 1:nzc
		izf = 2 * (izc - 1)
		for iyc in 1:nyc
			iyf = 2 * (iyc - 1)
			for ixc in 1:nxc
				ixf = 2 * (ixc - 1)
				current_coarse_row = (izc - 1) * nxc * nyc + (iyc - 1) * nxc + (ixc - 1) + 1
				current_fine_row = izf * nx * ny + iyf * nx + ixf
				f2c[current_coarse_row] = current_fine_row + 1
			end
		end
	end
	return f2c
end

"""
	Conjugate gradient solver util functions.
"""



