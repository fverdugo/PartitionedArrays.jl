"""
	build_p_matrix(ranks, nx, ny, nz, gnx, gny, gnz, npx, npy, npz) -> I, J, V, Ib, Vb

	Create the partitioned versions of A and b. 

	# Arguments

	- `nx`: points in the x direction for each process
	- `ny`: points in the y direction for each process
	- `nz`: points in the z direction for each process
	- `gnx`: total points in the x direction 
	- `gny`: total points in the y direction 
	- `gnz`: total points in the z direction 
	- `gix0`: gloabl starting index of points in x direction of process
	- `giy0`: global starting index of points in y direction of process
	- `giy0`: global starting index of points in z direction of process

	# Output

	- `I`: row vector psparse matrix A
	- `J`: col vector psparse matrix A
	- `V`: value vector psparse matrix A
	- `Ib`: row vector b
	- `b`: value vector b
	
"""
function build_matrix(nx, ny, nz, gnx, gny, gnz, gix0, giy0, giz0)
	row_count = nx * ny * nz

	@assert row_count > 0

	non_zeros_per_row = 27
	b = zeros(Float64, row_count)
	row_b = zeros(Int64, row_count)

	col_vec = zeros(Int64, row_count * non_zeros_per_row)
	row_vec = zeros(Int64, row_count * non_zeros_per_row)
	val_vec = zeros(Float64, row_count * non_zeros_per_row)

	current_vec_index = 0
	for iz in 1:nz
		giz = giz0 + iz - 1
		for iy in 1:ny
			giy = giy0 + iy - 1
			for ix in 1:nx
				gix = gix0 + ix - 1
				current_row = ((iz - 1) * nx * ny + (iy - 1) * nx + (ix - 1)) + 1
				current_global_row = (giz - 1) * gnx * gny + (giy - 1) * gnx + (gix - 1) + 1
				non_zeros_in_row_count = 0
				# for each value get 27 point stencil
				for sz in -1:1
					if giz + sz > 0 && giz + sz < gnz + 1
						for sy in -1:1
							if giy + sy > 0 && giy + sy < gny + 1
								for sx in -1:1
									if gix + sx > 0 && gix + sx < gnx + 1
										curcol = current_global_row + sz * gnx * gny + sy * gnx + sx
										current_vec_index += 1

										if curcol == current_global_row
											val_vec[current_vec_index] = 26.0
										else
											val_vec[current_vec_index] = -1.0
										end
										row_vec[current_vec_index] = current_global_row
										col_vec[current_vec_index] = curcol
										non_zeros_in_row_count += 1
									end
								end
							end
						end
					end
				end
				row_b[current_row] = current_global_row
				b[current_row] = 27.0 - non_zeros_in_row_count
			end
		end
	end
	first(row_vec, current_vec_index), first(col_vec, current_vec_index), first(val_vec, current_vec_index), b, row_b
end

"""
	build_p_matrix(ranks, nx, ny, nz, gnx, gny, gnz, npx, npy, npz) -> A, b

	Create the partitioned versions of A and b. 

	# Arguments

	- `ranks`: distribute object of the processes
	- `nx`: points in the x direction for each process
	- `ny`: points in the y direction for each process
	- `nz`: points in the z direction for each process
	- `gnx`: total points in the x direction 
	- `gny`: total points in the y direction 
	- `gnz`: total points in the z direction 
	- `npx`: processors in the x direction
	- `npy`: processors in the y direction 
	- `npz`: processors in the z direction 

	# Output

	- `A`: psparse matrix 
	- `b`: pvector right handside
"""
function build_p_matrix(ranks, nx, ny, nz, gnx, gny, gnz, npx, npy, npz)
	row_partition = uniform_partition(ranks, (npx, npy, npz), (gnx, gny, gnz))
	cis = CartesianIndices((gnx, gny, gnz))
	IJVb = map(row_partition) do my_rows
		gix0, giy0, giz0 = Tuple(cis[first(my_rows)])
		I, J, V, b, I_b = build_matrix(nx, ny, nz, gnx, gny, gnz, gix0, giy0, giz0)
		I, J, V, b, I_b
	end
	I, J, V, b, I_b = tuple_of_arrays(IJVb)

	T = SparseMatrixCSC{Float64, Int32}
	J_owner = find_owner(row_partition, J)
	col_partition = map(union_ghost, row_partition, J, J_owner)
	A = psparse(T, I, J, V, row_partition, col_partition, assembled = true) |> fetch
	row_partition = partition(axes(A, 2))
	b = pvector(I_b, b, row_partition) |> fetch
	return A, b
end


