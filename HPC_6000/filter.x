for i in `seq 1 6000`;do 

tr -d 'Any[];' < dipole_jimwlk_${i}.dat > correct_format_${i}.dat


done
