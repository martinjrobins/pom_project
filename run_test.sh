#!/bin/bash

COUNTER=0
for reversible in true false
do
    for diff_i in 1 2 5 1000
    do
        for sample_i in 1
        do
             echo $COUNTER
             echo $sample_i
             echo -e "#!/bin/bash\npython fit_pom.py $diff_i $sample_i $reversible" > job_$COUNTER.sh
             COUNTER=$((COUNTER + 1))
        done
    done
done

