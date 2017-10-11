#!/bin/bash

COUNTER=0
for reversible in true false
do
    for diff_i in `seq 0 20` 1000
    do
        for sample_i in `seq 1 20`
        do
             echo $COUNTER
             echo $sample_i
             echo -e "#!/bin/bash\npython fit_pom.py $diff_i $sample_i $reversible" > job_$COUNTER.sh
             COUNTER=$((COUNTER + 1))
        done
    done
done

