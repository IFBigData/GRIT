echo "--------------------------------------------------------------------"
echo "--------------------"
tail log_pisc_fine_* | grep Best-test-result | awk '{sum+=$8} END {print "Average = ", sum/NR}'
tail log_pisc_fine_* | grep Best-test-result | awk '{x[NR]=$8; s+=$8; n++} END{a=s/n; for (i in x){ss += (x[i]-a)^2} sd = sqrt(ss/n); print "SD = "sd}'
tail log_pisc_fine_* | grep Best-test-result | awk 'BEGIN { max = -inf } { if ($8 > max) { max = $8; line = $0 } } END { print line }'
echo "--------------------"
tail log_pisc_coarse_* | grep Best-test-result | awk '{sum+=$8} END {print "Average = ", sum/NR}'
tail log_pisc_coarse_* | grep Best-test-result | awk '{x[NR]=$8; s+=$8; n++} END{a=s/n; for (i in x){ss += (x[i]-a)^2} sd = sqrt(ss/n); print "SD = "sd}'
tail log_pisc_coarse_* | grep Best-test-result | awk 'BEGIN { max = -inf } { if ($8 > max) { max = $8; line = $0 } } END { print line }'

echo "--------------------"
tail log_pipa_fine_* | grep Best-test-result | awk '{sum+=$8} END {print "Average = ", sum/NR}'
tail log_pipa_fine_* | grep Best-test-result | awk '{x[NR]=$8; s+=$8; n++} END{a=s/n; for (i in x){ss += (x[i]-a)^2} sd = sqrt(ss/n); print "SD = "sd}'
tail log_pipa_fine_* | grep Best-test-result | awk 'BEGIN { max = -inf } { if ($8 > max) { max = $8; line = $0 } } END { print line }'

echo "--------------------"
tail log_pipa_coarse_* | grep Best-test-result | awk '{sum+=$8} END {print "Average = ", sum/NR}'
tail log_pipa_coarse_* | grep Best-test-result | awk '{x[NR]=$8; s+=$8; n++} END{a=s/n; for (i in x){ss += (x[i]-a)^2} sd = sqrt(ss/n); print "SD = "sd}'
tail log_pipa_coarse_* | grep Best-test-result | awk 'BEGIN { max = -inf } { if ($8 > max) { max = $8; line = $0 } } END { print line }'

