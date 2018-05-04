yyaxis left
plot(hash_size, CCR)
ylabel('CCR')
yyaxis right
plot(hash_size, time)
ylabel('Total Running Time')
ylabel('Total Running Time in Seconds')
xlabel('Hash Size')
title('CCR and Total Running Time vs Hash Size')