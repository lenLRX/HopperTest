nvcc -arch sm_90a hash_test.cu -I../../cuCollections/include/ --std=c++17 -o hash_test.exe --expt-relaxed-constexpr --expt-extended-lambda
nsys profile -f true -o hashtest_0pct ./hash_test.exe  0
ncu --set=full -f -o hashtest_0pct ./hash_test.exe 0

nsys profile -f true -o hashtest_30pct ./hash_test.exe  0.3
ncu --set=full -f -o hashtest_30pct ./hash_test.exe 0.3

nsys profile -f true -o hashtest_50pct ./hash_test.exe  0.5
ncu --set=full -f -o hashtest_50pct ./hash_test.exe 0.5


nsys profile -f true -o hashtest_70pct ./hash_test.exe  0.7
ncu --set=full -f -o hashtest_70pct ./hash_test.exe 0.7
