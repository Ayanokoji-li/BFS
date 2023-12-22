# BFS

## configuration of machine

>> CPU: i7-12700\
>> memory: Gloway 8+8+32+32 GB 3000MHz

## Algorithm
>>&emsp;&emsp;The basic idea of the algorithm is to seperate `vector` frontiers to each processors and find their neighbors relatively into local `vector` next_frontier_private. Use `vector` visited to store whether the node is visited to prevent self multiple find. Then coalesce into frontier remove multiple frontiers to reduce size. Loop will stop when frontier is empty.
>>&emsp;&emsp;To imporove, I use `vector of vector` to store the information of the graph similar to adjacent list to speed up when reading graph information.
>> &emsp;&emsp;I use two `vector` visited. One is shared, another is atomic. Only when two visited shows that the node hasn't been visited, add them into frontier so that we can remove multiple nodes and guarantee the performance.
>> &emsp;&emsp;I record the numbers of nodes each threads have found. Then use exclusive prefix sum to calculate the start index in `vector` frontier of each threads start to copy node info so that we can remove the data conflict when coalescing `vector` next_frontier_private.

## Benchmark
>> &emsp;&emsp;Run algorithm for 20 times. Record the time before and after the function. Use the answer of finded node list to calculate each time real edge/s. Output avarage and standard deviation.

## Compile
>>g++ -foo.c -o foo -O3 -fopenmp

## Running
>>./foo\
>>It will have a simple user interface like

        which file to load
        0:test.mtx
        1:web-Stanford.mtx
        2:soc-LiveJournal1.mtx
        3:roadNet-CA.mtx
        4:com-orkut.mtx
        5:RMAT1.mtx
        6:RMAT2.mtx
        7:RMAT3.mtx
        8:customed file

        which mode to use
        0:sequencial
        1:parallel

        whether to benchmark
        0: No
        1: Yes

        whether to test scalablility
        0: No
        1: Yes

        whether to auto allocate threads
        0: No
        1: Yes

>>There is a run_task.bash to run normal task benchmark.