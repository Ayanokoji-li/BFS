#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <cmath>
#include <set>
#include <random>
#include <atomic>

#define FILE_LOACTE "/media/ayanokouji/TOSHIBA EXT/"

#define START_NODE 2
#define THREAD_NUM_STEP 4
#define THREAD_NUM_START 4
static int file_index = 0;

std::vector<std::string> default_file_names = {"test.mtx","web-Stanford.mtx", "soc-LiveJournal1.mtx", "roadNet-CA.mtx", "com-orkut.mtx", "RMAT1.mtx", "RMAT2.mtx", "RMAT3.mtx"}; 
auto num_files = default_file_names.size();
std::string file_name;
bool is_parallel = false;
bool is_benchmark = false;
bool is_scalable = false;
bool auto_threads = true;
int thread_nums = omp_get_num_procs();
unsigned long long edge_num = 0;

// #define FROM_DISK

#define benchmark_Iteration 20
#define FIRST_GAP 300000
#define FIRST_GAP_THREAD 16
#define SECOND_GAP 2000000
#define SECOND_GAP_THREAD 18

// 图的数据结构
typedef std::vector<std::vector<unsigned long long>> Graph;

unsigned long long random(unsigned long long up_bound, const Graph &graph);

// 从mtx文件中读取图
Graph readGraph(std::string filename, Graph& graph, unsigned long long& max_node) 
{
    std::ifstream file(filename);
    if(file.fail()) 
    {
        std::cout << "file not found\n";
        exit(0);
    }


    for(auto i = 0; i < 2; i++) 
    {
        file.ignore(1000, '\n'); // 跳过前两行
    }


    file >> max_node >> max_node >> edge_num;
    graph.resize(max_node+1);
    unsigned long long from, to, length;
    while (file >> from >> to >> length) {
        graph[from].push_back(to);
        graph[to].push_back(from);
    }
    return graph;
}



// 广度优先搜索
void bfs(const Graph &graph, unsigned long long &max_node_index, unsigned long long start, std::vector<unsigned long long>&result, std::vector<unsigned long long>&parent, std::vector<unsigned long long>&distance) {
    
    result.clear();
    if(is_parallel == false)
    {
        std::vector<bool> visited(max_node_index, 0);
        std::queue<unsigned long long> q;
        q.push(start);
        visited[start] = true;
        while (!q.empty()) {
            unsigned long long node = q.front();
            q.pop();
            result.push_back(node);
            if(graph[node].empty()) continue; // 有些节点没有出度
            for (unsigned long long neighbor : graph.at(node)) {
                if (!visited[neighbor]) {
                    q.push(neighbor);
                    visited[neighbor] = 1;
                    parent[neighbor] = node;
                    distance[neighbor] = distance[node] + 1;
                }
            }
        }        
    }
    else
    {
        std::vector<std::atomic<bool>> visited(max_node_index+1);
        std::vector<bool> visited_share(max_node_index, 0);
        std::vector<unsigned long long> frontier;
        frontier.push_back(start);
        visited[start] = true;
        visited_share[start] = true;
        while (!frontier.empty()) {
            result.insert(result.end(), frontier.begin(), frontier.end());
            std::vector<unsigned long long> find_node_sum(thread_nums + 1, 0);
            auto copy = frontier;
            frontier.clear();
            #pragma omp parallel num_threads(thread_nums)
            {
                auto tid = omp_get_thread_num();
                std::vector<unsigned long long> next_frontier_private;
                #pragma omp for
                for(auto i = 0; i < copy.size(); i++) 
                {
                    unsigned long long node = copy[i];
                    if(graph[node].empty()) continue; // 有些节点没有出度
                    for (unsigned long long neighbor : graph[node]) 
                    {
                        bool tmp = false;
                        if(visited_share[neighbor] == false)
                        {
                            if(visited[neighbor].compare_exchange_strong(tmp, true))
                            {
                                visited_share[neighbor] = true;
                                next_frontier_private.push_back(neighbor);
                                distance[neighbor] = distance[node] + 1;
                                parent[neighbor] = node;
                                find_node_sum[tid]++;
                            }
                        }
                    }
                }

                #pragma omp single
                {
                    unsigned long long all_sum = 0;
                    for(int i = 0 ; i < thread_nums; i++)
                    {
                        unsigned long long tmp = find_node_sum[i];
                        find_node_sum[i] = all_sum;
                        all_sum += tmp;
                    }
                    find_node_sum[thread_nums] = all_sum;
                    frontier.resize(all_sum);
                }


                for(unsigned long long j = find_node_sum[tid]; j < find_node_sum[tid+1]; j++)
                {
                    frontier[j] = next_frontier_private[j - find_node_sum[tid]];
                }

            }
        }        
    }
}





void bfs_benchmark(const Graph& graph, unsigned long long& max_node_index, std::vector<unsigned long long>& result, std::vector<unsigned long long>& parent, std::vector<unsigned long long>& distance, double &average_edge_performance, double &std_deviation, std::vector<double>& times, std::vector<unsigned long long>& searched_edges)
{
    unsigned long long start_node = 0;
    for(int i = 0; i < benchmark_Iteration; i ++)
    {
        start_node = random(max_node_index, graph);
        result.clear();
        parent = std::vector<unsigned long long>(max_node_index+1, 0);
        distance = std::vector<unsigned long long>(max_node_index+1, 0);

        auto start = omp_get_wtime();
        bfs(graph, max_node_index, start_node, result, parent, distance);
        auto end = omp_get_wtime();
        auto time = end - start;
        times.push_back(time);
        searched_edges.push_back(result.size()-1);
    }
    std::vector<double> edge_performance;
    for(int i = 0; i < benchmark_Iteration; i ++)
    {
        edge_performance.push_back((double)(result.size()-1) / times[i] / 1e6);
    }
    average_edge_performance = 0;
    for(auto i : edge_performance)
    {
        average_edge_performance += i;
    }
    average_edge_performance /= benchmark_Iteration;

    std_deviation = 0;
    for(auto i : edge_performance)
    {
        std_deviation += (i - average_edge_performance) * (i - average_edge_performance);
    }
    std_deviation /= benchmark_Iteration;
    std_deviation = sqrt(std_deviation);
}

// 将结果写入文件
void writeNormalResult(const std::string &filename, const std::vector<unsigned long long> &result, double& time, const std::vector<unsigned long long> &parent, const std::vector<unsigned long long> &distance) {
    std::ofstream file(filename);
    file << "BFS results and profile" << std::endl;
    file << "number of nodes searched: " << result.size() << "\n";
    file << "time:" << time << "s\n";
    file << "nodes searched:\n";
    for (unsigned long long i = 0; i < result.size(); i++) {
        file << result[i] << "\t" << distance[result[i]] << "\t" << parent[result[i]] << "\n";
    }
}

void writeBenchmarkResult(const std::string &filename, const double edge_prof_average, const double edge_prof_std_deviation, const std::vector<double> &times, const std::vector<unsigned long long> &searched_edges) {
    std::ofstream file(filename);
    file << "Benchmark\n";
    file << "average edge performance: " << edge_prof_average << "M edges/s\n";
    file << "standard deviation: " << edge_prof_std_deviation << "M edges/s\n\n";
    file << "times:\n";
    for (unsigned long long i = 0; i < times.size(); i++) {
        file << "loop" << i << ": " << times[i] << "s\t\tedges: " << searched_edges[i] << "\n";
    }
}

void writeScaleResult(const std::string &filename, const std::vector<double> each_average_edge_performance, const std::vector<double> each_std_deviation)
{
    int max_threads = omp_get_num_procs();
    std::ofstream file(filename);
    file << "Scalability test\n";
    for(int i = THREAD_NUM_START; i <= max_threads; i+=THREAD_NUM_STEP)
    {
        auto index = (i - THREAD_NUM_START) / THREAD_NUM_STEP;
        file << i << " threads running:\n";
        file << "average edge performance: " << each_average_edge_performance[index] << "M edges/s\n";
        file << "standard deviation: " << each_std_deviation[index] << "M edges/s\n\n";
    }
}

int main(int argc, char **argv) {
    Graph graph;
    std::vector<std::string> mode_name = {"sequencial", "parallel"};

    if(argc < 3) 
    {
        std::cout << "which file to load\n";
        int i;
        for(i = 0; i < num_files; i++) {
            std::cout << i << ":" << default_file_names[i] << "\n";
        }
        std::cout << i << ":customed file\n";
        std::cin >> file_index;
        std::cout << std::endl;
        if(file_index >= num_files + 1) {
            std::cout << "invalid input\n";
            return 0;
        }
        else if(file_index == num_files)
        {
            std::cout << "input file name\n";
            std::cin >> file_name;
            std::cout << std::endl;
        }
        else
        {
            file_name = default_file_names[file_index];
        }


        std::cout << "which mode to use\n";

        for(auto i = 0; i < 2; i++) {
            std::cout << i << ":" << mode_name[i] << "\n";
        }
        std::cin >> is_parallel;
        std::cout << std::endl;

        std::cout << "whether to benchmark\n"
                    << "0: No\n"
                    << "1: Yes\n";
        std::cin >> is_benchmark;
        std::cout << std::endl;

        if(is_benchmark == true && is_parallel == true)
        {            
            std::cout << "whether to test scalablility\n"
                        << "0: No\n"
                        << "1: Yes\n";
            std::cin >> is_scalable;
            std::cout << std::endl;
        }

        if(is_parallel == true && is_scalable == false)
        {            
            std::cout << "whether to auto allocate threads\n"
                        << "0: No\n"
                        << "1: Yes\n";
            std::cin >> auto_threads;
            std::cout << std::endl;
        }

        if(!auto_threads)
        {
            std::cout << "how many threads to use\n";
            std::cin >> thread_nums;
            std::cout << std::endl;
        }
    }
    else
    {
        file_index = atoi(argv[1]);
        is_parallel = (bool)atoi(argv[2]);
        is_benchmark = (bool)atoi(argv[3]);
        if(argc > 4)
        {
            is_scalable = (bool)atoi(argv[4]);

            if(is_parallel == true && is_scalable == false)
            {
                auto_threads = (bool)atoi(argv[5]);
                if (auto_threads == false)
                {
                    thread_nums = atoi(argv[6]);
                }
            }
        }

        if(file_index >= num_files) {
            std::cout << "invalid input\n";
            return 0;
        }
        file_name = default_file_names[file_index];
    }
    unsigned long long max_node_index = 0;

    std::string tmp = file_name;
    if(file_index != num_files)
    {
        #ifdef FROM_DISK
        tmp = FILE_LOACTE + default_file_names[file_index];
        #endif        
    }

    readGraph(tmp, graph, max_node_index);
    std::vector<unsigned long long> result;
    std::vector<unsigned long long> parent(max_node_index+1, 0);
    std::vector<unsigned long long> distance(max_node_index+1, 0);

    if(auto_threads == true)
    {
        if(max_node_index <= FIRST_GAP)
        {
            omp_set_num_threads(FIRST_GAP_THREAD);
            thread_nums = FIRST_GAP_THREAD;
        }
        else if(max_node_index <= SECOND_GAP)
        {
            omp_set_num_threads(SECOND_GAP_THREAD);   
            thread_nums = SECOND_GAP_THREAD;
        }
        else
        {
            omp_set_num_threads(thread_nums);
        }
    }
    else
    {
        omp_set_num_threads(thread_nums);
    }

    if(is_benchmark == false)
    {
        auto start = omp_get_wtime();
        bfs(graph, max_node_index, START_NODE, result, parent, distance);
        auto end = omp_get_wtime();
        auto time = end - start;
        std::string result_file = "_res.txt";
        result_file = file_name + "_" + mode_name[is_parallel] + result_file;
        writeNormalResult(result_file, result, time, parent, distance);
    }
    else if(is_scalable == false)
    {
        std::vector<double> times;
        std::vector<unsigned long long> searched_edges;
        double average_edge_performance = 0;
        double std_deviation = 0;

        bfs_benchmark(graph, max_node_index, result, parent, distance, average_edge_performance, std_deviation, times, searched_edges);

        std::string result_file = "_res.txt";
        result_file = file_name + "_" + mode_name[is_parallel] + "_benchmark" + result_file;
        writeBenchmarkResult(result_file, average_edge_performance, std_deviation, times, searched_edges);
    }
    else if(is_scalable == true)
    {
        int max_threads = omp_get_num_procs();
        std::vector<double> each_average_edge_performance;
        std::vector<double> each_std_deviation;
        for(int i = THREAD_NUM_START; i <= max_threads; i += THREAD_NUM_STEP)
        {
            thread_nums = i;
            omp_set_num_threads(i);
            std::vector<double> times;
            std::vector<unsigned long long>searched_edges;
            double average_edge_performance = 0;
            double std_deviation = 0;

            bfs_benchmark(graph, max_node_index, result, parent, distance, average_edge_performance, std_deviation, times, searched_edges);            


            each_average_edge_performance.push_back(average_edge_performance);
            each_std_deviation.push_back(std_deviation);
        }

        std::string result_file = "_res.txt";
        result_file = file_name + "_" + mode_name[is_parallel] + "_scalable" + result_file;
        writeScaleResult(result_file, each_average_edge_performance, each_std_deviation);
    }   
    return 0;
}

// output random number between 1 and up_bound
unsigned long long random(unsigned long long up_bound, const Graph &graph)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long long> dis(1, up_bound);

    // find in graph whether the random number is a node
    while(true)
    {
        unsigned long long random_num = dis(gen);
        if(graph[random_num].empty() == false)
        {
            return random_num;
        }
    }
}