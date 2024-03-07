#include <vector>
#include <chrono>

#include "dataPoint.h"
#include "kmeans.h"
#include "import.h"

// Template function to allow dynamic use of 
template<typename Distance, typename UpdateStrategy>
void performClustering(std::vector<DataPoint<double>> dataPoints, int k, int maxIterations, int numThreads, std::string inputFile, std::string outputFile) {
    std::cout << "Dataset: " << inputFile << std::endl;

    KMeans<double, Distance, UpdateStrategy> kmeans(k, maxIterations);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto assignments = kmeans.cluster(dataPoints, numThreads);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Cluster Time: " << duration.count() << " ms" << std::endl;

    kmeans.writeToOutput(dataPoints, assignments, outputFile, true);
}


int main(int argc, char* argv[]) {
    std::string inputData = "iris";
    std::string distanceType = "Euclidean"; // Example default distance type
    std::string updateStrategy = "Lloyds"; // Default update strategy

    int k = 3;
    int maxIterations = 100;
    int numThreads = 2;

    // Parse command line arguments (simple example)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            inputData = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            distanceType = argv[++i];
        } else if (arg == "-u" && i + 1 < argc) {
            updateStrategy = argv[++i];
        } else if (arg == "-k" && i + 1 < argc) {
            k = std::stoi(argv[++i]);
        } else if (arg == "-m" && i + 1 < argc) {
            maxIterations = std::stoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        }
    }

    std::string inputFile, outputFile;
    int labelCol;
    std::vector<int> ignoreCols;

    if (inputData == "iris") {
        inputFile = "./data/iris.csv";
        outputFile = "./output/iris_out.csv";
        labelCol = 5;
        ignoreCols = {0};
    } else if (inputData == "beans") {
        inputFile = "./data/beans.csv";
        outputFile = "./output/beans_out.csv";
        labelCol = 16;
        ignoreCols = {};
    } else if (inputData == "synthetic") {
        inputFile = "./data/synthetic_300k.csv";
        outputFile = "./output/synthetic_300k_out.csv";
        labelCol = 3;
        ignoreCols = {0};
    }

    std::vector<DataPoint<double>> dataPoints = importDataPointsFromCSV<double>(inputFile, true, labelCol, ignoreCols);
    std::vector<int> assignments;

    // Slightly unwieldy but the best way to allow an interface when templates must be known at compile time
    if (distanceType == "Euclidean" && updateStrategy == "Lloyds") {
        performClustering<EuclideanDistance<double>, LloydsUpdateStrategy<double>>(dataPoints, k, maxIterations, numThreads, inputFile, outputFile);
    } else if (distanceType == "Euclidean" && updateStrategy == "KPP") {
        performClustering<EuclideanDistance<double>, KMeansPlusPlusUpdateStrategy<double, EuclideanDistance<double>>>(dataPoints, k, maxIterations, numThreads,inputFile, outputFile);
    } else if (distanceType == "Cosine" && updateStrategy == "Lloyds") {
        performClustering<CosineDistance<double>, LloydsUpdateStrategy<double>>(dataPoints, k, maxIterations, numThreads, inputFile, outputFile);
    } else if (distanceType == "Cosine" && updateStrategy == "KPP") {
        performClustering<CosineDistance<double>, KMeansPlusPlusUpdateStrategy<double, CosineDistance<double>>>(dataPoints, k, maxIterations, numThreads, inputFile, outputFile);
    } else {
        std::cerr << "Unsupported distance type or update strategy" << std::endl;
        return 1;
    }

    return 0;

}
