#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "DataPoint.h" 

template<typename T>
std::vector<DataPoint<T>> importDataPointsFromCSV(const std::string& filePath, bool hasHeader = false, int labelColumn = -1, std::vector<int> ignoreCols = {}) {
    std::vector<DataPoint<T>> dataPoints;
    std::ifstream file(filePath);
    std::string line;

    if (hasHeader && !std::getline(file, line)) {
        // Skip header line
        return dataPoints; 
    }

    while (std::getline(file, line)) {
        std::vector<T> coords;
        std::string label;
        std::stringstream ss(line);
        std::string item;
        int currentColumn = 0;

        while (std::getline(ss, item, ',')) {
            if (currentColumn == labelColumn) {
                label = item;
            }

            // If not label column and not in ignoreCols, convert and add to coords
            else if (std::find(ignoreCols.begin(), ignoreCols.end(), currentColumn) == ignoreCols.end()) {
                std::stringstream convertor(item);
                T value;
                convertor >> value; // Convert string to T
                coords.push_back(value);
            }
            
            currentColumn++; // Increment currentColumn after processing each item
        }

        // Only add data point if it contains coordinates (avoids adding empty or fully ignored rows)
        if (!coords.empty()) {
            dataPoints.emplace_back(DataPoint<T>(coords, label));
        }
    }

    return dataPoints;
}
