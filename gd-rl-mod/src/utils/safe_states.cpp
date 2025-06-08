#include "safe_states.hpp"
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <Geode/Geode.hpp>

std::unordered_map<int, SafeState> g_safeStateMap;

void loadSafeStatesFromFile(const std::string &path)
{
    g_safeStateMap.clear();
    std::ifstream file(path);
    if (!file.is_open())
    {
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        int percent, gamemode;
        float y, rotation, yVel;
        if (iss >> percent >> y >> gamemode >> rotation >> yVel)
        {
            g_safeStateMap[percent] = {y, gamemode, rotation, yVel};
        }
    }
}

void saveSafeStatesToFile(const std::unordered_map<int, SafeState> &map)
{
    std::ofstream file("src/safe_states/stereo_madness_states.txt");
    // std::ofstream file("src/safe_states/GD_Training_Ship_states.txt");
    for (const auto &[percent, state] : map)
    {
        file << percent << " " << state.y << " " << state.gamemode << " " << state.rotation << " " << state.yVelocity << "\n";
    }
    file.close();
}