#pragma once
#include <unordered_map>
#include <string>

struct SafeState {
    float y;
    int gamemode;
    float rotation;
    float yVelocity;
};

extern std::unordered_map<int, SafeState> g_safeStateMap;

void loadSafeStatesFromFile(const std::string& path);

void saveSafeStatesToFile(const std::unordered_map<int, SafeState>& map);
