#pragma once

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

namespace controls
{
    void pressJump();
    void releaseJump();
    void resetLevel();
    void loadFromPercent(int percent);
    void freeze();
    void unfreeze();
    void step(int frames, bool press_jump); // Defined in main.cpp :(

} // namespace controls
