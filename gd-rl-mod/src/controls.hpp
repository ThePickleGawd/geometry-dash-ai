#pragma once

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <thread>
#include <chrono>

using namespace geode::prelude;

namespace controls {

inline void pressJump() {
    if (auto pl = GameManager::sharedState()->getPlayLayer()) {
        if (auto player = pl->m_player1) {
            player->pushButton(PlayerButton::Jump);
        }
    }
}

inline void releaseJump() {
    if (auto pl = GameManager::sharedState()->getPlayLayer()) {
        if (auto player = pl->m_player1) {
            player->releaseButton(PlayerButton::Jump);
        }
    }
}

inline void resetLevel() {
    if (auto pl = GameManager::sharedState()->getPlayLayer()) {
        pl->resetLevel();
    }
}

inline void freeze() {
    if (auto pl = GameManager::sharedState()->getPlayLayer()) {
        if (auto player = pl->m_player1) {
            player->lockPlayer();
        }
    }
}

inline void unfreeze() {
    if (auto pl = GameManager::sharedState()->getPlayLayer()) {
        if (auto player = pl->m_player1) {
            player->m_isLocked = false;
        }
    }
}

void step(int frames, bool press_jump);


} // namespace controls
