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

inline void step(int frames, bool jumping) {
    // Freeze the game if not already frozen
    freeze();


    // if (auto pl = GameManager::sharedState()->getPlayLayer()) {
    //     if (pl->m_player1 && pl->m_player1->m_isLocked) {
    //         log::info("stepping");
    //         pl->m_player1->m_isLocked = false; 
    //         for (int i = 0; i < frames; i++) {
    //             pl->updateTimeWarp(1.0f / 60.0f);
    //             pl->postUpdate(1.0f / 60.0f);
    //         }
    //         pl->m_player1->lockPlayer();
    //     }
    // }


    // Unfreeze game so the player can move forward
    unfreeze();
    // Jump in the game if a jump command is sent, and release if release command is sent
    if (jumping) {
        pressJump();
    } else {
        releaseJump();
    }

    // TODO: figure out logic for running one frame and freezing again

    // std::this_thread::sleep_for(std::chrono::seconds(5));
    // freeze();
    // log::info("waited for 5 seconds");
}


} // namespace controls
