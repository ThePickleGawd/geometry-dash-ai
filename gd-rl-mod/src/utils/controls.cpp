#include "controls.hpp"
#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <iostream>

using namespace geode::prelude;

namespace controls
{
    void pressJump()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->pushButton(PlayerButton::Jump);
            }
        }
    }

    void loadFromPercent(int percent) {
        // TODO: handle correct y positions and game modes (eg: whether to be in cube or ship mode)
        if (auto pl = GameManager::sharedState()->getPlayLayer()) {
            auto checkpoints = pl->m_checkpointArray;
            if (!checkpoints) return;
            float targetX = (percent / 100.0f) * pl->m_levelLength;
            pl->m_player1->setPositionX(targetX);

            auto checkpoint = pl->createCheckpoint();
            pl->storeCheckpoint(checkpoint);

            pl->m_currentCheckpoint = checkpoint;

            pl->loadFromCheckpoint(checkpoint);
            pl->m_player1->loadFromCheckpoint(checkpoint->m_player1Checkpoint);

            // doesn't correctly respawn while frozen for some reason
            controls::unfreeze();

            pl->destroyPlayer(pl->m_player1, nullptr);
        }
    }


    void releaseJump()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->releaseButton(PlayerButton::Jump);
            }
        }
    }

    void resetLevel()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            pl->resetLevel();
        }
    }

    void freeze()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->lockPlayer();
            }
        }
    }

    void unfreeze()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->m_isLocked = false;
            }
        }
    }
} // namespace controls
